import io
import joblib
import keras
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="배송 지연 예측 대시보드", layout="wide")
st.title("🚚 배송 지연 예측 시스템")

# 1. assets 로드
@st.cache_resource
def load_assets():
    clf_model = keras.models.load_model("delivery_clf_0.9.keras")
    reg_model = keras.models.load_model("delivery_reg_0.9.keras")
    meta = joblib.load("delivery_meta_0.9.joblib")
    return clf_model, reg_model, meta

try:
    clf, reg, meta = load_assets()
    scaler = meta["scaler"]
    features = meta["features"]
    df_history = meta["df_history"]
    fill_values = meta["fill_values"]
    RECENT_N = meta["RECENT_N"]
    EWM_HALFLIFE = meta["EWM_HALFLIFE"]
except Exception as e:
    st.error(f"모델 및 전처리 데이터 로드 실패: {e}")
    st.stop()

# 2. 전처리 함수 모듈화
def preprocess_raw_data(raw_df, history_df):
    """사용자가 업로드한 Raw 엑셀을 모델 파이프라인에 맞게 전처리"""
    df = raw_df.copy()

    # 헤더 정리 (skiprows=1 및 컬럼 인덱싱 처리)
    if 'CrtDate' not in df.columns and len(df) > 1:
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        if len(df.columns) >= 7:
            df = df[df.columns[6:]].copy()

    # 날짜 파싱
    date_cols = ['CrtDate', 'DueDate']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col].astype(str), format='%Y%m%d', errors='coerce')

    df = df.dropna(subset=['CrtDate', 'DueDate']).copy()

    # 수치형 변환
    for c in ['OrdQty', 'PurLt', 'OrdLt']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 파생 변수 생성
    df['Due_Crt_diff'] = (df['DueDate'] - df['CrtDate']).dt.days
    df['Lt_Gap'] = df['Due_Crt_diff'] - df['PurLt']
    df['Due_Week'] = df['DueDate'].dt.isocalendar().week.astype(int)

    m, w = df['DueDate'].dt.month, df['DueDate'].dt.dayofweek
    df['Due_Month_sin'] = np.sin(2 * np.pi * m / 12)
    df['Due_Month_cos'] = np.cos(2 * np.pi * m / 12)
    df['Due_Dow_sin'] = np.sin(2 * np.pi * w / 7)
    df['Due_Dow_cos'] = np.cos(2 * np.pi * w / 7)

    # 과거 실적 집계 (과거 DB + 신규 데이터 결합)
    combined = pd.concat([history_df, df], axis=0, ignore_index=True)

    def calc_stats(target_df, group_col, prefix):
        ev = combined[[group_col, 'TrnDate', 'DelayDays', 'IsDelay']].dropna(subset=['TrnDate']).sort_values('TrnDate').copy()
        g = ev.groupby(group_col)['DelayDays']

        ev['cnt'] = ev.groupby(group_col).cumcount() + 1
        ev['sum'] = g.cumsum()
        ev['dsum'] = ev.groupby(group_col)['IsDelay'].cumsum()
        ev['recent'] = g.transform(lambda s: s.rolling(RECENT_N, min_periods=1).mean())
        ev['ewm'] = g.transform(lambda s: s.ewm(halflife=EWM_HALFLIFE).mean())

        ev = ev.rename(columns={'TrnDate': 'd'})[[group_col, 'd', 'cnt', 'sum', 'dsum', 'recent', 'ewm']].sort_values('d')

        left = target_df[[group_col, 'CrtDate']].rename(columns={'CrtDate': 'd'}).sort_values('d')
        left['_i'] = left.index

        m_df = pd.merge_asof(
            left, ev, on='d', by=group_col, direction='backward', allow_exact_matches=False
        ).set_index('_i').sort_index()

        out = pd.DataFrame(index=target_df.index)
        out[prefix + '_Cnt'] = m_df['cnt'].fillna(0).values
        out[prefix + '_MeanDelay'] = (m_df['sum'] / m_df['cnt']).values
        out[prefix + '_DelayRate'] = (m_df['dsum'] / m_df['cnt']).values
        out[prefix + '_Recent'] = m_df['recent'].values
        out[prefix + '_Ewm'] = m_df['ewm'].values
        return out

    for gcol, pre in [('Vndnr', 'Vnd'), ('Itnbr', 'Item'), ('ITCLS', 'Cls')]:
        if gcol in df.columns:
            stats = calc_stats(df, gcol, pre)
            df = pd.concat([df, stats], axis=1)

    # 업체 부하량 피처 (VndOpenCnt, VndOpenQty, VndLoadRatio)
    qty = df['OrdQty'].fillna(0).to_numpy(dtype=float)
    load_cnt, load_qty = np.zeros(len(df)), np.zeros(len(df))

    # 단순화된 오픈 수량 할당
    df['VndOpenCnt'] = load_cnt
    df['VndOpenQty'] = load_qty
    df['VndLoadRatio'] = 1.0

    # 과거 실적 결측치 대체
    for pre in ['Vnd', 'Item', 'Cls']:
        for suf in ['_MeanDelay', '_DelayRate', '_Recent', '_Ewm']:
            col_name = pre + suf
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna(fill_values.get(col_name, 0))

    return df

# 3. UI 구성
uploaded_file = st.file_uploader("원본 납기 엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file is not None:
    raw_df = pd.read_excel(uploaded_file)
    st.subheader("📄 업로드 원본 데이터 확인")
    st.dataframe(raw_df.head(), use_container_width=True)

    if st.button("예측 실행"):
        with st.spinner("전처리 및 예측 연산 수행 중..."):
            try:
                # 데이터 자동으로 파이프라인 전처리
                processed_df = preprocess_raw_data(raw_df, df_history)

                # 피처 추출 및 스케일링
                X = processed_df[features].astype(float)
                X_scaled = scaler.transform(X)

                # 추론
                p_delay = clf.predict(X_scaled)
                reg_output = reg.predict(X_scaled)
                pred_delay_days = p_delay.flatten() * np.expm1(reg_output.flatten())

                # 결과 가공
                result_df = processed_df.copy()
                result_df["예측 지연 일수"] = np.round(pred_delay_days, 2)

                st.success("예측이 성공적으로 완료되었습니다!")
                st.subheader("📋 예측 결과")
                st.dataframe(result_df[['Vndnr', 'Itnbr', 'CrtDate', 'DueDate', '예측 지연 일수']], use_container_width=True)

                # 엑셀 다운로드
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    result_df.to_excel(writer, index=False, sheet_name="예측결과")

                st.download_button(
                    label="📥 전체 예측 결과 엑셀 다운로드",
                    data=buffer.getvalue(),
                    file_name="delivery_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"전처리 및 예측 과정에서 오류가 발생했습니다: {e}")
