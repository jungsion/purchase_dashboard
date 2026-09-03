
import io
import joblib
import keras
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="배송 지연 예측 시스템", layout="wide")
st.title("🚚 배송 지연 예측 대시보드")

# ---------------------------------------------------------
# 1. 모델 및 메타 정보 로드
# ---------------------------------------------------------
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

    # 전처리 하이퍼파라미터 및 과거 데이터 로드
    RECENT_N = meta.get("RECENT_N", 5)
    EWM_HALFLIFE = meta.get("EWM_HALFLIFE", 3.0)
    fill_values = meta.get("fill_values", {})
    df_history = meta.get("df_history", None)
except Exception as e:
    st.error(f"모델 및 메타 자원 로드 실패: {e}")
    st.stop()


# ---------------------------------------------------------
# 2. 전처리 핵심 함수 모듈
# ---------------------------------------------------------
def make_past_stats(target_df, history_df, group_col, prefix, recent_n=5, ewm_halflife=3.0):
    """과거 입고 실적 기반 누적 통계 피처 생성 (merge_asof 기반)"""
    combined = pd.concat([history_df, target_df], axis=0, ignore_index=True)

    ev = combined[[group_col, 'TrnDate', 'DelayDays', 'IsDelay']].dropna(subset=['TrnDate']).sort_values('TrnDate').copy()
    g = ev.groupby(group_col)['DelayDays']

    ev['cnt'] = ev.groupby(group_col).cumcount() + 1
    ev['sum'] = g.cumsum()
    ev['dsum'] = ev.groupby(group_col)['IsDelay'].cumsum()
    ev['recent'] = g.transform(lambda s: s.rolling(recent_n, min_periods=1).mean())
    ev['ewm'] = g.transform(lambda s: s.ewm(halflife=ewm_halflife).mean())

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

def make_vendor_load(df):
    """발주 등록 시점 기준 업체별 미입고 부하량 산출"""
    c_dates = df['CrtDate'].to_numpy(dtype='datetime64[D]')
    d_dates = df['DueDate'].to_numpy(dtype='datetime64[D]')
    vnds = df['Vndnr'].astype(str).to_numpy()
    qtys = df['OrdQty'].fillna(0).to_numpy(dtype=float)

    n = len(df)
    load_cnt = np.zeros(n, dtype=float)
    load_qty = np.zeros(n, dtype=float)

    for i in range(n):
        c_i = c_dates[i]
        v_i = vnds[i]
        cond = (c_dates < c_i) & (d_dates >= c_i) & (vnds == v_i)
        load_cnt[i] = cond.sum()
        load_qty[i] = qtys[cond].sum()

    out = pd.DataFrame(index=df.index)
    out['VndOpenCnt'] = load_cnt
    out['VndOpenQty'] = load_qty
    out['VndLoadRatio'] = load_qty / (qtys + 1e-5)
    return out

def pipeline_preprocess(raw_df, history_df=None):
    """Raw 엑셀 데이터의 1행을 헤더로 지정하고 전체 전처리 파이프라인 수행"""
    df = raw_df.copy()

    # [핵심] 1번째 row(index 0)의 값을 컬럼명(Header)으로 지정
    df.columns = df.iloc[0].values
    df = df[1:].reset_index(drop=True)

    # 날짜 및 수치형 컬럼 변환
    for c in ['CrtDate', 'DueDate']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str), format='%Y%m%d', errors='coerce')

    df = df.dropna(subset=['CrtDate', 'DueDate']).copy()

    for c in ['OrdQty', 'PurLt', 'OrdLt']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 파생변수 생성
    df['Due_Crt_diff'] = (df['DueDate'] - df['CrtDate']).dt.days
    df['Lt_Gap'] = df['Due_Crt_diff'] - df['PurLt']
    df['Due_Week'] = df['DueDate'].dt.isocalendar().week.astype(int)

    m = df['DueDate'].dt.month
    w = df['DueDate'].dt.dayofweek
    df['Due_Month_sin'] = np.sin(2 * np.pi * m / 12)
    df['Due_Month_cos'] = np.cos(2 * np.pi * m / 12)
    df['Due_Dow_sin'] = np.sin(2 * np.pi * w / 7)
    df['Due_Dow_cos'] = np.cos(2 * np.pi * w / 7)

    # 업체 부하량 피처 생성
    load_df = make_vendor_load(df)
    df = pd.concat([df, load_df], axis=1)

    # 과거 입고 실적 기반 피처 생성 (history_df 존재 시)
    if history_df is not None and not history_df.empty:
        for gcol, pre in [('Vndnr', 'Vnd'), ('Itnbr', 'Item'), ('ITCLS', 'Cls')]:
            if gcol in df.columns and gcol in history_df.columns:
                stats = make_past_stats(df, history_df, gcol, pre, RECENT_N, EWM_HALFLIFE)
                df = pd.concat([df, stats], axis=1)

    # 결측치 채우기
    for pre in ['Vnd', 'Item', 'Cls']:
        for suf in ['_MeanDelay', '_DelayRate', '_Recent', '_Ewm']:
            col_name = pre + suf
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna(fill_values.get(col_name, 0.0))

    return df


# ---------------------------------------------------------
# 3. Streamlit 화면 구성
# ---------------------------------------------------------
uploaded_file = st.file_uploader("발주 Raw Data 엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 엑셀 파일 로드 (header=None으로 일단 읽어옵니다)
    raw_df = pd.read_excel(uploaded_file, header=None)
    st.subheader("📄 업로드 원본 데이터 (1행 헤더 변환 전)")
    st.dataframe(raw_df.head(), use_container_width=True)

    if st.button("🚀 전처리 및 예측 실행"):
        with st.spinner("1행 컬럼 변환 및 전처리 파이프라인 수행 중..."):
            try:
                # 전처리 적용 (1행 컬럼 변환 포함)
                processed_df = pipeline_preprocess(raw_df, df_history)

                # 모델 예측용 피처 추출 및 스케일링
                X = processed_df[features].fillna(0.0).astype(float)
                X_scaled = scaler.transform(X)

                # 예측 수행
                p_delay = clf.predict(X_scaled).flatten()
                reg_output = reg.predict(X_scaled).flatten()

                # 지연 일수 계산
                pred_delay_days = p_delay * np.expm1(reg_output)

                # 결과 데이터 프레임 생성
                result_df = processed_df.copy()
                result_df["지연 위험도"] = np.where(p_delay >= 0.5, "⚠️ 지연 위험", "✅ 정상")
                result_df["예측 지연 일수"] = np.round(pred_delay_days, 2)

                st.success("전처리 및 예측이 완료되었습니다!")
                st.subheader("📊 예측 결과 요약")

                # 주요 결과 컬럼 출력
                display_cols = [c for c in ['Vndnr', 'Itnbr', 'CrtDate', 'DueDate', '지연 위험도', '예측 지연 일수'] if c in result_df.columns]
                st.dataframe(result_df[display_cols], use_container_width=True)

                # 결과 엑셀 다운로드
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    result_df.to_excel(writer, index=False, sheet_name="예측결과")

                st.download_button(
                    label="📥 전체 예측 결과 다운로드 (엑셀)",
                    data=buffer.getvalue(),
                    file_name="delivery_predictions.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"전처리 및 예측 처리 중 오류 발생: {e}")
