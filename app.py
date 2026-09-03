
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="발주 데이터 전처리 시스템", layout="wide")
st.title("🛠️ 발주 데이터 전처리 파이프라인")

# ---------------------------------------------------------
# 1. 메타 데이터 로드 (과거 이력 및 기준값)
# ---------------------------------------------------------
@st.cache_resource
def load_meta():
    try:
        meta = joblib.load("delivery_meta_0.9.joblib")
        return meta
    except Exception as e:
        return None

meta = load_meta()
if meta:
    RECENT_N = meta.get("RECENT_N", 5)
    EWM_HALFLIFE = meta.get("EWM_HALFLIFE", 3.0)
    fill_values = meta.get("fill_values", {})
    df_history = meta.get("df_history", None)
else:
    # 메타 파일이 없을 경우 기본 파라미터 적용
    RECENT_N = 5
    EWM_HALFLIFE = 3.0
    fill_values = {}
    df_history = None
    st.info("💡 메타 파일(delivery_meta_0.9.joblib) 없이 기본 전처리 규칙으로 동작합니다.")


# ---------------------------------------------------------
# 2. 전처리 핵심 로직 함수
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
    """Raw 엑셀 데이터의 1행을 컬럼명으로 지정 후 전처리 파이프라인 수행"""
    df = raw_df.copy()

    # 1) 컬럼명 먼저 지정 (1번째 행의 값을 헤더로 승격)
    df.columns = df.iloc[0].values
    df = df[1:].reset_index(drop=True)

    # 2) 7번째 컬럼부터 데이터 슬라이싱 (필요에 따라 주석 제어)
    if len(df.columns) >= 7:
        df = df[df.columns[6:]].copy()

    # 3) 날짜 및 수치형 컬럼 변환
    for c in ['CrtDate', 'DueDate']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c].astype(str), format='%Y%m%d', errors='coerce')

    # 필수 날짜 결측치 제거
    df = df.dropna(subset=['CrtDate', 'DueDate']).copy()

    for c in ['OrdQty', 'PurLt', 'OrdLt']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # 4) 날짜 기반 기본 파생변수
    df['Due_Crt_diff'] = (df['DueDate'] - df['CrtDate']).dt.days
    df['Lt_Gap'] = df['Due_Crt_diff'] - df['PurLt']
    df['Due_Week'] = df['DueDate'].dt.isocalendar().week.astype(int)

    m = df['DueDate'].dt.month
    w = df['DueDate'].dt.dayofweek
    df['Due_Month_sin'] = np.sin(2 * np.pi * m / 12)
    df['Due_Month_cos'] = np.cos(2 * np.pi * m / 12)
    df['Due_Dow_sin'] = np.sin(2 * np.pi * w / 7)
    df['Due_Dow_cos'] = np.cos(2 * np.pi * w / 7)

    # 5) 업체 부하량 피처 생성
    load_df = make_vendor_load(df)
    df = pd.concat([df, load_df], axis=1)

    # 6) 과거 입고 실적 기반 통계 생성 (history_df 존재 시)
    if history_df is not None and not history_df.empty:
        for gcol, pre in [('Vndnr', 'Vnd'), ('Itnbr', 'Item'), ('ITCLS', 'Cls')]:
            if gcol in df.columns and gcol in history_df.columns:
                stats = make_past_stats(df, history_df, gcol, pre, RECENT_N, EWM_HALFLIFE)
                df = pd.concat([df, stats], axis=1)

    # 7) 과거 통계 결측치 채우기
    for pre in ['Vnd', 'Item', 'Cls']:
        for suf in ['_MeanDelay', '_DelayRate', '_Recent', '_Ewm']:
            col_name = pre + suf
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna(fill_values.get(col_name, 0.0))

    return df


# ---------------------------------------------------------
# 3. Streamlit 화면 구성
# ---------------------------------------------------------
uploaded_file = st.file_uploader("전처리할 엑셀 파일(Raw Data)을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file is not None:
    # 엑셀 로드 (헤더 없이 업로드)
    raw_df = pd.read_excel(uploaded_file, header=None)

    st.subheader("1. 업로드 원본 데이터 (1행 컬럼 변환 전)")
    st.dataframe(raw_df.head(), use_container_width=True)

    if st.button("⚙️ 전처리 실행"):
        with st.spinner("데이터 전처리 변환 진행 중..."):
            try:
                # 전처리 수행
                processed_df = pipeline_preprocess(raw_df, df_history)

                st.success("✅ 전처리가 성공적으로 완료되었습니다!")

                # 데이터 정보 요약
                st.markdown(f"**총 행 수:** `{len(processed_df)}` 개 / **총 컬럼 수:** `{len(processed_df.columns)}` 개")

                # 전처리 완료 데이터 표 출력
                st.subheader("2. 전처리 완료 데이터 표")
                st.dataframe(processed_df, use_container_width=True)

                # 전처리 데이터 다운로드 버튼
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    processed_df.to_excel(writer, index=False, sheet_name="전처리데이터")

                st.download_button(
                    label="📥 전처리 완료 데이터 엑셀 다운로드",
                    data=buffer.getvalue(),
                    file_name="processed_delivery_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

            except Exception as e:
                st.error(f"전처리 과정에서 오류가 발생했습니다: {e}")
