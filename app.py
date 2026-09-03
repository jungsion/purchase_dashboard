
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
    df = raw_df.copy()
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    data_cols = df.columns[0:]
    df = df[data_cols].copy()

    last_trn = df['TrnDate'].max()
    print(f"데이터 내 마지막 입고일: {last_trn.date()}\n")

    yearly = df.assign(Year=df['DueDate'].dt.year).groupby('Year').agg(
        건수      = ('DelayDays', 'size'),
        지연률    = ('IsDelay',   lambda s: round(s.mean()*100, 1)),
        평균지연일 = ('DelayDays', lambda s: round(s.mean(), 2)),
        지연건평균 = ('DelayDays', lambda s: round(s[s > 0].mean(), 2) if (s > 0).any() else 0),
        최대지연일 = ('DelayDays', 'max'),
    )

    # 미입고 건이 어느 연도 납기에 몰려 있는지
    # — 검증구간에서 빠진 '지연 후보'의 규모
    if len(open_orders):
        oo = open_orders.assign(Year=open_orders['DueDate'].dt.year)
        oo = oo[oo['DueDate'] <= last_trn]

    # 판단 가이드 출력
    p95_delay = df.loc[df['IsDelay'] == 1, 'DelayDays'].quantile(0.95)
    safe_cutoff = last_trn - pd.Timedelta(days=float(p95_delay))


RECENT_N = 10          # 최근 몇 건을 볼 것인가
EWM_HALFLIFE = 5       # EWMA 반감기(건 단위) — 최근 건에 가중

def make_past_stats(df, group_col, prefix):
    ev = df[[group_col, 'TrnDate', 'DelayDays', 'IsDelay']].sort_values('TrnDate').copy()

    g = ev.groupby(group_col)['DelayDays']

    ev['cnt']  = ev.groupby(group_col).cumcount() + 1          # 누적 납품 건수
    ev['sum']  = g.cumsum()                                    # 누적 지연일
    ev['dsum'] = ev.groupby(group_col)['IsDelay'].cumsum()     # 누적 지연 건수

    # [ver0.9 신규] 최근 N건 평균 / 지수가중 평균.
    #   각 '입고 이벤트' 시점의 값이므로, 아래 merge_asof가 발주일 직전 이벤트를 집어오면
    #   그 시점까지의 정보만 담기게 된다 → 누수 없음.
    ev['recent'] = g.transform(
        lambda s: s.rolling(RECENT_N, min_periods=1).mean()
    )
    ev['ewm'] = g.transform(
        lambda s: s.ewm(halflife=EWM_HALFLIFE).mean()
    )

    ev = ev.rename(columns={'TrnDate': 'd'})[
        [group_col, 'd', 'cnt', 'sum', 'dsum', 'recent', 'ewm']
    ].sort_values('d')

    left = df[[group_col, 'CrtDate']].rename(
        columns={'CrtDate': 'd'}
    ).sort_values('d')

    left['_i'] = left.index

    # 발주일보다 '앞선' 입고 실적만 매칭
    # (allow_exact_matches=False → 같은 날 입고분도 제외)
    m = pd.merge_asof(
        left,
        ev,
        on='d',
        by=group_col,
        direction='backward',
        allow_exact_matches=False
    ).set_index('_i').sort_index()

    out = pd.DataFrame(index=df.index)

    out[prefix + '_Cnt'] = m['cnt'].fillna(0).values
    # 과거 실적 건수(신뢰도)

    out[prefix + '_MeanDelay'] = (m['sum'] / m['cnt']).values
    # 과거 전체 평균 지연일

    out[prefix + '_DelayRate'] = (m['dsum'] / m['cnt']).values
    # 과거 지연 발생률

    out[prefix + '_Recent'] = m['recent'].values
    # [ver0.9] 최근 N건 평균

    out[prefix + '_Ewm'] = m['ewm'].values
    # [ver0.9] 지수가중 평균

    return out


    for gcol, pre in [
        ('Vndnr', 'Vnd'),
        ('Itnbr', 'Item'),
        ('ITCLS', 'Cls')
    ]:
        df = pd.concat([
            df,
            make_past_stats(df, gcol, pre)
        ], axis=1)

    def make_vendor_load(df, group_col='Vndnr'):
        n = len(df)
        load_cnt = np.zeros(n)
        load_qty = np.zeros(n)

        qty = df['OrdQty'].fillna(0).to_numpy(dtype=float)

        for _, idx in df.groupby(group_col).groups.items():
            pos = np.asarray(idx)                      # df는 RangeIndex라 위치와 동일
            t   = df['CrtDate'].to_numpy()[pos]        # 각 건의 발주 시점

            # 발주(열림) 쪽 누적
            crt = df['CrtDate'].to_numpy()[pos]
            o   = np.argsort(crt)
            crt_s = crt[o]
            qc  = np.concatenate([[0], np.cumsum(qty[pos][o])])

            opened_cnt = np.searchsorted(crt_s, t, side='right')
            opened_qty = qc[opened_cnt]

            # 입고(닫힘) 쪽 누적
            trn = df['TrnDate'].to_numpy()[pos]
            o2  = np.argsort(trn)
            trn_s = trn[o2]
            qc2 = np.concatenate([[0], np.cumsum(qty[pos][o2])])

            closed_cnt = np.searchsorted(trn_s, t, side='right')
            closed_qty = qc2[closed_cnt]

            load_cnt[pos] = opened_cnt - closed_cnt
            load_qty[pos] = opened_qty - closed_qty

        return load_cnt, load_qty


    df['VndOpenCnt'], df['VndOpenQty'] = make_vendor_load(df)

    # 업체 규모 차이를 보정 — 절대 건수보다 '평소 대비 얼마나 밀려 있나'가 신호에 가깝다
    vnd_mean_load = df.groupby('Vndnr')['VndOpenCnt'].transform('mean')
    df['VndLoadRatio'] = df['VndOpenCnt'] / vnd_mean_load.replace(0, np.nan)

    CUTOFF = pd.Timestamp('2026-01-01')     # ← 셀 3의 'safe_cutoff' 결과를 보고 필요시 앞당길 것

    df['Due_Crt_diff'] = (df['DueDate'] - df['CrtDate']).dt.days   # 발주~납기 여유일
    df['Lt_Gap']       = df['Due_Crt_diff'] - df['PurLt']          # 표준 리드타임 대비 여유
    df['Due_Week']     = df['DueDate'].dt.isocalendar().week.astype(int)

    # [ver0.9 수정] 순환 인코딩
    m, w = df['DueDate'].dt.month, df['DueDate'].dt.dayofweek

    df['Due_Month_sin'], df['Due_Month_cos'] = (
        np.sin(2*np.pi*m/12),
        np.cos(2*np.pi*m/12)
    )

    df['Due_Dow_sin'], df['Due_Dow_cos'] = (
        np.sin(2*np.pi*w/7),
        np.cos(2*np.pi*w/7)
    )

    FEATURES = [
        'OrdQty', 'PurLt', 'OrdLt', 'Due_Crt_diff', 'Lt_Gap', 'Due_Week',
        'Due_Month_sin', 'Due_Month_cos', 'Due_Dow_sin', 'Due_Dow_cos',

        'Vnd_Cnt', 'Vnd_MeanDelay', 'Vnd_DelayRate', 'Vnd_Recent', 'Vnd_Ewm',
        'Item_Cnt', 'Item_MeanDelay', 'Item_DelayRate', 'Item_Recent', 'Item_Ewm',
        'Cls_Cnt', 'Cls_MeanDelay', 'Cls_DelayRate', 'Cls_Recent', 'Cls_Ewm',

        'VndOpenCnt', 'VndOpenQty', 'VndLoadRatio',                    # [ver0.9 신규]
    ]

    CAT_FEATURES = ['Vndnr', 'ITTYP', 'ITCLS']   # LightGBM 전용 (품번은 카디널리티가 커서 제외)

    # 과거 실적이 없는 건(첫 거래 등)은 '학습구간' 평균으로 대체 — 검증구간 정보를 쓰면 안 된다
    for pre in ['Vnd', 'Item', 'Cls']:
        for suf in ['_MeanDelay', '_DelayRate', '_Recent', '_Ewm']:
            fill = df.loc[df['DueDate'] < CUTOFF, pre + suf].mean()
            df[pre + suf] = df[pre + suf].fillna(fill)

    df['VndLoadRatio'] = df['VndLoadRatio'].fillna(1.0)

    d = df.dropna(subset=FEATURES + ['DelayDays']).copy()

    for c in CAT_FEATURES:
        d[c] = d[c].astype(str).astype('category')

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
