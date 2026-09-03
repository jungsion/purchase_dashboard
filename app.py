import io
import joblib
import keras
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="배송 지연 예측 대시보드", layout="wide")
st.title("🚚 배송 지연 예측 시스템")

# 모델 및 전처리 파일 로드
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
except Exception as e:
    st.error(f"모델 로드 실패: {e}")
    st.stop()

uploaded_file = st.file_uploader("예측할 엑셀 파일을 업로드하세요", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.subheader("📄 업로드 데이터 확인")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("예측 실행"):
        with st.spinner("예측 중입니다..."):
            # 1. 피처 순서 맞추기 및 스케일링
            X = df[features]
            X_scaled = scaler.transform(X)

            # 2. 분류 및 회귀 모델 추론
            p_delay = clf.predict(X_scaled)
            reg_output = reg.predict(X_scaled)

            # 3. 후처리 연산 (P(delay) * expm1(reg_output))
            pred_delay_days = p_delay.flatten() * np.expm1(reg_output.flatten())

            # 4. 결과 테이블 생성
            result_df = df.copy()
            result_df["예측 지연 일수"] = np.round(pred_delay_days, 2)

        st.success("예측 완료!")
        st.subheader("📋 예측 결과")
        st.dataframe(result_df, use_container_width=True)

        # 5. 엑셀 다운로드 버퍼 생성 및 버튼 추가
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=False, sheet_name="예측결과")

        st.download_button(
            label="📥 예측 결과 엑셀 다운로드",
            data=buffer.getvalue(),
            file_name="delivery_predictions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
