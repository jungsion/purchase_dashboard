{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ec9298bc-cb9e-4e4d-8910-08481a342958",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-09-03 14:20:25.518 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:25.519 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.178 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\gicon\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-09-03 14:20:26.178 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.179 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.180 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.181 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.181 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.182 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:TensorFlow GPU support is not available on native Windows for TensorFlow >= 2.11. Even if CUDA/cuDNN are installed, GPU will not be used. Please use WSL2 or the TensorFlow-DirectML plugin.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-09-03 14:20:26.386 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.387 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.387 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.388 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.389 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.389 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.390 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-09-03 14:20:26.390 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import io\n",
    "import joblib\n",
    "import keras\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import streamlit as st\n",
    "\n",
    "st.set_page_config(page_title=\"배송 지연 예측 대시보드\", layout=\"wide\")\n",
    "st.title(\"🚚 배송 지연 예측 시스템\")\n",
    "\n",
    "# 모델 및 전처리 파일 로드\n",
    "@st.cache_resource\n",
    "def load_assets():\n",
    "    clf_model = keras.models.load_model(\"delivery_clf_0.9.keras\")\n",
    "    reg_model = keras.models.load_model(\"delivery_reg_0.9.keras\")\n",
    "    meta = joblib.load(\"delivery_meta_0.9.joblib\")\n",
    "    return clf_model, reg_model, meta\n",
    "\n",
    "try:\n",
    "    clf, reg, meta = load_assets()\n",
    "    scaler = meta[\"scaler\"]\n",
    "    features = meta[\"features\"]\n",
    "except Exception as e:\n",
    "    st.error(f\"모델 로드 실패: {e}\")\n",
    "    st.stop()\n",
    "\n",
    "uploaded_file = st.file_uploader(\"예측할 엑셀 파일을 업로드하세요\", type=[\"xlsx\", \"xls\"])\n",
    "\n",
    "if uploaded_file is not None:\n",
    "    df = pd.read_excel(uploaded_file)\n",
    "    st.subheader(\"📄 업로드 데이터 확인\")\n",
    "    st.dataframe(df.head(), use_container_width=True)\n",
    "\n",
    "    if st.button(\"예측 실행\"):\n",
    "        # 1. 필요 피처 존재 여부 검증\n",
    "        missing_features = [col for col in features if col not in df.columns]\n",
    "        \n",
    "        if missing_features:\n",
    "            st.error(f\"⚠️ 업로드된 엑셀 파일에 필요한 컬럼이 없습니다!\")\n",
    "            st.warning(f\"**누락된 컬럼 목록:** {missing_features}\")\n",
    "            st.info(f\"**모델 필요 컬럼 전체:** {features}\")\n",
    "        else:\n",
    "            with st.spinner(\"예측 중입니다...\"):\n",
    "                # 2. 피처 순서 맞추기 및 스케일링\n",
    "                X = df[features]\n",
    "                X_scaled = scaler.transform(X)\n",
    "\n",
    "                # 3. 분류 및 회귀 모델 추론\n",
    "                p_delay = clf.predict(X_scaled)\n",
    "                reg_output = reg.predict(X_scaled)\n",
    "\n",
    "                # 4. 후처리 연산 (P(delay) * expm1(reg_output))\n",
    "                pred_delay_days = p_delay.flatten() * np.expm1(reg_output.flatten())\n",
    "\n",
    "                # 5. 결과 테이블 생성\n",
    "                result_df = df.copy()\n",
    "                result_df[\"예측 지연 일수\"] = np.round(pred_delay_days, 2)\n",
    "\n",
    "            st.success(\"예측 완료!\")\n",
    "            st.subheader(\"📋 예측 결과\")\n",
    "            st.dataframe(result_df, use_container_width=True)\n",
    "\n",
    "            # 6. 엑셀 다운로드 버퍼 생성 및 버튼 추가\n",
    "            buffer = io.BytesIO()\n",
    "            with pd.ExcelWriter(buffer, engine=\"openpyxl\") as writer:\n",
    "                result_df.to_excel(writer, index=False, sheet_name=\"예측결과\")\n",
    "\n",
    "            st.download_button(\n",
    "                label=\"📥 예측 결과 엑셀 다운로드\",\n",
    "                data=buffer.getvalue(),\n",
    "                file_name=\"delivery_predictions.xlsx\",\n",
    "                mime=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\",\n",
    "            )"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
