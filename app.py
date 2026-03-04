import streamlit as st
import joblib
import numpy as np
import random
import pandas as pd
import time
import os
import base64

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Smart Climate Monitoring System",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model, accuracy = joblib.load("climate_model.pkl")

# ------------------ TITLE ------------------
st.title("🌍 Smart Climate Monitoring & Early Warning System")
st.markdown("### AI-Based Real-Time Climate Risk Prediction")

# ------------------ SIDEBAR ------------------
st.sidebar.header("📊 Model Information")
st.sidebar.write("**Algorithm:** Random Forest")
st.sidebar.write(f"**Model Accuracy:** {accuracy*100:.2f}%")

# Show Feature Importance Image
if os.path.exists("feature_importance.png"):
    st.sidebar.subheader("Feature Importance")
    st.sidebar.image("feature_importance.png")

# Show Confusion Matrix
if os.path.exists("confusion_matrix.png"):
    st.sidebar.subheader("Confusion Matrix")
    st.sidebar.image("confusion_matrix.png")

# ------------------ METRIC COLUMNS ------------------
col1, col2, col3 = st.columns(3)

# ------------------ SESSION STORAGE ------------------
if "data_log" not in st.session_state:
    st.session_state.data_log = []

# ------------------ GENERATE LIVE DATA ------------------
temperature = random.uniform(15, 45)
humidity = random.uniform(20, 95)
aqi = random.uniform(30, 350)

input_data = np.array([[temperature, humidity, aqi]])
prediction = model.predict(input_data)[0]

# ------------------ DISPLAY METRICS ------------------
with col1:
    st.metric("🌡 Temperature (°C)", f"{temperature:.2f}")

with col2:
    st.metric("💧 Humidity (%)", f"{humidity:.2f}")

with col3:
    st.metric("🌫 AQI", f"{aqi:.2f}")

# ------------------ RISK DISPLAY ------------------
if prediction == 0:
    st.success("🟢 Risk Level: LOW")

elif prediction == 1:
    st.warning("🟡 Risk Level: MEDIUM")

else:
    st.error("🔴 Risk Level: HIGH")

    # ----------- AUTO ALARM -----------
    if os.path.exists("alarm.wav"):
        with open("alarm.wav", "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()

            audio_html = f"""
            <audio autoplay>
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
            """

            st.markdown(audio_html, unsafe_allow_html=True)
    else:
        st.error("Alarm file not found!")

# ------------------ STORE DATA ------------------
st.session_state.data_log.append({
    "Temperature": temperature,
    "Humidity": humidity,
    "AQI": aqi
})

df = pd.DataFrame(st.session_state.data_log)

# ------------------ LIVE GRAPH ------------------
st.subheader("📈 Live Climate Trend")
st.line_chart(df)

# ------------------ DATA TABLE ------------------
with st.expander("📄 View Historical Data"):
    st.dataframe(df)

# ------------------ AUTO REFRESH ------------------
time.sleep(5)
st.rerun()