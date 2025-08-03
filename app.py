import streamlit as st
import numpy as np
import pickle
import time

# Load the trained model
model = pickle.load(open('linear_model.pkl', 'rb'))

# Page Config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Header
st.title("❤️ Heart Disease Prediction")
st.caption("A simple ML-powered tool to predict the risk of Heart Disease.")

# Use Columns to make the layout attractive
st.subheader("📝 Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', 1, 120, 50)
    sex = st.radio('Sex', ['Male', 'Female'])
    cp = st.selectbox('Chest Pain Type (0-3)', (0, 1, 2, 3))
    trestbps = st.slider('Resting Blood Pressure (mmHg)', 80, 200, 120)
    chol = st.slider('Serum Cholesterol (mg/dl)', 100, 600, 200)
    fbs = st.radio('Fasting Blood Sugar > 120 mg/dl', (0, 1))

with col2:
    restecg = st.selectbox('Resting ECG (0-2)', (0, 1, 2))
    thalach = st.slider('Max Heart Rate Achieved', 60, 220, 150)
    exang = st.radio('Exercise Induced Angina', (0, 1))
    oldpeak = st.number_input('Oldpeak (ST Depression)', 0.0, 10.0, 1.0)
    slope = st.selectbox('Slope of Peak Exercise ST', (0, 1, 2))
    ca = st.selectbox('Major Vessels (0-3)', (0, 1, 2, 3))
    thal = st.selectbox('Thalassemia (1=Normal, 2=Fixed, 3=Reversible)', (1, 2, 3))

sex_val = 1 if sex == 'Male' else 0

# Predict Button
st.markdown("---")
st.subheader("🔍 Prediction")

if st.button("✨ Predict Heart Disease Risk"):
    with st.spinner("🔎 Analyzing patient data..."):
        time.sleep(1.5)  # Simulate processing time
        
        # Prepare input data
        input_data = (age, sex_val, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal)
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data_as_numpy_array)

        # Show results with style
        if prediction[0] == 0:
            st.success("✅ The Person does **NOT** have Heart Disease.")
        else:
            st.error("⚠️ The Person **HAS** Heart Disease.")


# Info Section
with st.expander("ℹ️ Learn About the Parameters"):
    st.write("""
    **Chest Pain Type (cp):**  
    0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic  

    **fbs:** 1 = fasting blood sugar > 120 mg/dl  

    **restecg:** 0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy  

    **thal:** 1 = normal, 2 = fixed defect, 3 = reversible defect  
    """)

