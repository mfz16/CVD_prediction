import streamlit as st
import numpy as np
import pandas as pd
import joblib

#from utils import wrangle
import pickle
import sys
#import path
import os


#dir = path.Path(__file__).abspath()
#sys.path.append(dir.parent.parent)

# load model
#path_to_model = './CVD/stacking_model.pkl'
path_to_model = os.path.join("./CVD/stacking_model.pkl")


with open(path_to_model, 'rb') as file:
    model = pickle.load(file)


# Function to make predictions
def predict(model, features):
    prediction = model.predict(features)
    return prediction

# Main function to run the Streamlit app
def main():
    st.title("Cardiovascular Disease Prediction")

    # Input fields for patient information
    age = st.slider("Age", min_value=1, max_value=100, value=50)
    sex = st.radio("Sex", options=["Male (M)", "Female (F)"])
    chest_pain_type = st.selectbox("Chest Pain Type", ["Typical Angina (TA)", "Atypical Angina (ATA)", "Non-Anginal Pain (NAP)", "Asymptomatic (ASY)"])
    resting_blood_pressure = st.slider("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200, value=120)
    cholesterol = st.slider("Cholesterol (mm/dl)", min_value=50, max_value=500, value=200)
    fasting_blood_sugar = st.radio("Fasting Blood Sugar (mg/dl)", options=["≤ 120 (0)", "> 120 (1)"])
    resting_ecg_results = st.selectbox("Resting Electrocardiogram Results", ["Normal", "ST-T Wave Abnormality (ST)", "Left Ventricular Hypertrophy (LVH)"])
    max_heart_rate = st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=202, value=150)
    exercise_induced_angina = st.radio("Exercise-Induced Angina", options=["Yes (Y)", "No (N)"])
    oldpeak = st.slider("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=0.0)
    st_slope = st.selectbox("ST Slope", ["Upsloping (Up)", "Flat", "Downsloping (Down)"])

    # Convert categorical features to numerical values
    sex_encoded = 1 if sex.startswith("M") else 0
    chest_pain_type_mapping = {"Typical Angina (TA)": 0, "Atypical Angina (ATA)": 1, "Non-Anginal Pain (NAP)": 2, "Asymptomatic (ASY)": 3}
    chest_pain_type_encoded = chest_pain_type_mapping[chest_pain_type]
    fasting_blood_sugar_encoded = 1 if fasting_blood_sugar.startswith(">") else 0
    resting_ecg_results_mapping = {"Normal": 0, "ST-T Wave Abnormality (ST)": 1, "Left Ventricular Hypertrophy (LVH)": 2}
    resting_ecg_results_encoded = resting_ecg_results_mapping[resting_ecg_results]
    exercise_induced_angina_encoded = 1 if exercise_induced_angina.startswith("Y") else 0
    st_slope_mapping = {"Upsloping (Up)": 0, "Flat": 1, "Downsloping (Down)": 2}
    st_slope_encoded = st_slope_mapping[st_slope]

    # Collect features into a DataFrame
    features = pd.DataFrame({
        'age': [age],
        'sex': [sex_encoded],
        'chest pain type': [chest_pain_type_encoded],
        'resting bp s': [resting_blood_pressure],
        'cholesterol': [cholesterol],
        'fasting blood sugar': [fasting_blood_sugar_encoded],
        'resting ecg': [resting_ecg_results_encoded],
        'max heart rate': [max_heart_rate],
        'exercise angina': [exercise_induced_angina_encoded],
        'oldpeak': [oldpeak],
        'ST slope': [st_slope_encoded]
    })

    # Load the trained model
    #model = joblib.load('CVD\stacking_model.pkl')

    # Make prediction
    if st.button("Predict"):
        prediction = predict(model, features)
        if prediction[0] == 0:
            st.write("The patient is predicted not to have cardiovascular disease.")
        else:
            st.write("The patient is predicted to have cardiovascular disease.")

if __name__ == "__main__":
    main()
