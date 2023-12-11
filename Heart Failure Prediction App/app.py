import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Heart Failure Prediction App

This app predicts the likelihood of **Heart Failure**!
""")

st.sidebar.header('User Input Features')

# Load the heart failure dataset
heart_data = pd.read_csv('heart.csv')

# No need for encoding as the data is already numeric

# Collects user input features into dataframe
def user_input_features():
    age = st.sidebar.slider('Age', 0, 120, int(heart_data['age'].mean()))
    sex = st.sidebar.selectbox('Sex | Male - 1 Female - 0', [0, 1])
    cp = st.sidebar.slider('Chest Pain Type (cp) | (1: typical angina / 2: atypical angina / 3: non-anginal pain / 4: asymptomatic)', int(heart_data['cp'].min()), int(heart_data['cp'].max()), int(heart_data['cp'].mean()))
    trtbps = st.sidebar.slider('Resting Blood Pressure (trtbps)', 80, 250,  int(heart_data['trtbps'].mean()))
    chol = st.sidebar.slider('Cholesterol (chol)', int(heart_data['chol'].min()), int(heart_data['chol'].max()), int(heart_data['chol'].mean()))
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs) (1 = true; 0 = false)', [0, 1])
    restecg = st.sidebar.slider('Resting Electrocardiographic Results (restecg) | (0: normal / 1: having ST-T wave abnormality / 2: showing probable or definite left ventricular hypertrophy by Estes" criteria)', int(heart_data['restecg'].min()), int(heart_data['restecg'].max()), int(heart_data['restecg'].mean()))
    thalachh = st.sidebar.slider('Maximum Heart Rate Achieved (thalachh)', int(heart_data['thalachh'].min()), int(heart_data['thalachh'].max()), int(heart_data['thalachh'].mean()))
    exng = st.sidebar.selectbox('Exercise Induced Angina (exng)', [0, 1])
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest (oldpeak)', float(heart_data['oldpeak'].min()), float(heart_data['oldpeak'].max()), float(heart_data['oldpeak'].mean()))
    slp = st.sidebar.slider('Slope of the Peak Exercise ST Segment (slp)', int(heart_data['slp'].min()), int(heart_data['slp'].max()), int(heart_data['slp'].mean()))
    caa = st.sidebar.slider('Number of Major Vessels (caa)', int(heart_data['caa'].min()), int(heart_data['caa'].max()), int(heart_data['caa'].mean()))
    thall = st.sidebar.slider('Thal Rate (thall)', int(heart_data['thall'].min()), int(heart_data['thall'].max()), int(heart_data['thall'].mean()))
    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trtbps': trtbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalachh': thalachh,
            'exng': exng,
            'oldpeak': oldpeak,
            'slp': slp,
            'caa': caa,
            'thall': thall}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Displays the user input features
st.subheader('User Input features')
st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('heart_failure_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)

st.subheader('Prediction')
st.write(f'The predicted output is: {prediction[0]}')
st.write('"0" = less chance of heart attack "1" = more chance of heart attack')

st.subheader('Prediction Probability')
st.write(prediction_proba)
