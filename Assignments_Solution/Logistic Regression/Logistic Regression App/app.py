import streamlit as st
import joblib
import numpy as np
import pandas as pd

# load model
model = joblib.load("titanic_model.pkl")
columns = joblib.load("train_columns.pkl")

st.title("Titanic Survival Prediction")

st.write("Enter passenger details:")

pclass = st.selectbox("Passenger Class", [1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.slider("Age", 0,80,25)
sibsp = st.number_input("Siblings/Spouses",0,10)
parch = st.number_input("Parents/Children",0,10)
fare = st.number_input("Fare",0.0,500.0)
embarked = st.selectbox("Embarked",["C","Q","S"])

# encoding
sex = 1 if sex=="female" else 0

embarked_Q = 1 if embarked=="Q" else 0
embarked_S = 1 if embarked=="S" else 0

# input dataframe
input_data = pd.DataFrame(
[[pclass,sex,age,sibsp,parch,fare,embarked_Q,embarked_S]],
columns=columns
)

if st.button("Predict Survival"):

    prediction = model.predict(input_data)[0]

    if prediction==1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")