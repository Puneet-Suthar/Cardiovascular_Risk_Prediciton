import streamlit as st
import datetime
from datetime import date
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

st.write("# 10 Year Heart Disease Prediction")

# Defining the age calculator 
def calculateAge(birthDate):
    today = date.today()
    age = today.year - birthDate.year - ((today.month, today.day) <(birthDate.month, birthDate.day))
    return age
 

# Data collection form user 
col1, col2, col3 = st.columns(3)

gender = col1.selectbox("Enter your gender",["Male", "Female"])

dob = col2.date_input("When\'s your birthday",datetime.date(2019, 7, 6), min_value = datetime.date(1930, 7, 6))
age = calculateAge(dob)    

education = col3.selectbox("Highest academic qualification",["High school diploma", "Undergraduate degree", "Postgraduate degree", "PhD"])

isSmoker = col1.selectbox("Are you currently a smoker?",["Yes","No"])

yearsSmoking = col2.number_input("Number of daily cigarettes",value = 0)

BPMeds = col3.selectbox("Are you currently on BP medication?",["Yes","No"])

stroke = col1.selectbox("Have you ever experienced a stroke?",["Yes","No"])

hyp = col2.selectbox("Do you have hypertension?",["Yes","No"])

diabetes = col3.selectbox("Do you have diabetes?",["Yes","No"])

chol = col1.number_input("Enter your cholesterol level")

sys_bp = col2.number_input("Enter your systolic blood pressure")

dia_bp = col3.number_input("Enter your diastolic blood pressure")

bmi = col1.number_input("Enter your BMI")

heart_rate = col2.number_input("Enter your resting heart rate")

glucose = col3.number_input("Enter your glucose level")

#st.button('Predict')


# Storing the collected data into a dataframe

df_pred = pd.DataFrame([[age,education,gender,isSmoker,yearsSmoking,BPMeds,stroke,hyp,diabetes,chol,sys_bp,dia_bp,bmi,heart_rate,glucose]],

columns= ['age','education','sex','is_smoking','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose'])

df_pred['sex'] = df_pred['sex'].apply(lambda x: 1 if x == 'Male' else 0)

df_pred['prevalentHyp'] = df_pred['prevalentHyp'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['prevalentStroke'] = df_pred['prevalentStroke'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['diabetes'] = df_pred['diabetes'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['BPMeds'] = df_pred['BPMeds'].apply(lambda x: 1 if x == 'Yes' else 0)

df_pred['is_smoking'] = df_pred['is_smoking'].apply(lambda x: 1 if x == 'Yes' else 0)

def transform(data):
    result = 4
    if(data=='High school diploma'):
        result = 1
    elif(data=='Undergraduate degree'):
        result = 2
    elif(data=='Postgraduate degree'):
        result = 3
    return(result)

df_pred['education'] = df_pred['education'].apply(transform)

# loading trained model

pickle_in = open('CHD_RF_classifier.pkl', 'rb')
model = pickle.load(pickle_in)

pickle_in = open('CHD_standardizer.pkl', 'rb')
scaler = pickle.load(pickle_in)

# Doing prediciton
df_pred = scaler.transform(df_pred)
prediction = model.predict(df_pred)


if st.button('Predict'):

    if(prediction[0]==0):
        st.balloons()
        st.write('<p class="big-font">You likely will NOT DEVELOP heart disease in 10 years.</p>',unsafe_allow_html=True)

    else:
        st.write('<p class="big-font">You are likely to DEVELOP heart disease in 10 years.</p>',unsafe_allow_html=True)




