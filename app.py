import streamlit as st
import numpy as np
import pandas as pd
import joblib
import datetime

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("model_loan_defaulter.pkl")

feature_names = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
    'AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
    'DAYS_REGISTRATION', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS',
    'REGION_RATING_CLIENT_W_CITY', 'OBS_30_CNT_SOCIAL_CIRCLE',
    'DEF_30_CNT_SOCIAL_CIRCLE'
]

THRESHOLD = 0.39

# ---------------------------
# UI Design
# ---------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("🏦 Home Loan Defaulter Prediction App")
st.write("Fill in the information to predict the loan default risk ⚠️")

with st.form("prediction_form"):

    name_contract = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    gender = st.selectbox("Gender", ["M", "F"])

    income = st.number_input("Annual Income", min_value=1000, max_value=5000000, value=120000)
    credit = st.number_input("Credit Amount", min_value=5000, max_value=3000000, value=300000)
    goods = st.number_input("Goods Price", min_value=5000, max_value=3000000, value=270000)

    suite = st.selectbox("Type Suite", ['Unaccompanied','Family','Spouse, partner','Children','Other_A','Other_B','Group of people'])
    income_type = st.selectbox("Income Type", ['Working','State servant','Commercial associate','Pensioner','Unemployed','Student','Businessman','Maternity leave'])
    edu_type = st.selectbox("Education Level", ['Secondary / secondary special', 'Higher education','Incomplete higher','Lower secondary','Academic degree'])
    family = st.selectbox("Family Status", ['Single / not married','Married','Civil marriage','Widow','Separated'])
    housing = st.selectbox("Housing Type", ['House / apartment','Rented apartment','With parents','Municipal apartment','Office apartment','Co-op apartment'])

    region_pop = st.slider("Region Population Relative", min_value=0.0, max_value=0.06, value=0.02)

    # Negative days corrected!
    age = st.number_input("Days of Birth (negative)", min_value=-30000, max_value=-5000, value=-14000)
    employed = st.number_input("Days Employed (negative)", min_value=-20000, max_value=0, value=-500)
    reg_days = st.number_input("Registration Days (negative)", min_value=-20000, max_value=0, value=-4500)

    occupation = st.selectbox("Occupation Type", 
        ['Laborers','Core staff','Accountants','Managers','Drivers','Sales staff',
         'Cleaning staff','Cooking staff','Private service staff','Medicine staff',
         'Security staff','High skill tech staff','Waiters/barmen staff',
         'Low-skill Laborers','Realty agents','Secretaries','IT staff','HR staff','Other'])

    members = st.number_input("Family Members", min_value=1, max_value=15, value=2)
    rating = st.slider("Region Rating Client w City", min_value=1, max_value=3, value=2)

    obs30 = st.number_input("OBS 30 Social Circle", min_value=0, max_value=30, value=1)
    def30 = st.number_input("DEF 30 Social Circle", min_value=0, max_value=30, value=0)

    submitted = st.form_submit_button("Predict")

# ---------------------------
# Prediction
# ---------------------------
if submitted:
    new_data = pd.DataFrame({
        'NAME_CONTRACT_TYPE': [name_contract],
        'CODE_GENDER': [gender],
        'AMT_INCOME_TOTAL': [income],
        'AMT_CREDIT': [credit],
        'AMT_GOODS_PRICE': [goods],
        'NAME_TYPE_SUITE': [suite],
        'NAME_INCOME_TYPE': [income_type],
        'NAME_EDUCATION_TYPE': [edu_type],
        'NAME_FAMILY_STATUS': [family],
        'NAME_HOUSING_TYPE': [housing],
        'REGION_POPULATION_RELATIVE': [region_pop],
        'DAYS_BIRTH': [age],
        'DAYS_EMPLOYED': [employed],
        'DAYS_REGISTRATION': [reg_days],
        'OCCUPATION_TYPE': [occupation],
        'CNT_FAM_MEMBERS': [members],
        'REGION_RATING_CLIENT_W_CITY': [rating],
        'OBS_30_CNT_SOCIAL_CIRCLE': [obs30],
        'DEF_30_CNT_SOCIAL_CIRCLE': [def30],
    })[feature_names]

    proba = float(model.predict_proba(new_data)[:, 1])
    prediction = int(proba >= THRESHOLD)

    st.subheader("📌 Prediction Result")
    st.metric("Probability of Default", f"{proba:.2f}")
    st.success("🟢 Low Default Risk") if prediction == 0 else st.error("🔴 High Default Risk")

    with open("prediction_log.txt", "a") as f:
        f.write(f"\n{datetime.datetime.now()} | Proba={proba:.3f} | Pred={prediction}")

st.write("---")
st.write("🎯 Prediction threshold used:", THRESHOLD)
