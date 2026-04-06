import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.title("Credit Risk Prediction App")

# -------- INPUTS -------- #

age = st.slider("Age", 18, 70)
income = st.number_input("Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.slider("Credit Score", 300, 850)
months_employed = st.slider("Months Employed", 0, 120)
num_credit_lines = st.slider("Number of Credit Lines", 1, 5)
interest_rate = st.slider("Interest Rate", 1.0, 30.0)
loan_term = st.slider("Loan Term", 12, 60)
dti = st.slider("DTI Ratio", 0.1, 1.0)

# Categorical Inputs
education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
loan_purpose = st.selectbox("Loan Purpose", ["Home", "Auto", "Business", "Education", "Other"])

has_mortgage = st.selectbox("Has Mortgage", ["Yes", "No"])
has_dependents = st.selectbox("Has Dependents", ["Yes", "No"])
has_cosigner = st.selectbox("Has Co-Signer", ["Yes", "No"])

# -------- ENCODING -------- #

def preprocess():
    data = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti,
        'HasMortgage': 1 if has_mortgage == "Yes" else 0,
        'HasDependents': 1 if has_dependents == "Yes" else 0,
        'HasCoSigner': 1 if has_cosigner == "Yes" else 0
    }

    df = pd.DataFrame([data])

    # Add all required columns with 0
    for col in [
        'Education_High School','Education_Master\'s','Education_PhD',
        'EmploymentType_Part-time','EmploymentType_Self-employed','EmploymentType_Unemployed',
        'MaritalStatus_Married','MaritalStatus_Single',
        'LoanPurpose_Business','LoanPurpose_Education','LoanPurpose_Home','LoanPurpose_Other'
    ]:
        df[col] = 0

    # Set correct one-hot values
    if education == "High School":
        df['Education_High School'] = 1
    elif education == "Master's":
        df['Education_Master\'s'] = 1
    elif education == "PhD":
        df['Education_PhD'] = 1

    if employment == "Part-time":
        df['EmploymentType_Part-time'] = 1
    elif employment == "Self-employed":
        df['EmploymentType_Self-employed'] = 1
    elif employment == "Unemployed":
        df['EmploymentType_Unemployed'] = 1

    if marital == "Married":
        df['MaritalStatus_Married'] = 1
    elif marital == "Single":
        df['MaritalStatus_Single'] = 1

    if loan_purpose == "Business":
        df['LoanPurpose_Business'] = 1
    elif loan_purpose == "Education":
        df['LoanPurpose_Education'] = 1
    elif loan_purpose == "Home":
        df['LoanPurpose_Home'] = 1
    elif loan_purpose == "Other":
        df['LoanPurpose_Other'] = 1

    return df

# -------- PREDICTION -------- #

if st.button("Predict"):
    input_df = preprocess()
    proba = model.predict_proba(input_df)[0][1]

    st.write(f"Risk Score: {proba*100:.2f}%")

    if proba > 0.4:
        st.error("⚠️ High Risk: Likely to Default")
    else:
        st.success("✅ Low Risk: Safe Customer")