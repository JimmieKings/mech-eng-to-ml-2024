import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Loading the trained Random Forest model
model = joblib.load('churn_model.pkl')

# Load any necessary encoders 

with open('encoder.pkl', 'rb') as encoder_file:
     encoder = joblib.load(encoder_file)

# Streamlit app title
st.title("Expresso Customer Churn Prediction")

# Instructions for the user
st.write("Enter the required features to predict whether a customer will churn or not.")

# Inputting fields for the relevant columns/features from the dataset

# Region (assume you had encoded region or have predefined categories)
region = st.selectbox('Region', ['Region1', 'Region2', 'Region3', 'Unknown'])

# Tenure (number of months)
tenure = st.number_input('Tenure (Months)', min_value=0, max_value=100)

# Montant (monthly payment)
montant = st.number_input('Montant', min_value=0.0)

# Frequency of recharge
frequency_rech = st.number_input('Frequency of Recharge')

# Revenue
revenue = st.number_input('Revenue', min_value=0.0)

# ARPU segment
arpu_segment = st.number_input('ARPU Segment', min_value=0.0)

# Frequency of use
frequency = st.number_input('Frequency', min_value=0)

# Data volume
data_volume = st.number_input('Data Volume (MB)', min_value=0.0)

# On-net calls
on_net = st.number_input('On Net Calls (Minutes)', min_value=0.0)

# Orange network usage
orange = st.number_input('Orange Network Usage (Minutes)', min_value=0.0)

# Tigo network usage
tigo = st.number_input('Tigo Network Usage (Minutes)', min_value=0.0)

# ZONE1 (Zone 1 network usage)
zone1 = st.number_input('Zone 1 Usage (Minutes)', min_value=0.0)

# ZONE2 (Zone 2 network usage)
zone2 = st.number_input('Zone 2 Usage (Minutes)', min_value=0.0)

# MRG (Monthly recharge group)
mrg = st.number_input('MRG (Monthly Recharge Group)', min_value=0)

# Regularity of usage
regularity = st.number_input('Regularity', min_value=0)

# Top Pack
top_pack = st.selectbox('Top Pack', ['Pack1', 'Pack2', 'Pack3', 'No Pack'])

# Frequency of top pack usage
freq_top_pack = st.number_input('Frequency of Top Pack', min_value=0)

# Prepare the input data for the model in the form of a DataFrame
input_data = pd.DataFrame({
    'REGION': [region],
    'TENURE': [tenure],
    'MONTANT': [montant],
    'FREQUENCE_RECH': [frequency_rech],
    'REVENUE': [revenue],
    'ARPU_SEGMENT': [arpu_segment],
    'FREQUENCE': [frequency],
    'DATA_VOLUME': [data_volume],
    'ON_NET': [on_net],
    'ORANGE': [orange],
    'TIGO': [tigo],
    'ZONE1': [zone1],
    'ZONE2': [zone2],
    'MRG': [mrg],
    'REGULARITY': [regularity],
    'TOP_PACK': [top_pack],
    'FREQ_TOP_PACK': [freq_top_pack]
})

# encode any categorical variables here before making predictions
# input_data['REGION'] = encoder.transform(input_data['REGION'])

# Prediction button
if st.button("Predict Churn"):
    # Predict churn probability
    prediction = model.predict(input_data)
    churn_prob = model.predict_proba(input_data)[0][1]  # Probability of class 1 (churn)

    # Display the prediction result
    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Churn Probability: {churn_prob:.2f}")

