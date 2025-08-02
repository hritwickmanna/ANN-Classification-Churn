import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

st.title("Bank Customer Churn Prediction")

# Input fields for customer data
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance")
creditscore = st.number_input("Credit Score")
estimatedsalary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10, 5)
number_of_products = st.slider("Number of Products", 1, 4, 2)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Create DataFrame including Geography
input_data = pd.DataFrame({
    'CreditScore': [creditscore],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [number_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimatedsalary]
})

# One-hot encode the Geography feature
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)
# Combine the one-hot encoded Geography with the input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
# Scale the input data
input_scaled = scaler.transform(input_data)

# Make predictions
predictions = model.predict(input_scaled)
if predictions[0][0] > 0.5:
    st.write("Customer is likely to leave the bank.")
else:
    st.write("Customer is likely to stay with the bank.")
