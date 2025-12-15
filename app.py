import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic
from datetime import datetime
import csv

# Load model and encoder
model = joblib.load("fraud_detection_model.jb")
encoder = joblib.load("label_encoder.jb")

# Haversine distance function
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Streamlit UI
st.title("Fraud Detection System")
st.write("Enter the transaction details below:")

merchant = st.text_input("Merchant Name")
category = st.text_input("Category")
amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
lat = st.number_input("Latitude", format="%.6f")
long = st.number_input("Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")
hour = st.slider("Transaction Hour", 0, 23, 12)
day = st.slider("Transaction Day", 1, 31, 15)
month = st.slider("Transaction Month", 1, 12, 6)
gender = st.selectbox("Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

# Calculate distance
distance = haversine(lat, long, merch_lat, merch_long)

if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        # Build input dataframe
        input_data = pd.DataFrame([[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])
        
        # Encode categoricals
        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        # Hash cc_num
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"

        st.subheader(f"Prediction: {result}")

        # ✅ Save to log CSV
        log_row = [datetime.now(), merchant, category, amt, distance, hour, day, month, gender, cc_num, result]

        # Append to CSV, create header if needed
        try:
            with open("fraud_predictions_log.csv", "x") as f:
                writer = csv.writer(f)
                writer.writerow(['datetime', 'merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num', 'result'])
                writer.writerow(log_row)
        except FileExistsError:
            with open("fraud_predictions_log.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow(log_row)

    else:
        st.error("Please fill all required fields!")

# ✅ Always show the log table below
st.write("---")
st.write("### 📂 Past Predictions Log")

try:
    log_df = pd.read_csv("fraud_predictions_log.csv")
    log_df = log_df.sort_values(by="datetime", ascending=False)
    st.dataframe(log_df)
except FileNotFoundError:
    st.info("No predictions saved yet. Make a prediction first!")
