import pandas as pd
import numpy as np
from pycaret.regression import *
from ydata_profiling import ProfileReport
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st
import os
import webbrowser

# Paths
MODEL_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\traffic_regression_model.pkl"
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"
ALL_MODELS_PATH = "all_tpot_models.pkl"
TRAIN_PROFILE_REPORT_PATH = "traffic_training_data_profiling_report.html"
USER_PROFILE_REPORT_PATH = "traffic_user_data_profiling_report.html"
TPOT_MODEL_PATH = "tpot_best_model.pkl"

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset.csv")

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Ensure proper datetime format

# Drop rows with invalid dates
data = data.dropna(subset=['Date'])

# Extract Date Components
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday

# Apply sine and cosine transformation for cyclical features (Month, Day, Weekday)
data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)

data['Day_sin'] = np.sin(2 * np.pi * data['Day'] / 31)
data['Day_cos'] = np.cos(2 * np.pi * data['Day'] / 31)

data['Weekday_sin'] = np.sin(2 * np.pi * data['Weekday'] / 7)
data['Weekday_cos'] = np.cos(2 * np.pi * data['Weekday'] / 7)

# Drop the original Date column
data.drop(columns=['Date'], inplace=True)

# Handle categorical columns by encoding them
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))  # Ensure consistent string type for encoding
    encoders[col] = le

# Save the encoders
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

# Fill missing values
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# YData Profiling for training data
train_profile = ProfileReport(data, title="Training Data Profiling Report", explorative=True)
train_profile.to_file(TRAIN_PROFILE_REPORT_PATH)

# Open the training data profiling report automatically in the browser
webbrowser.open(TRAIN_PROFILE_REPORT_PATH)

# Save cleaned data for TPOT usage
data.to_csv("cleaned_traffic_data.csv", index=False)

# TPOT Optimization: Run the TPOT pipeline
data = pd.read_csv("cleaned_traffic_data.csv")
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Run TPOT for multiple models, save all pipelines
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=123)
tpot.fit(X_train, y_train)

# Save all models
with open(ALL_MODELS_PATH, 'wb') as f:
    pickle.dump(tpot.fitted_pipeline_, f)

# Save best model
with open(TPOT_MODEL_PATH, 'wb') as f:
    pickle.dump(tpot.fitted_pipeline_, f)

# Streamlit App
st.title("Bangalore Traffic Prediction App")

# Best model summary
st.subheader("Best Model Information")
try:
    # Load the best model
    with open(MODEL_PATH, 'rb') as f:
        best_model = pickle.load(f)

    # Display model information
    st.write(f"**Best Model:** {best_model.__class__.__name__}")
except FileNotFoundError:
    st.error("The best model file was not found. Please run the training process to generate it.")

# File uploader for user data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Process uploaded file
    input_data = pd.read_csv(uploaded_file)
    input_data['Date'] = pd.to_datetime(input_data['Date'], errors='coerce')
    input_data = input_data.dropna(subset=['Date'])

    # Extract and transform date components
    input_data['Year'] = input_data['Date'].dt.year
    input_data['Month'] = input_data['Date'].dt.month
    input_data['Day'] = input_data['Date'].dt.day
    input_data['Weekday'] = input_data['Date'].dt.weekday

    input_data['Month_sin'] = np.sin(2 * np.pi * input_data['Month'] / 12)
    input_data['Month_cos'] = np.cos(2 * np.pi * input_data['Month'] / 12)

    input_data['Day_sin'] = np.sin(2 * np.pi * input_data['Day'] / 31)
    input_data['Day_cos'] = np.cos(2 * np.pi * input_data['Day'] / 31)

    input_data['Weekday_sin'] = np.sin(2 * np.pi * input_data['Weekday'] / 7)
    input_data['Weekday_cos'] = np.cos(2 * np.pi * input_data['Weekday'] / 7)

    # Handle categorical columns
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = input_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
            input_data[col] = encoders[col].transform(input_data[col].astype(str))

    # YData Profiling for user data
    user_profile = ProfileReport(input_data, title="User Data Profiling Report", explorative=True)
    user_profile.to_file(USER_PROFILE_REPORT_PATH)

    # Open the user data profiling report automatically in the browser
    webbrowser.open(USER_PROFILE_REPORT_PATH)

    X_user = input_data.drop(columns=['Traffic_Volume'], errors='ignore')

    # Make predictions using the selected pipeline
    try:
        predictions = tpot.fitted_pipeline_.predict(X_user)
        input_data['Traffic_Volume'] = predictions
        st.write("Predictions:")
        st.write(input_data)

        # Offer download of predictions
        csv = input_data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")
    except FileNotFoundError:
        st.error("The optimized pipeline file was not found. Please run the training process to generate it.")
