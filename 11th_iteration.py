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
PIPELINE_DIR = "pipelines/"
PROFILE_REPORT_PATH = "traffic_data_profiling_report.html"
TPOT_MODEL_PATH = "tpot_best_model.pkl"
YDATA_PROFILE_PATH = "user_data_profiling_report.html"

# Ensure the pipeline directory exists
os.makedirs(PIPELINE_DIR, exist_ok=True)

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset - Copy.csv")

# Data Preprocessing
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Fill missing values
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# YData Profiling for training data (will run each time)
profile = ProfileReport(data, title="Bangalore Traffic Data Profiling Report", explorative=True)
profile.to_file(PROFILE_REPORT_PATH)

# Save cleaned data for TPOT usage
data.to_csv("cleaned_traffic_data.csv", index=False)

# TPOT Optimization
data = pd.read_csv("cleaned_traffic_data.csv")
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Train TPOT and save all pipelines
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=123)
tpot.fit(X_train, y_train)

# Save all pipelines
for idx, pipeline in enumerate(tpot.evaluated_individuals_.items()):
    pipeline_name = f"{PIPELINE_DIR}pipeline_{idx}.pkl"
    with open(pipeline_name, 'wb') as f:
        pickle.dump(pipeline, f)

# Save best model
with open(TPOT_MODEL_PATH, 'wb') as f:
    pickle.dump(tpot.fitted_pipeline_, f)

# Streamlit App
st.title("Bangalore Traffic Prediction App")

# Load all pipelines
pipeline_files = [f for f in os.listdir(PIPELINE_DIR) if f.endswith(".pkl")]
pipelines = {}
for file in pipeline_files:
    with open(f"{PIPELINE_DIR}{file}", 'rb') as f:
        pipelines[file] = pickle.load(f)

# Dropdown for custom pipeline selection
st.subheader("Select a Custom Pipeline")
selected_pipeline = st.selectbox("Choose a pipeline:", options=list(pipelines.keys()))

# File uploader for user data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Process uploaded file
    input_data = pd.read_csv(uploaded_file)

    # Handle categorical columns
    for col in categorical_columns:
        if col in input_data.columns:
            input_data[col] = input_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
            input_data[col] = encoders[col].transform(input_data[col].astype(str))

    # Fill missing values
    input_data = input_data.fillna(input_data.mean())
    for col in categorical_columns:
        input_data[col] = input_data[col].fillna(input_data[col].mode()[0])

    # Ensure all columns are numeric
    input_data = input_data.apply(pd.to_numeric, errors='coerce')
    input_data = input_data.fillna(0)

    X_user = input_data.drop(columns=['Traffic_Volume'], errors='ignore')

    # YData Profiling for user-uploaded data (will run each time)
    profile = ProfileReport(input_data, title="User Data Profiling Report", explorative=True)
    profile.to_file(YDATA_PROFILE_PATH)

    # Open the YData profiling report automatically in the browser
    webbrowser.open(YDATA_PROFILE_PATH)

    # Use selected pipeline for predictions
    try:
        # Extract the pipeline from the dictionary
        pipeline = pipelines[selected_pipeline][0]  # The first element is the pipeline

        # Make predictions using the selected pipeline
        predictions = pipeline.predict(X_user)
        input_data['Traffic_Volume'] = predictions
        st.write("Predictions:")
        st.write(input_data)

        # Offer download of predictions
        csv = input_data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")
    except FileNotFoundError:
        st.error("The selected pipeline file was not found. Please train the models again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Automatically open YData profiling report for training data
webbrowser.open(PROFILE_REPORT_PATH)
