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
TPOT_MODEL_PATH = "tpot_best_model.pkl"
TRAINING_PROFILE_PATH = "training_data_profiling_report.html"
USER_PROFILE_PATH = "user_data_profiling_report.html"
COLUMN_MODELS_DIR = "column_pipelines"  # Directory to save column-specific pipelines

# Ensure the directory exists
if not os.path.exists(COLUMN_MODELS_DIR):
    os.makedirs(COLUMN_MODELS_DIR)

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset - Copy.csv")

# Data Preprocessing
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

# Save the encoders
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

# Fill missing values
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# Generate and open YData profiling for training data
training_profile = ProfileReport(data, title="Training Data Profiling Report", explorative=True)
training_profile.to_file(TRAINING_PROFILE_PATH)
webbrowser.open(TRAINING_PROFILE_PATH)

# Save cleaned data for TPOT usage
data.to_csv("cleaned_traffic_data.csv", index=False)

# TPOT Optimization: Run the TPOT pipeline
data = pd.read_csv("cleaned_traffic_data.csv")
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Run TPOT for multiple models
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=123)
tpot.fit(X_train, y_train)

# Save all models (as a combined file and separate pipelines)
with open(ALL_MODELS_PATH, 'wb') as f:
    pickle.dump(tpot.fitted_pipeline_, f)

# Save best model
with open(TPOT_MODEL_PATH, 'wb') as f:
    pickle.dump(tpot.fitted_pipeline_, f)

# Save separate pipelines for each column
for column in X_train.columns:
    column_X_train = X_train[[column]]
    column_X_test = X_test[[column]]

    column_tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=123)
    column_tpot.fit(column_X_train, y_train)

    column_model_path = os.path.join(COLUMN_MODELS_DIR, f"{column}_pipeline.pkl")
    with open(column_model_path, 'wb') as f:
        pickle.dump(column_tpot.fitted_pipeline_, f)

# Streamlit App
st.title("Bangalore Traffic Prediction App")

# Load best model
try:
    with open(MODEL_PATH, 'rb') as f:
        best_model = pickle.load(f)
    st.subheader("Best Model Information")
    st.write(f"**Best Model:** {best_model.__class__.__name__}")
except FileNotFoundError:
    st.error("The best model file was not found. Please run the training process to generate it.")

# File uploader for user data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Process uploaded file
    user_data = pd.read_csv(uploaded_file)

    # Handle categorical columns
    for col in categorical_columns:
        if col in user_data.columns:
            user_data[col] = user_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
            user_data[col] = encoders[col].transform(user_data[col].astype(str))

    # Fill missing values
    user_data = user_data.fillna(user_data.mean())
    for col in categorical_columns:
        user_data[col] = user_data[col].fillna(user_data[col].mode()[0])

    # Generate and open YData profiling for user-provided data
    user_profile = ProfileReport(user_data, title="User Data Profiling Report", explorative=True)
    user_profile.to_file(USER_PROFILE_PATH)
    webbrowser.open(USER_PROFILE_PATH)

    # Ensure all columns are numeric
    user_data = user_data.apply(pd.to_numeric, errors='coerce')
    user_data = user_data.fillna(0)

    # Make predictions using the best model
    X_user = user_data.drop(columns=['Traffic_Volume'], errors='ignore')
    try:
        predictions = tpot.fitted_pipeline_.predict(X_user)
        user_data['Traffic_Volume'] = predictions
        st.write("Predictions:")
        st.write(user_data)

        # Offer download of predictions
        csv = user_data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")
    except FileNotFoundError:
        st.error("The optimized pipeline file was not found. Please run the training process to generate it.")

# Extract model names from the TPOT pipeline
model_names = []
for step_name, step_model in tpot.fitted_pipeline_.steps:
    if hasattr(step_model, 'named_steps'):  # For pipelines within pipelines
        for sub_step_name, sub_step_model in step_model.named_steps.items():
            if hasattr(sub_step_model, 'get_params'):
                model_names.append(sub_step_model.__class__.__name__)
    else:
        if hasattr(step_model, 'get_params'):
            model_names.append(step_model.__class__.__name__)

# Remove duplicates
model_names = list(set(model_names))

# Streamlit dropdown for selecting a custom pipeline
st.subheader("Choose a Custom Pipeline")
selected_pipeline = st.selectbox("Select a pipeline:", model_names)
st.write(f"Selected Pipeline: {selected_pipeline}")
