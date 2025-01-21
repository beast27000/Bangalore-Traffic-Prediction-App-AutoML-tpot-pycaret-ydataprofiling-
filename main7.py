import pandas as pd
import pycaret
from ydata_profiling import ProfileReport
from pycaret.regression import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st
import os
import webbrowser
import hashlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from tpot import TPOTRegressor

# Paths
MODEL_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\traffic_regression_model.pkl"
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"
PROFILE_REPORT_PATH = "traffic_data_profiling_report.html"
OPTIMIZED_PIPELINE_DIR = "optimized_pipelines"

# Ensure directory exists
if not os.path.exists(OPTIMIZED_PIPELINE_DIR):
    os.makedirs(OPTIMIZED_PIPELINE_DIR)

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset.csv")

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday
data = data.drop(columns=['Date'])

# Handle categorical columns by encoding them
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions',
                       'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Save the encoders to a file
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# YData Profiling
profile = ProfileReport(data, title="Bangalore Traffic Data Profiling Report", explorative=True)
profile.to_file(PROFILE_REPORT_PATH)
webbrowser.open(PROFILE_REPORT_PATH)

# Save cleaned data for reusability
data.to_csv("cleaned_traffic_data.csv", index=False)

# PyCaret Setup
exp1 = setup(data=data, target='Traffic_Volume', session_id=123)
best_model = compare_models()

# Save the best model
save_model(best_model, MODEL_PATH)

# Available simple regression models
simple_models = {
    'Random Forest': RandomForestRegressor(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Support Vector Regression (SVR)': SVR()
}

# TPOT Model
tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, random_state=42)
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']
tpot.fit(X, y)

# Save the best pipeline from TPOT
tpot_best_pipeline = tpot.fitted_pipeline_
tpot_pipeline_file = os.path.join(OPTIMIZED_PIPELINE_DIR, "tpot_best_pipeline.pkl")
with open(tpot_pipeline_file, 'wb') as f:
    pickle.dump(tpot_best_pipeline, f)

# Streamlit App
st.title("Bangalore Traffic Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.write(input_data)

    try:    
        # Preprocess the input data
        input_data['Date'] = pd.to_datetime(input_data['Date'], errors='coerce')
        input_data['Year'] = input_data['Date'].dt.year
        input_data['Month'] = input_data['Date'].dt.month
        input_data['Day'] = input_data['Date'].dt.day
        input_data['Weekday'] = input_data['Date'].dt.weekday
        input_data = input_data.drop(columns=['Date'])

        for col in categorical_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])
                input_data[col] = encoders[col].transform(input_data[col])

        non_target_columns = [col for col in input_data.columns if col != 'Traffic_Volume']
        input_data[non_target_columns] = input_data[non_target_columns].fillna(input_data[non_target_columns].mean())
        for col in categorical_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].fillna(input_data[col].mode()[0])

        # User selects a simple model or TPOT model
        model_choice = st.selectbox("Select a regression model", list(simple_models.keys()) + ["Best Pipeline (TPOT)"])

        if model_choice == "Best Pipeline (TPOT)":
            # Load the best pipeline from TPOT
            with open(tpot_pipeline_file, 'rb') as f:
                selected_model = pickle.load(f)
        else:
            # Load the selected simple model
            selected_model = simple_models[model_choice]

        # Train the model on the full data
        X = data.drop(columns=['Traffic_Volume'])
        y = data['Traffic_Volume']
        selected_model.fit(X, y)

        # Make predictions
        predictions = selected_model.predict(input_data.drop(columns=['Traffic_Volume']))
        input_data['Traffic_Volume'] = predictions

        st.write("Predictions:")
        st.write(input_data)

        # Option to download the predictions
        csv = input_data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
