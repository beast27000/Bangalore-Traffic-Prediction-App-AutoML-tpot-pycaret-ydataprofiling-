# NOT WORKING PLS IGNORE 

import pandas as pd
import os
import pickle
import webbrowser
from ydata_profiling import ProfileReport
from pycaret.regression import setup, compare_models, save_model
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Paths
MODEL_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\traffic_regression_model.pkl"
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"
PROFILE_REPORT_PATH = "traffic_data_profiling_report.html"
OPTIMIZED_PIPELINE_DIR = "optimized_pipelines"

# Ensure directory exists for pipelines
if not os.path.exists(OPTIMIZED_PIPELINE_DIR):
    os.makedirs(OPTIMIZED_PIPELINE_DIR)

# Load dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset.csv")

# Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Weekday'] = data['Date'].dt.weekday
data = data.drop(columns=['Date'])

# Handle categorical columns
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions',
                       'Traffic Signal Compliance', 'Roadwork and Construction Activity']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Save encoders
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(encoders, f)

# Fill missing values
data = data.fillna(data.mean())
for col in categorical_columns:
    data[col] = data[col].fillna(data[col].mode()[0])

# YData Profiling
profile = ProfileReport(data, title="Bangalore Traffic Data Profiling Report", explorative=True)
profile.to_file(PROFILE_REPORT_PATH)
webbrowser.open(PROFILE_REPORT_PATH)

# Save cleaned data
data.to_csv("cleaned_traffic_data.csv", index=False)

# PyCaret Model Setup and Save Best Model
exp1 = setup(data=data, target='Traffic_Volume', session_id=123)
best_model = compare_models()
save_model(best_model, MODEL_PATH)

# TPOT Optimization
X = data.drop(columns=['Traffic_Volume'])
y = data['Traffic_Volume']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=123)
tpot.fit(X_train, y_train)

# Save all pipelines
for i, (pipeline_name, pipeline_details) in enumerate(tpot.evaluated_individuals_.items(), 1):
    pipeline_file = os.path.join(OPTIMIZED_PIPELINE_DIR, f"pipeline_{pipeline_name}.pkl")
    with open(pipeline_file, 'wb') as f:
        pickle.dump(pipeline_details, f)

# Save best pipeline
best_pipeline_path = os.path.join(OPTIMIZED_PIPELINE_DIR, "best_pipeline.pkl")
with open(best_pipeline_path, 'wb') as f:
    pickle.dump(tpot.fitted_pipeline_, f)

# Streamlit App
st.title("Bangalore Traffic Prediction App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("Input Data:")
    st.write(input_data)

    try:
        # Preprocess input data
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

        # Load best pipeline and predict
        with open(best_pipeline_path, 'rb') as f:
            best_pipeline = pickle.load(f)
        predictions = best_pipeline.predict(input_data.drop(columns=['Traffic_Volume']))
        input_data['Traffic_Volume'] = predictions

        st.write("Predictions:")
        st.write(input_data)

        # Download predictions
        csv = input_data.to_csv(index=False)
        st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
