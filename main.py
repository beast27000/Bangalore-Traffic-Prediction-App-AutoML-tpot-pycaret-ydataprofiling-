import pandas as pd  
import pycaret  
from ydata_profiling import ProfileReport 
from pycaret.regression import *  
from tpot import TPOTRegressor  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
import pickle  
import streamlit as st  
import os

# Paths
MODEL_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\traffic_regression_model.pkl" 
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"  
OPTIMIZED_PIPELINE_PATH = "optimized_pipeline.pkl"  

# Load the dataset
data = pd.read_csv("C:/Advanced projects/Bangalore_Traffic/Banglore_traffic_Dataset.csv")  # Load the traffic dataset

#Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convert 'Date' column to datetime
data['Year'] = data['Date'].dt.year  # Extract year from the 'Date' column
data['Month'] = data['Date'].dt.month  # Extract month from the 'Date' column
data['Day'] = data['Date'].dt.day  # Extract day from the 'Date' column
data['Weekday'] = data['Date'].dt.weekday  # Extract weekday from the 'Date' column

data = data.drop(columns=['Date'])  # Drop the original 'Date' column as it's no longer needed

# Handle categorical columns by encoding them
categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Traffic Signal Compliance', 'Roadwork and Construction Activity']  # Define categorical columns

encoders = {}  # Initialize a dictionary to store encoders
for col in categorical_columns:  # Loop through each categorical column
    le = LabelEncoder()  # Create a LabelEncoder instance
    data[col] = le.fit_transform(data[col])  # Encode the column and replace it with encoded values
    encoders[col] = le  # Store the encoder for future use

# Save the encoders to a file
with open(ENCODER_PATH, 'wb') as f:  # Open the encoder file in write-binary mode
    pickle.dump(encoders, f)  # Save all encoders to the file

data = data.fillna(data.mean())  # Fill missing numerical values with the column mean
for col in categorical_columns:  # Handle missing values in categorical columns
    data[col] = data[col].fillna(data[col].mode()[0])  # Fill missing values with the most frequent value

#  YData Profiling
print("Generating data profiling report...")  # Notify user that profiling is starting
profile = ProfileReport(data, title="Bangalore Traffic Data Profiling Report", explorative=True)  # Generate profiling report
profile.to_file("traffic_data_profiling_report.html")  # Save the profiling report as an HTML file
print("Data profiling complete. Report saved as 'traffic_data_profiling_report.html'.")  # Notify user that profiling is complete

data.to_csv("cleaned_traffic_data.csv", index=False)  # Save the cleaned dataset for further use

#  PyCaret 
print("Starting PyCaret setup...")  # Notify user that PyCaret setup is starting
exp1 = setup(data=data, target='Traffic_Volume', session_id=123)  # Initialize PyCaret without the 'silent' argument
print("Comparing models...")  # Notify user that models are being compared
best_model = compare_models()  # Compare models and select the best one

save_model(best_model, MODEL_PATH)  # Save the best model to a file
print(f"Best model saved at {MODEL_PATH}")  # Notify user where the model is saved

#  Check if optimized pipeline exists, if not, run optimization
if not os.path.exists(OPTIMIZED_PIPELINE_PATH):  # Check if the optimized pipeline file exists
    print("Running TPOT optimization...")  # Notify user that TPOT optimization is starting

    data = pd.read_csv("cleaned_traffic_data.csv")  # Reload the preprocessed dataset

    X = data.drop(columns=['Traffic_Volume'])  # Define features by dropping the target column
    y = data['Traffic_Volume']  # Define target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)  # Split the data into train and test sets

    tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=123)  # Initialize TPOT for optimization
    tpot.fit(X_train, y_train)  # Fit the TPOT model on the training data

    tpot.export(OPTIMIZED_PIPELINE_PATH)  # Save the optimized pipeline
    print("Optimization complete. Best pipeline saved as 'optimized_pipeline.py'.")  # Notify user that optimization is complete
else:
    print("Optimized pipeline already exists. Skipping optimization.")  # Skip optimization if the pipeline exists

#  Load the saved model and label encoder
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):  # Check if both model and encoder files exist
    with open(MODEL_PATH, 'rb') as model_file, open(ENCODER_PATH, 'rb') as encoder_file:  # Open the files
        model = load_model(MODEL_PATH)  # Load the trained model
        encoders = pickle.load(encoder_file)  # Load the encoders
        print("Model and encoder loaded successfully.")  # Notify user that files are loaded
else:
    st.error("Model or encoder not found! Please ensure both files exist.")  # Show error in Streamlit if files are missing
    print(f"Model file exists: {os.path.exists(MODEL_PATH)}")  # Debug: check if model file exists
    print(f"Encoder file exists: {os.path.exists(ENCODER_PATH)}")  # Debug: check if encoder file exists

# Paths
ENCODER_PATH = r"C:\\Advanced projects\\Bangalore_Traffic\\label_encoders.pkl"  
OPTIMIZED_PIPELINE_PATH = "optimized_pipeline.pkl"  

# Load encoders and pipeline
if os.path.exists(OPTIMIZED_PIPELINE_PATH) and os.path.exists(ENCODER_PATH):
    with open(OPTIMIZED_PIPELINE_PATH, 'rb') as pipeline_file, open(ENCODER_PATH, 'rb') as encoder_file:
        optimized_pipeline = pickle.load(pipeline_file)  # Load the optimized pipeline
        encoders = pickle.load(encoder_file)  # Load the label encoders
        print("Optimized pipeline and encoders loaded successfully.")
else:
    st.error("Optimized pipeline or encoder file not found! Ensure both files exist.")
    print(f"Pipeline file exists: {os.path.exists(OPTIMIZED_PIPELINE_PATH)}")
    print(f"Encoder file exists: {os.path.exists(ENCODER_PATH)}")

# Streamlit App for Predictions
st.title("Bangalore Traffic Prediction App")  # Set the app title

# File uploader for input data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])  # Allow user to upload a CSV file

if uploaded_file is not None:  # Check if a file has been uploaded
    input_data = pd.read_csv(uploaded_file)  # Read the uploaded CSV file
    st.write("Input Data:")  # Display the input data
    st.write(input_data)  # Show the input data

    try:
        # Preprocess the input data
        input_data['Date'] = pd.to_datetime(input_data['Date'], errors='coerce')  # Convert 'Date' column to datetime
        input_data['Year'] = input_data['Date'].dt.year  # Extract year from the 'Date' column
        input_data['Month'] = input_data['Date'].dt.month  # Extract month from the 'Date' column
        input_data['Day'] = input_data['Date'].dt.day  # Extract day from the 'Date' column
        input_data['Weekday'] = input_data['Date'].dt.weekday  # Extract weekday from the 'Date' column

        input_data = input_data.drop(columns=['Date'])  # Drop the original 'Date' column

        # Encode categorical columns
        categorical_columns = ['Area Name', 'Road/Intersection Name', 'Weather Conditions', 'Traffic Signal Compliance', 'Roadwork and Construction Activity']
        for col in categorical_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].apply(lambda x: x if x in encoders[col].classes_ else encoders[col].classes_[0])  # Handle unseen labels
                input_data[col] = encoders[col].transform(input_data[col])  # Transform the column using the encoder

        # Fill missing values in non-target columns
        non_target_columns = [col for col in input_data.columns if col != 'Traffic_Volume']  # Exclude the target column
        input_data[non_target_columns] = input_data[non_target_columns].fillna(input_data[non_target_columns].mean())  # Fill missing numerical values with the mean
        for col in categorical_columns:
            if col in input_data.columns:
                input_data[col] = input_data[col].fillna(input_data[col].mode()[0])  # Fill missing values with the most frequent value

        # Make predictions using the optimized pipeline
        if 'optimized_pipeline' in locals():
            predictions = optimized_pipeline.predict(input_data.drop(columns=['Traffic_Volume']))  # Predict without the target column
            input_data['Traffic_Volume'] = predictions  # Populate the blank 'Traffic_Volume' column with predictions

            st.write("Predictions:")  # Display the predictions
            st.write(input_data)  # Show the data with predictions

            # Option to download the predictions
            csv = input_data.to_csv(index=False)
            st.download_button("Download Predictions", data=csv, file_name="traffic_predictions.csv", mime="text/csv")
        else:
            st.error("Optimized pipeline not loaded. Please ensure the file exists.")
    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
