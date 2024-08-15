import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

# Load the pre-trained model and the scaler
@st.cache_resource()
def load_road_monitoring_model():
    # Define the path to Model 4 directory
    model4_dir = os.path.join(os.path.dirname(__file__), 'Model 4 - Road Traffic Monitoring Level Prediction')
    
    # Load the model and scaler
    model = load_model(os.path.join(model4_dir, 'model4.h5'))
    scaler = joblib.load(os.path.join(model4_dir, 'scaler.pkl'))
    return model, scaler

def road_monitoring_app():
    st.title("Road Traffic Monitoring Level Prediction")

    # Sidebar inputs
    st.sidebar.title("Input Parameters")

    # Input fields for the features
    date_time = st.sidebar.date_input("Select Date")
    time = st.sidebar.time_input("Select Time")
    junction = st.sidebar.selectbox("Junction", [1, 2, 3, 4])
    vehicles = st.sidebar.slider("Number of Vehicles", 1, 180, 10)

    # Convert selected date and time to the necessary features
    year = date_time.year
    month = date_time.month
    date_no = date_time.day
    hour = time.hour
    day = date_time.strftime("%A")  # Extract the day name

    # One-hot encoding for the 'Day' feature
    day_columns = {
        "Monday": "Day_Monday", 
        "Tuesday": "Day_Tuesday", 
        "Wednesday": "Day_Wednesday", 
        "Thursday": "Day_Thursday", 
        "Friday": "Day_Friday", 
        "Saturday": "Day_Saturday", 
        "Sunday": "Day_Sunday"
    }

    # Predict button
    if st.sidebar.button("Predict Monitoring Level"):
        # Prepare input for the model
        input_data = pd.DataFrame({
            'Year': [year],
            'Month': [month],
            'Date_no': [date_no],
            'Hour': [hour],
            'Junction': [junction],
            'Vehicles': [vehicles],
        })
        
        # Add day columns with default 0s
        for day_column in day_columns.values():
            input_data[day_column] = 0
        
        # Set the correct day column to 1
        input_data[day_columns[day]] = 1

        # Ensure all necessary columns are present (fill missing with zeros if needed)
        all_columns = [
            'Junction', 'Vehicles', 'Year', 'Month', 'Date_no', 'Hour',
            'Day_Friday', 'Day_Monday', 'Day_Saturday', 'Day_Sunday',
            'Day_Thursday', 'Day_Tuesday', 'Day_Wednesday'
        ]
        for col in all_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder the columns to match the order during training
        input_data = input_data[all_columns]

        # Load the model and scaler
        model, scaler = load_road_monitoring_model()

        # Scale the input data using the loaded scaler
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Determine monitoring level based on the number of vehicles
        if vehicles > 100:
            monitoring_level = "High Monitoring"
        elif vehicles > 50:
            monitoring_level = "Medium Monitoring"
        else:
            monitoring_level = "Low Monitoring"

        # Display the prediction
        st.write(f"Recommended Monitoring Level: {monitoring_level}")
        st.write("The recommended monitoring level is based on the number of vehicles on the road. Adjustments in monitoring are suggested to ensure road safety and efficiency.")

    st.sidebar.title("About")
    st.sidebar.info("This app predicts the required level of road traffic monitoring based on the number of vehicles and other relevant features, helping to ensure road safety and efficiency.")
