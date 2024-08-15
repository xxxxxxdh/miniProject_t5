import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

# Function to load the pre-trained model
@st.cache_resource()
def load_traffic_prediction_model():
    model2_dir = os.path.join(os.path.dirname(__file__), 'Model 2 - traffic predict')
    model_path = os.path.join(model2_dir, 'concat_traffic_model.h5')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    
    model = load_model(model_path)
    return model

def traffic_prediction_app():
    st.title("Traffic Prediction Based on Vehicle Count")

    # Sidebar inputs
    st.sidebar.title("Input Parameters")

    # Input fields for the features
    day_of_week = st.sidebar.selectbox('Day of the Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    car_count = st.sidebar.slider('Car Count', 0, 1000, 50)
    bike_count = st.sidebar.slider('Bike Count', 0, 1000, 10)
    bus_count = st.sidebar.slider('Bus Count', 0, 100, 5)
    truck_count = st.sidebar.slider('Truck Count', 0, 100, 5)
    time_of_day = st.sidebar.slider('Time of Day (Hour)', 0, 24, 12)

    # Convert day of the week to an integer
    day_of_week_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week)

    # Calculate total vehicle count
    total_vehicle_count = car_count + bike_count + bus_count + truck_count
    st.sidebar.write("Total Vehicle Count:", total_vehicle_count)

    # Predict button
    if st.sidebar.button("Predict Traffic Situation"):
        # Create a DataFrame with the input data
        input_data_df = pd.DataFrame({
            'Day of the week': [day_of_week_num],
            'Time of Day (Hour)': [time_of_day],
            'Car Count': [car_count],
            'Bike Count': [bike_count],
            'Bus Count': [bus_count],
            'Truck Count': [truck_count],
            'Total Vehicle Count': [total_vehicle_count]
        })

        # Convert the DataFrame to a NumPy array
        input_data = input_data_df.to_numpy()

        # Load the model
        model = load_traffic_prediction_model()

        # Make a prediction
        prediction = model.predict(input_data)

        # Check if the prediction contains NaN values
        if np.isnan(prediction).any():
            st.write("Prediction contains NaN values. Please check the model and input data.")
            return

        # Get the predicted class
        predicted_class = np.argmax(prediction, axis=1)

        # Map prediction to traffic situation
        traffic_situations = {0: "No Traffic", 1: "Moderate Traffic", 2: "Heavy Traffic"}
        predicted_situation = traffic_situations.get(predicted_class[0], "Unknown")

        # Display the prediction
        st.write(f"Predicted Traffic Situation: {predicted_situation}")

    st.sidebar.title("About")
    st.sidebar.info("""
        This application predicts traffic conditions based on various user inputs such as the day of the week, 
        the number of cars, bikes, buses, and trucks, and the time of day. The prediction is based on a machine 
        learning model trained to assess the traffic situation and classify it into categories such as Heavy Traffic, 
        Moderate Traffic, and No Traffic.
    """)

if __name__ == "__main__":
    traffic_prediction_app()
