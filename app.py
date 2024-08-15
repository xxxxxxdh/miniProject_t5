# File: app.py

import streamlit as st
from Speed_Limit import speed_limit_app
from Traffic_Prediction import traffic_prediction_app as traffic_prediction_app
from Speed_Warning import speed_warning_app
from Road_Monitoring import road_monitoring_app

# Set the page configuration (only once, at the top)
st.set_page_config(
    page_title="Traffic Safety and Monitoring Models",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main app to connect all models
def main():
    st.title("Traffic Safety and Monitoring Models")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a model", 
                                    ["Speed Limit Sign Recognition", 
                                     "Traffic Prediction Based on Vehicle Count", 
                                     "Speed or Warning Sign Recognition", 
                                     "Road Traffic Monitoring Level Prediction"])
    
    # Display the selected model's app
    if app_mode == "Speed Limit Sign Recognition":
        speed_limit_app()
    elif app_mode == "Traffic Prediction Based on Vehicle Count":
        traffic_prediction_app()
    elif app_mode == "Speed or Warning Sign Recognition":
        speed_warning_app()
    elif app_mode == "Road Traffic Monitoring Level Prediction":
        road_monitoring_app()

if __name__ == "__main__":
    main()
