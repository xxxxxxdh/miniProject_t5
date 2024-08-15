import os
import json
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import io

# Function to load the pre-trained model
@st.cache_resource()
def load_model_from_files():
    try:
        model_dir = 'Model 1 - Speed limit sign recognition'
        config_path = os.path.join(model_dir, 'config.json')
        weights_path = os.path.join(model_dir, 'model.weights.h5')

        # Check if files exist
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file {weights_path} does not exist.")

        # Load the model configuration
        with open(config_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)

        # Load the model weights
        model.load_weights(weights_path)

        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Predict label for the uploaded image
def predict_label(image, model, sign_names):
    try:
        # Convert the PIL image to a BytesIO object
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Preprocess the image
        img = tf.keras.utils.load_img(
            img_byte_arr, target_size=(224, 224)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Predict the label
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        label = sign_names.get(np.argmax(score), "Unknown Sign")
        return label
    except Exception as e:
        st.error('Error predicting label: {}'.format(str(e)))
        return None

# Streamlit application for speed limit sign recognition
def speed_limit_app():
    # Mapping the model output to human-readable labels
    sign_names = {
        0: 'Speed limit (5km/h)',
        1: 'Speed limit (15km/h)',
        2: 'Speed limit (30km/h)',
        3: 'Speed limit (40km/h)',
        4: 'Speed limit (50km/h)',
        5: 'Speed limit (60km/h)',
        6: 'Speed limit (70km/h)',
        7: 'Speed limit (80km/h)',
    }

    # Title
    st.title('Traffic Sign Recognition')
    
    # About section
    st.sidebar.title("About")
    st.sidebar.info("""
        This application is part of a Traffic Safety and Monitoring project. It uses a convolutional neural network (CNN) model to recognize traffic signs, 
        specifically speed signs. Upload an image of a traffic sign, and the model will classify it.
    """)

    # Load the model
    with st.spinner('Loading model...'):
        model = load_model_from_files()

    # Model loading error handling
    if model is None:
        st.warning("Failed to load the model. Please ensure that the model file is correct and try again.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Upload an image (JPEG, JPG, PNG)", type=["jpg", "jpeg", "png"])

    # Image analysis
    if uploaded_file is not None:
        try:
            with st.spinner('Analyzing image...'):
                image = Image.open(uploaded_file)
                label = predict_label(image, model, sign_names)
                if label is not None:
                    st.image(image, caption='Uploaded Image', use_column_width=True)
                    st.markdown(f'<style> .predicted-label {{ font-size: 25px; font-weight: bold; color: Yellow; }} </style>', unsafe_allow_html=True)
                    st.write(f'<span><p>Predicted label:</p><p class="predicted-label">{label}</p></span>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    speed_limit_app()
