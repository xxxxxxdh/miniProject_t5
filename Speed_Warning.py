import tensorflow as tf
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import os
import io

# Function to load the pre-trained model
@st.cache_resource()
def load_trained_model():
    try:
        # Define the path to Model 3 directory
        model3_dir = os.path.join(os.path.dirname(__file__), 'Model 3 - Speed or warning sign recognition')
        
        # Load the model
        model_path = os.path.join(model3_dir, 'signs_model.keras')
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error('Error loading model: {}'.format(str(e)))
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
            img_byte_arr, target_size=(150, 150)
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

# Streamlit application for Speed or Warning Sign Recognition
def speed_warning_app():
    # Mapping the model output to human-readable labels
    sign_names = {
        0: 'Speed sign!',
        1: 'Warning sign!!',
    }

    # Title
    st.title('Traffic Sign Recognition')

    # About section
    st.sidebar.title("About")
    st.sidebar.info("""
        This application is part of a Traffic Safety and Monitoring project. It uses a convolutional neural network (CNN) model to recognize traffic signs, 
        specifically speed signs and warning signs. Upload an image of a traffic sign, and the model will classify it as either a speed sign or a warning sign.
    """)

    # Load the model
    with st.spinner('Loading model...'):
        model = load_trained_model()

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

# Main function to run the Streamlit app
if __name__ == "__main__":
    speed_warning_app()
