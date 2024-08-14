import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import os


def set_bg_from_url(url, opacity=1):
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
            .sidebar .sidebar-content {{
                background-color: transparent;
            }}
            .sidebar .sidebar-content .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            /* Custom CSS for headings */
            h1 {{
                font-family: 'Comic Sans MS', cursive, sans-serif;
                color: #FF7F50; /* Coral color */
                font-size: 3.5em;
                text-align: center;
            }}
            h2 {{
                font-family: 'Comic Sans MS', cursive, sans-serif;
                color: #00CED1; /* Dark turquoise color */
                font-size: 2.5em;
                text-align: center;
            }}
            /* Custom CSS for subheaders */
            .css-1p3imzw.e1jwwf4i4 {{
                font-family: 'Comic Sans MS', cursive, sans-serif;
                color: #FF1493; /* Deep pink color */
                font-size: 1.8em;
                text-align: center;
            }}
            /* Custom CSS for subheader "Skin Cancer Detection" */
            .disease-subheader {{
                font-family: 'Comic Sans MS', cursive, sans-serif;
                color: #FF7F50; /* Coral color */
                font-size: 2.2em;
                text-align: center;
            }}
            /* Custom CSS for predicted disease */
            .predicted-disease {{
                font-family: 'Comic Sans MS', cursive, sans-serif;
                color: #008000; /* Green color */
                font-size: 1.5em;
                text-align: center;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


def detect_disease(image):
    # Placeholder function for disease detection
    return "Skin Cancer"


def main():
    set_bg_from_url("https://cdn.create.vista.com/api/media/small/309772574/stock-photo-doctor-with-medical-healthcare-graphic-in-hospital", opacity=0.875)

    st.title("Multiple Disease Detection")

    # Define the side menu
    with st.sidebar:
        choice = option_menu('Multiple Disease Detection System',
                             ['Skin Cancer',
                              'Diabetic Retinopathy',
                              'Osteoporosis',
                              'Arrhythmia',
                              'Breast Cancer',
                              'Brain Tumor'],
                             icons=['hospital', 'hospital', 'hospital', 'hospital', 'hospital', 'hospital'],
                             default_index=0)


    def predict(img_path, categories, model):

        img_size=224

        label_dict = {i: category for i, category in enumerate(categories)}

        # Read the input image (grayscale)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Convert grayscale image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Resize the image to img_size x img_size
        resized_img = cv2.resize(img_rgb, (img_size, img_size))

        # Normalize pixel values to range [0, 1]
        normalized_img = resized_img / 255.0

        # Reshape input for model prediction
        input_img = normalized_img.reshape(-1, img_size, img_size, 3)  # Reshape input for model prediction

        # Make a prediction
        prediction = model.predict(input_img)

        predicted_class_index = np.argmax(prediction)
        predicted_class = label_dict[predicted_class_index]

        return predicted_class, input_img


    # Handle different disease detection modules
    if choice == "Arrhythmia":
    
        st.markdown('<p class="disease-subheader">Arrhythmia</p>', unsafe_allow_html=True)

        # Upload image
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            # Define the path where you want to save the uploaded image
            img_path = os.path.join("upload/", f"img.png")

            # Save the uploaded image locally
            image.save(img_path)
            
            categories = ['Flutter Waves', 'Murmur', 'Normal Sinus Rhythm', 'Q Wave', 'Sinus Arrest', 'Ventricular Prematured Depolarization']
            model = load_model("../models/arrhythmia.keras")

            disease, img = predict(img_path, categories, model)

            # Display predicted disease and uploaded image
            st.write(f'<p class="predicted-disease">Predicted Disease: {disease}</p>', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Image", use_column_width=True)

    elif choice == "Diabetic Retinopathy":
        st.markdown('<p class="disease-subheader">Diabetic Retinopathy</p>', unsafe_allow_html=True)

        # Upload image
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            # Define the path where you want to save the uploaded image
            img_path = os.path.join("upload/", f"img.png")

            # Save the uploaded image locally
            image.save(img_path)

            categories = ['No_DR', 'DR']
            model = load_model("../models/diabeticRetinopathy.keras")

            disease, img = predict(img_path, categories, model)

            # Display predicted disease and uploaded image
            st.write(f'<p class="predicted-disease">Predicted Disease: {disease}</p>', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Image", use_column_width=True)

    elif choice == "Osteoporosis":
        st.markdown('<p class="disease-subheader">Osteoporosis</p>', unsafe_allow_html=True)

        # Upload image
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            # Define the path where you want to save the uploaded image
            img_path = os.path.join("upload/", f"img.png")

            # Save the uploaded image locally
            image.save(img_path)

            categories = ['normal', 'osteoporosis']
            model = load_model("../models/osteoporosis.keras")
    
            disease, img = predict(img_path, categories, model)

            # Display predicted disease and uploaded image
            st.write(f'<p class="predicted-disease">Predicted Disease: {disease}</p>', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Image", use_column_width=True)

    elif choice == "Skin Cancer":
        st.markdown('<p class="disease-subheader">Skin Cancer</p>', unsafe_allow_html=True)

        # Upload image
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            # Define the path where you want to save the uploaded image
            img_path = os.path.join("upload/", f"img.png")

            # Save the uploaded image locally
            image.save(img_path)

            categories = ['benign', 'malignant']
            model = load_model("../models/skinCancer.keras")

            disease, img = predict(img_path, categories, model)

            # Display predicted disease and uploaded image
            st.write(f'<p class="predicted-disease">Predicted Disease: {disease}</p>', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Image", use_column_width=True)

    elif choice == "Brain Tumor":
        st.markdown('<p class="disease-subheader">Brain Tumor</p>', unsafe_allow_html=True)

        # Upload image
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            # Define the path where you want to save the uploaded image
            img_path = os.path.join("upload/", f"img.png")

            # Save the uploaded image locally
            image.save(img_path)

            categories = ['glioma', 'meningioma', 'notumor', 'pituitary']
            model = load_model("../models/brainTumor.keras")

            disease, img = predict(img_path, categories, model)

            # Display predicted disease and uploaded image
            st.write(f'<p class="predicted-disease">Predicted Disease: {disease}</p>', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Image", use_column_width=True)

    elif choice == "Breast Cancer":
        st.markdown('<p class="disease-subheader">Breast Cancer</p>', unsafe_allow_html=True)

        # Upload image
        uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            # Define the path where you want to save the uploaded image
            img_path = os.path.join("upload/", f"img.png")

            # Save the uploaded image locally
            image.save(img_path)

            categories = ['benign', 'malignant', 'normal']
            model = load_model("../models/breastCancer.keras")


            disease, img = predict(img_path, categories, model)

            # Display predicted disease and uploaded image
            st.write(f'<p class="predicted-disease">Predicted Disease: {disease}</p>', unsafe_allow_html=True)
            st.image(img, caption="Uploaded Image", use_column_width=True)


if __name__ == "__main__":
    main()
