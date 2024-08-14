# conda activate base
# pip install -U streamlit
# pip install -U plotly

# you can run your app with: streamlit run appBT.py

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('BrainTumor_1.h5')  # Replace with the path to your saved model

st.title('Brain Tumor Detection 🧠')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read and preprocess the image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.resize(image, (128, 128))  # Resize to match the input size of your model
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = image.astype(np.float32) / 255.0  # Normalize pixel values

    # Reshape the image to match the input shape expected by your model
    image = np.expand_dims(image, axis=0)

    # Make predictions
    prediction = model.predict(image)

    # Interpret the prediction
    result_text = ''
    if prediction[0, 0] > 0.5:
        result_text = 'The model predicts that the image contains a tumor.'
    else:
        result_text = 'The model predicts that the image does not contain a tumor.'
        st.balloons()

    # Display the result
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write(result_text)