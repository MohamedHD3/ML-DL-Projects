# conda activate base
# pip install -U streamlit
# pip install -U plotly

# you can run your app with: streamlit run appBT.py

import streamlit as st
import cv2
import numpy as np
from keras.models import load_model

background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://www.rsm.ac.uk/media/5471621/radiology.jpg?anchor=center&amp;mode=crop&amp;width=1025&amp;height=410&amp;rnd=133397671960000000.jpg");
    background-size: 100vw 100vh;  
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)

# st.text_input("", placeholder="Streamlit CSS ")

input_style = """
<style>
input[type="text"] {
    background-color: transparent;
    color: #a19eae;  
}
div[data-baseweb="base-input"] {
    background-color: transparent !important;
}
[data-testid="stAppViewContainer"] {
    background-color: transparent !important;
}
[data-testid="stHeader"]{
    background-color: transparent !important;
}
</style>
"""
st.markdown(input_style, unsafe_allow_html=True)

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