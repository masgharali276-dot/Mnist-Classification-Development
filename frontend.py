import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢")

st.title("🔢 MNIST Handwritten Digit Recognition")
st.write("ANN Model with FastAPI Backend & Streamlit Frontend")

FASTAPI_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Upload a Handwritten Digit Image (28x28 Grayscale)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    if st.button("Classify Digit"):
        # Convert PIL image to bytes for API upload
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()

        files = {"file": ("image.png", img_bytes, "image/png")}
        
        try:
            with st.spinner("Classifying..."):
                response = requests.post(FASTAPI_URL, files=files)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Predicted Digit:** {result['predicted_digit']}")
                    st.info(f"**Confidence:** {result['confidence']}%")
                else:
                    st.error("Error in API response.")
        except Exception as e:
            st.error(f"Failed to connect to FastAPI backend: {e}")