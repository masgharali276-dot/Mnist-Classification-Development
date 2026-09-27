from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI(title="MNIST Digit Classifier API")

# Load trained ANN model
model = tf.keras.models.load_model("mnist_model.h5")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    # Image load karke grayscale mein convert karein
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    # 28x28 resize karein (MNIST format)
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    # Shape set karein: (1, 28, 28)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.get("/")
def read_root():
    return {"message": "MNIST FastAPI Backend is running!"}

@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image(contents)
    
    # Model prediction
    predictions = model.predict(processed_image)
    predicted_digit = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100, 2)
    }