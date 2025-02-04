from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, validator
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from explainability.grad_cam import compute_gradcam
import matplotlib.pyplot as plt
import io
import base64
import os
from typing import List

app = FastAPI()

# Global variables
MODEL_PATH = "mnist_model.h5"
model = None
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# Request/Response Models
class TrainRequest(BaseModel):
    epochs: int

class PredictionRequest(BaseModel):
    images: List[List[float]]  # List of images (either 28x28 or 784)

    @validator("images", pre=True, each_item=True)
    def check_image_shape(cls, image):
        # Allow both (28,28) and (784) but reject anything else
        if len(image) not in [28, 784]:
            raise ValueError("Each image must have 784 (flattened) or 28x28 pixels.")
        return image
    
# Utility Functions
def build_model():
    global model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def load_trained_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        build_model()

def train_model(epochs: int):
    global model
    load_trained_model()  # Load model if exists, otherwise build new
    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))
    model.save(MODEL_PATH)  # Save model after training

def predict_images(images):
    images = np.array(images, dtype=np.float32)  # Ensure it's a NumPy array

    # 🛠️ Step 1: Handle Flattened Inputs (28x28 vs. 784)
    if images.ndim == 2 and images.shape[1] == 784:  # If input is (N, 784), reshape it
        images = images.reshape(-1, 28, 28)
    elif images.ndim == 3 and images.shape[1:] == (28, 28):  # If already (N, 28, 28), do nothing
        pass
    else:
        raise ValueError(f"Invalid input shape: {images.shape}. Expected (N, 784) or (N, 28, 28).")

    # 🛠️ Step 2: Normalize Data Automatically
    images = images / 255.0  # Ensure values are between 0 and 1

    # 🛠️ Step 3: Predict and Return Output
    predictions = model.predict(images)
    return np.argmax(predictions, axis=1).tolist()

def generate_gradcam(image):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not trained. Train the model first.")
    
    # Using a workaround with Grad-CAM on a fully connected model
    # Grad-CAM is intended for CNNs, so we'll directly apply it to the output layer
    heatmap = compute_gradcam(model, np.expand_dims(image, axis=0), last_conv_layer_name="dense")
    buf = io.BytesIO()
    plt.imshow(heatmap)
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to the AI-Accelerated Deep Learning Framework!"}

@app.post("/predict/")  # Prediction API
def predict(request: PredictionRequest):
    try:
        predictions = predict_images(request.images)
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))  # Return a user-friendly message
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred.")

@app.post("/train/")  # Training API
def train(request: TrainRequest):
    try:
        train_model(request.epochs)
        return {"message": f"Model trained for {request.epochs} epochs."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/gradcam/")  # Grad-CAM API
async def generate_gradcam_endpoint(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()  # Read the image data
        img = Image.open(io.BytesIO(img_bytes)).convert('L').resize((28, 28))  # Convert to grayscale and resize
        img_array = np.array(img) / 255.0  # Normalize
        heatmap_base64 = generate_gradcam(img_array)
        return {"filename": image.filename, "gradcam": heatmap_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")
