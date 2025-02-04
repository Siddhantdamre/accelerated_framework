# ai_accelerated_framework/main.py

import numpy as np
import tensorflow as tf
from data.loader import load_data
from models.base_model import build_model
from explainability.grad_cam import compute_gradcam, display_gradcam
from explainability.shap_explain import explain_with_shap
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
from tensorflow.keras.datasets import mnist

# Step 1: Load and Normalize Data
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to 0-1 range
    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

# Step 2: Data Augmentation
def augment_data(x_train, y_train):
    datagen = ImageDataGenerator(
        rotation_range=10, 
        width_shift_range=0.1, 
        height_shift_range=0.1
    )
    datagen.fit(x_train)
    return datagen.flow(x_train, y_train, batch_size=32)

# Step 3: Build a Simple Neural Network
def build_model(hp=None):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(
            hp.Int('units', min_value=32, max_value=512, step=32) if hp else 128,
            activation='relu'
        ),
        Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Step 4: Hyperparameter Tuning
def tune_hyperparameters(x_train, y_train, x_test, y_test):
    tuner = kt.RandomSearch(
        build_model,  # Pass the model function
        objective='val_accuracy', 
        max_trials=5, 
        directory='tuning_results', 
        project_name='mnist_tuning'
    )
    tuner.search(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
    return tuner

# Step 5: Training and Testing
def main():
    # 1. Load Data
    (x_train, y_train), (x_test, y_test) = load_data()

    # 2. Augment Data
    train_gen = augment_data(x_train, y_train)

    # 3. Hyperparameter Tuning
    print("\nStarting hyperparameter tuning...")
    tuner = tune_hyperparameters(x_train, y_train, x_test, y_test)
    best_model = tuner.get_best_models(num_models=1)[0]
    print("\nBest model summary:")
    best_model.summary()

    # 4. Train Best Model on Augmented Data
    print("\nTraining the best model with augmented data...")
    best_model.fit(train_gen, epochs=5, validation_data=(x_test, y_test))
    
     # 2.2. Build and Train the Model
    model = build_model()
    model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))

    # 2.3. Grad-CAM Explainability
    print("\nGenerating Grad-CAM visualization...")
    sample_image = np.expand_dims(x_test[0], axis=0)  # Take one test sample
    sample_image = np.expand_dims(sample_image, axis=-1)  # Add channel dimension
    heatmap = compute_gradcam(model, sample_image, last_conv_layer_name="flatten")  # Replace with the actual layer name
    display_gradcam(x_test[0], heatmap)

    # 2.4. SHAP Explainability
    print("\nGenerating SHAP explainability...")
    sample_data = x_test[:100]  # Take a batch of samples
    sample_data = np.expand_dims(sample_data, axis=-1)  # Add channel dimension
    explain_with_shap(model, sample_data)

if __name__ == "__main__":
    main()
