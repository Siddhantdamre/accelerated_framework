# models/base_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def build_model():
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images into a vector
        Dense(128, activation='relu'), # Fully connected layer with 128 neurons
        Dense(10, activation='softmax') # Output layer for 10 classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
