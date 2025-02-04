# data/loader.py
import numpy as np
from tensorflow.keras.datasets import mnist

def load_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize data (0-1 range)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(f"Training data shape: {x_train.shape}")
    print(f"Testing data shape: {x_test.shape}")

    return (x_train, y_train), (x_test, y_test)
