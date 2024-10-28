# Import libraries
import os
import pandas as pd
from tensorflow import keras
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuDNN.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuFFT.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*cuBLAS.*")

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape the data for saving (flatten images)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Create DataFrames
train_df = pd.DataFrame(X_train_flat)
train_df['label'] = y_train

test_df = pd.DataFrame(X_test_flat)
test_df['label'] = y_test

# Create the data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save DataFrames as CSV in the data folder
train_df.to_csv('data/mnist_train.csv', index=False)
test_df.to_csv('data/mnist_test.csv', index=False)