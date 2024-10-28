import tensorflow as tf
import pandas as pd
import numpy as np
from utils.visualization import visualize_predictions  # Import the visualization function


# Load the test dataset from CSV
test_df = pd.read_csv('./data/mnist_test.csv') 
X_test_flat = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Reshape the images back to 28x28 and add channel dimension
X_test = X_test_flat.reshape(X_test_flat.shape[0], 28, 28, 1)

# Load the trained model
model = tf.keras.models.load_model('./models/model.h5')  

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Optionally, you can also make predictions and visualize results
predictions = model.predict(X_test)

# Example: Print the first 5 predictions
# print("First 5 predictions:", np.argmax(predictions[:5], axis=1))

visualize_predictions(X_test, y_test, predictions, 10)
