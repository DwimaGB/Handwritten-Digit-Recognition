import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# Load the training data from CSV
train_df = pd.read_csv('./data/mnist_train.csv')

# Separate features and labels
X_train_flat = train_df.drop('label', axis=1).values  # All columns except 'label'
y_train = train_df['label'].values  # 'label' column

# Load the test data from CSV
test_df = pd.read_csv('./data/mnist_test.csv')

# Separate features and labels
X_test_flat = test_df.drop('label', axis=1).values  # All columns except 'label'
y_test = test_df['label'].values  # 'label' column

# Reshape the images back to 28x28 and add channel dimension
X_train = X_train_flat.reshape(X_train_flat.shape[0], 28, 28, 1)  # Shape: (num_samples, 28, 28, 1)
X_test = X_test_flat.reshape(X_test_flat.shape[0], 28, 28, 1)  # Shape: (num_samples, 28, 28, 1)

# Normalize the pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Create a CNN model
model = tf.keras.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Input layer
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),  # First convolutional layer
    layers.MaxPooling2D(),  # First pooling layer
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),  # Second convolutional layer
    layers.MaxPooling2D(),  # Second pooling layer
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),  # Third convolutional layer
    layers.Flatten(),  # Flatten layer
    layers.Dense(128, activation='relu'),  # Fully connected layer with more neurons
    layers.Dense(10, activation='softmax')  # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)  # Increased epochs to 10
accuracy, loss = model.evaluate(X_test, y_test)

# Print accuracy
print(f'Test accuracy: {accuracy:.4f}')

# Save the model
model.save('./models/model.h5')
