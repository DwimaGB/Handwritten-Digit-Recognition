import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# Load the training data from CSV
train_df = pd.read_csv('./data/mnist_train.csv')
X_train_flat = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

# Load the test data from CSV
test_df = pd.read_csv('./data/mnist_test.csv')
X_test_flat = test_df.drop('label', axis=1).values
y_test = test_df['label'].values

# Reshape the images back to 28x28 and add channel dimension
X_train = X_train_flat.reshape(X_train_flat.shape[0], 28, 28, 1).astype('float32') / 255.0
X_test = X_test_flat.reshape(X_test_flat.shape[0], 28, 28, 1).astype('float32') / 255.0

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
])

# Create a CNN model with Dropout and L2 regularization
model = tf.keras.Sequential([
    data_augmentation,
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    layers.Flatten(),
    layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the Model
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2)
accuracy, loss = model.evaluate(X_test, y_test)

# Print accuracy
print(f'Test accuracy: {accuracy:.4f}')

# Save the model
model.save('./models/model.h5')
