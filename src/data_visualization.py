import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
train_df = pd.read_csv('data/mnist_train.csv')

# Separate features and labels
X_train_flat = train_df.drop('label', axis=1).values
y_train = train_df['label'].values

# Reshape the images back to 28x28
X_train = X_train_flat.reshape(X_train_flat.shape[0], 28, 28)

# Plot the first 25 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])

plt.show()