import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(images, true_labels, predictions, num_images=10):
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].reshape(28, 28), cmap=plt.cm.binary)
        # Show true label and predicted label
        plt.xlabel(f'True: {true_labels[i]}\nPred: {np.argmax(predictions[i])}')
    plt.show()
