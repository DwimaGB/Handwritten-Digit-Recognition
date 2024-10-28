import tkinter as tk
from tkinter import messagebox
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt

class DigitDrawer:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Digit Drawer")
        
        # Increase the canvas size
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()
        
        # Create a larger white image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.predict_digit)
        
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        
    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-10, y-10, x+10, y+10, fill='black', outline='black')
        self.draw.ellipse((x-10, y-10, x+10, y+10), fill='black')
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)  # Reset to white image
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self, event):
        # Prepare the image for prediction
        img = self.image.resize((28, 28))  # Resize to 28x28 for the model
        img = np.invert(np.array(img))  # Invert the colors
        img = img / 255.0  # Normalize to [0, 1]
        img = img.reshape(1, 28, 28, 1)  # Reshape for model input

        # Display the drawn image
        plt.subplot(1, 3, 1)  # Drawn Image
        plt.imshow(np.array(self.image), cmap='gray')  # Show original drawn image
        plt.title("Drawn Image")
        plt.axis('off')

        # Predict the digit
        predictions = self.model.predict(img)
        predicted_digit = np.argmax(predictions[0])

        # Show the prediction
        messagebox.showinfo("Prediction", f"The predicted digit is: {predicted_digit}")

        # Automatically clear the canvas after prediction
        self.clear_canvas()

    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    # Load the trained model
    model = tf.keras.models.load_model('./models/model.h5')  # Adjust path based on your structure
    app = DigitDrawer(model)
    app.run()