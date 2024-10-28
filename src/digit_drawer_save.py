import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import os

class DigitDrawerSave:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Digit Drawer - Save to Data Folder")

        # Canvas size
        self.canvas_width = 400
        self.canvas_height = 400
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack()

        # Create a white image
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Buttons
        self.save_button = tk.Button(self.root, text="Save", command=self.save_digit)
        self.save_button.pack()

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        
    def paint(self, event):
        x, y = event.x, event.y
        brush_size = 10  # Brush size for drawing
        self.canvas.create_oval(x - brush_size, y - brush_size, x + brush_size, y + brush_size, fill='black', outline='black')
        self.draw.ellipse((x - brush_size, y - brush_size, x + brush_size, y + brush_size), fill='black')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)  # Reset to white image
        self.draw = ImageDraw.Draw(self.image)

    def save_digit(self):
        # Save the drawn image
        save_path = './data'
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create data folder if it doesn't exist

        # Get the filename from user input
        filename = "digit.png"  # Default filename
        self.image.save(os.path.join(save_path, filename))  # Save the image
        messagebox.showinfo("Saved", f"Digit saved as {filename} in the 'data' folder.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DigitDrawerSave()
    app.run()
