import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from algorithm import color_image_compression
import matplotlib.pyplot as plt

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
    if file_path:
        load_and_display_image(file_path)

def load_and_display_image(file_path):
    original_image, reconstructed_luminance, compressed_image = color_image_compression(file_path)
    display_images(original_image, reconstructed_luminance, compressed_image)

def display_images(original_image, reconstructed_luminance, compressed_image):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(reconstructed_luminance, cmap='gray')
    plt.title('Reconstructed Luminance')

    plt.subplot(1, 3, 3)
    plt.imshow(compressed_image)
    plt.title('Compressed Image')

    plt.show()

root = tk.Tk()
root.title("Color Image Compression Coding")

canvas = tk.Canvas(root)
canvas.pack()

open_button = tk.Button(root, text="Upload Image", command=open_file)
open_button.pack()

canvas = tk.Canvas(root)
canvas.pack()

root.mainloop()
