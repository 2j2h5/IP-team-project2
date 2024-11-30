from PIL import Image, ImageTk
from simple_conv_net import SimpleConvNet
from hangul_conv_net import HangulConvNet
from classifier import Classifier
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import plate_focus
import os

# Call for the classification function
def do_classification():
    classifier = Classifier(network, hangul_network)
    plate_focus.plate_focus(preview_label.file_path)
    plate = Image.open('plate.jpg').convert('L')
    if os.path.exists("plate.jpg"):
        os.remove("plate.jpg")
    classifier.set_image(plate)
    license_number = classifier.get_license_plate()
    print(license_number)
    vehicle_type = classifier.parse_license_plate()
    print(vehicle_type)

# import image from file dialog
def import_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        do_classification.image_path = file_path
        load_preview(file_path, preview_label)

# load image to preview label
def load_preview(file_path, preview_label):
    try:
        image = Image.open(file_path)
        image.thumbnail((200, 200))
        tk_image = ImageTk.PhotoImage(image)
        preview_label.config(image=tk_image)
        preview_label.image = tk_image
        preview_label.file_path = file_path
        preview_label.pack()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load image: {e}")

#prepare SimpleConvNet
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("params.pkl")

#prepare HangulConvNet
hangul_network = HangulConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=54, weight_init_std=0.01)
hangul_network.load_params("hangul-params.pkl")

#prepare image

# image_path = 'images/number-plate/number-plate-clean-1.jpg'
# image_path2 = 'images/number-plate/number-plate-clean-2.jpg'
# image_path3 = 'images/number-plate/number-plate-clean-3.jpg'

# plate_image = Image.open(image_path).convert('L')
# plate_image2 = Image.open(image_path2).convert('L')
# plate_image3 = Image.open(image_path3).convert('L')

# Build GUI
root = tk.Tk()
root.title("Image Processing group project 1")

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

import_button = tk.Button(button_frame, text="Import Image", command=import_image)
import_button.pack(side=tk.LEFT, padx=5)

process_button = tk.Button(button_frame, text="Process", command=do_classification)
process_button.pack(side=tk.LEFT, padx=5)

preview_label = tk.Label(root)
preview_label.pack(pady=10)

root.mainloop()

