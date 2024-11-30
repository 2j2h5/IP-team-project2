from PIL import Image, ImageTk
from simple_conv_net import SimpleConvNet
from hangul_conv_net import HangulConvNet
from classifier import Classifier
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import plate_focus
import os

# Call for the classification function==============================================
def do_classification():
    classifier = Classifier(network, hangul_network) # prepare our Classifier

    plate_focus.plate_focus(preview_label.file_path) # YOLO model detect car number plate and save as 'plate.jpg' after grayscaling, binarizing, denoizing
    plate = Image.open('plate.jpg').convert('L')
    # plate.show() #if you want to look result of YOLO models detection, just decomment plate.show()
    if os.path.exists("plate.jpg"):
        os.remove("plate.jpg")
    
    classifier.set_image(plate) # our classfier do classfy what the number of plate using Simple Conv Net, Hangul Conv Net
    license_number = classifier.get_license_plate()
    vehicle_type = classifier.parse_license_plate()

    license_label.config(text=f"License Number: {license_number}") # visualize result of our classifier on to tkinter GUI
    vehicle_label.config(text=f"Vehicle Type: {vehicle_type['vehicleBodyType']}")
    usage_label.config(text=f"Usage: {vehicle_type['usage']}")
    

# Import image from file dialog======================================================
def import_image():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
    if file_path:
        do_classification.image_path = file_path
        load_preview(file_path, preview_label)

    license_label.config(text="License Number: ") # clear labels if there is new image unclassified
    vehicle_label.config(text="Vehicle Type: ")
    usage_label.config(text="Usage: ")


# Load image to preview label=========================================================
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


# Prepare CNN models==================================================================================
# prepare SimpleConvNet
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params("params.pkl") #use pre-trained parameter

# prepare HangulConvNet
hangul_network = HangulConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=500, output_size=54, weight_init_std=0.01)
hangul_network.load_params("hangul-params.pkl") #use pre-trained parameter


# Build GUI===========================================================================================
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

license_label = tk.Label(root, text="License Number: ", font=("", 10))
license_label.pack(pady=5)

vehicle_label = tk.Label(root, text="Vehicle Type: ", font=("", 10))
vehicle_label.pack(pady=5)

usage_label = tk.Label(root, text="Usage: ", font=("", 10))
usage_label.pack(pady=5)

root.mainloop()