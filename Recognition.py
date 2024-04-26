import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from skimage import color
from skimage.feature import hog
import numpy as np
import pandas as pd
from main import MyDenseNet

# Load the model and the flower names
model = MyDenseNet()
state_dict = torch.load('best_model_flowers.pth', map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
model.load_state_dict(state_dict)
model.eval()
flower_names = pd.read_csv('flower_102_name.csv', index_col='Index')

# Preprocessing of the images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_path):
    image = Image.open(image_path)
    image = transform(image)
    image_np = np.array(image.permute(1, 2, 0))
    image_hsv = color.rgb2hsv(image_np)
    fd, _ = hog(image_hsv, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), channel_axis=-1, visualize=True)
    fd = torch.tensor(fd, dtype=torch.float32).unsqueeze(0)
    return image.unsqueeze(0), fd

# GUI
window = tk.Tk()
window.title("Flower Recognizer")

def load_image():
    file_path = filedialog.askopenfilename()
    image, fd = process_image(file_path)
    output, index = predict(image, fd)
    show_result(output, index, file_path)

def predict(image, fd):
    outputs = model(image, fd)
    _, preds = torch.max(outputs, 1)
    index = preds.item()
    flower_name = flower_names.loc[index+1, 'Name']
    return flower_name, index

def show_result(flower_name, index, file_path):
    label.config(text=f"Predicted Flower: {flower_name}")
    img = Image.open(file_path)
    img = img.resize((250, 250), Image.Resampling.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img


btn_load = tk.Button(window, text="Load Image", command=load_image)
btn_load.pack()

label = tk.Label(window, text="Please load an image.", font=('Arial', 14))
label.pack()

panel = tk.Label(window)
panel.pack()

window.mainloop()
