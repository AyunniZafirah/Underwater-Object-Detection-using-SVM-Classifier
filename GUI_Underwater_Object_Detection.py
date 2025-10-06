  #latest
##########################REQUIRED LIBRARIES#############################
# pip install numpy
# pip install opencv-python
# pip install matplotlib
# pip install scikit-learn
# pip install Pillow
#########################################################################
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.svm import SVC
from tkinter import Tk, Button, Label, filedialog, Canvas
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split

dir = "C:\\Users\\Acer\\Desktop\\projects\\Underwater Object Detection\\datasets_underwater_object\\train"
categories = ['fish', 'jellyfish', 'penguin','puffin', 'shark', 'starfish', 'stingray']
data = []

#########################################################################

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)


#preprocessing data#
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        animals_img = cv2.imread(imgpath,0)
        try:
            animals_img = cv2.resize(animals_img,(50,50))
            image = np.array(animals_img).flatten()

            data.append([image, label])
        except Exception as e:
            pass
    

pick_in = open('data2.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20)


model = SVC(C=1,kernel='poly',gamma='auto')
model.fit(xtrain, ytrain)

pick=open('MODEL.sav','wb')
pickle.dump(model,pick)
pick.close()

# Initialize Tkinter
root = Tk()
root.title("Underwater Object Detection - Group 7")  # Added title and group information

# Set background image
background_image = Image.open("C:\\Users\\Acer\\Desktop\\projects\\background image.jpg")
background_photo = ImageTk.PhotoImage(background_image)
background_label = Label(root, image=background_photo)
background_label.place(relx=0.5, rely=0.5, anchor="center")  # Center the background image

# Set the window size to match the image dimensions
window_width = background_image.width
window_height = background_image.height
root.geometry(f"{window_width}x{window_height}")

# Title
title_label = Label(root, text="Underwater Object Detection", font=("Times New Roman", 20, "bold"), bg="#6B9AC4")
title_label.place(relx=0.5, rely=0.08, anchor="center")  # Center the title
group_label = Label(root, text="GROUP 7", font=("Times New Roman", 14), bg="#6B9AC4")
group_label.place(relx=0.5, rely=0.17, anchor="center")  # Center the group label

# Variables
selected_image_path = ""
selected_image_label = Label(root, text="Selected Image: None", bg="lightblue")
result_label = Label(root, text="Prediction: ", bg="lightblue", font=("Rubik", 16,"bold"))
image_label = Label(root, bg="lightblue")

# Function to open a file dialog and get the selected image path
def open_file_dialog():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename()
    selected_image_label.config(text="Image Path : " + selected_image_path)

# Function to classify the selected image

def classify_image():
    global selected_image_path
    if selected_image_path:
        animals_img = cv2.imread(selected_image_path, 0)
        try:
            
            animals_img = cv2.resize(animals_img, (50, 50))
            image = np.array(animals_img).flatten()
            prediction = model.predict([image])
            result_label.config(text="Prediction: " + categories[prediction[0]])

            mypet = animals_img.reshape(50, 50)
            plt.imshow(mypet, cmap='gray')
            plt.show()
        except Exception as e:
            result_label.config(text="Prediction: Error processing image")
    else:
        result_label.config(text="Prediction: Please select an image first")

# Function to display help information
def display_help():
    help_text = """
    Welcome to Underwater Object Detection - Group 7 Interface!

    1. Click "Select Image" to choose an image for classification.
    2. After selecting an image, click "Classify Image" to see the prediction.
    3. The prediction result will be displayed below the buttons.

    """
    help_window = Tk()
    help_window.title("Help - Underwater Object Detection - Group 7")

    help_label = Label(help_window, bg="lightblue", text=help_text, padx=20, pady=20)
    help_label.pack()

# Button to open file dialog
select_button = Button(root, text="Select Image", command=open_file_dialog, width=20, height=2)  # Increased button size
select_button.place(relx=0.5, rely=0.35, anchor="center")  # Center the button

# Selected image path label
selected_image_label.place(relx=0.5, rely=0.45, anchor="center")  # Center the label

# Button to classify the selected image
classify_button = Button(root, text="Classify Image", command=classify_image, width=20, height=2)  # Increased button size
classify_button.place(relx=0.5, rely=0.60, anchor="center")  # Center the button

# Button for help
help_button = Button(root, text="Help",  bg="gray", command=display_help, width=10, height=1)
help_button.place(relx=0.05, rely=0.95, anchor="sw")  # Bottom-left corner

# Layout
result_label.place(relx=0.5, rely=0.75, anchor="center")  # Center the label
# Run Tkinter main loop
root.mainloop()
