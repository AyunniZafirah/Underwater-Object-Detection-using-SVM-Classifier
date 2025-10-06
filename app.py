from flask import Flask, render_template, request, url_for
import cv2
import numpy as np
import pickle
import os

# Load trained model
model = pickle.load(open("MODEL.sav", "rb"))
categories = ['fish', 'jellyfish', 'penguin', 'puffin', 'shark', 'starfish', 'stingray']

# Setup app and upload folder
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            # Save uploaded image so we can display it later
            filepath = os.path.join(UPLOAD_FOLDER, "last.jpg")
            file.save(filepath)
            image_url = url_for('static', filename="uploads/last.jpg")

            # Read image for prediction
            img = cv2.imdecode(np.frombuffer(open(filepath, "rb").read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (50, 50))
            image = np.array(img).flatten()

            # Predict
            pred = model.predict([image])[0]
            prediction = categories[pred]

    return render_template("index.html", prediction=prediction, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
