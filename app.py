import os
import time
from flask import Flask, render_template, request, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Disable GPU if not available
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Upload folder
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Disease categories and prescriptions
class_labels = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Scald', 'Narrow Brown Spot']
prescriptions = {
    'Bacterial Leaf Blight': 'Use Streptomycin and copper-based fungicides.',
    'Brown Spot': 'Apply nitrogen fertilizers and improve drainage.',
    'Healthy': 'No action required.',
    'Leaf Blast': 'Apply fungicides like Propiconazole.',
    'Leaf Scald': 'Use resistant varieties and copper-based fungicides.',
    'Narrow Brown Spot': 'Increase nitrogen fertilizer and improve drainage.'
}

# Load the model once at startup
MODEL_PATH = os.path.join("model", "rice_disease_model.h5")
model = load_model(MODEL_PATH, compile=False)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to predict disease
def predict_disease(image_path):
    image = Image.open(image_path).resize((150, 150))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)

    disease = class_labels[class_idx]
    prescription = prescriptions[disease]

    return disease, prescription

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    disease = None
    prescription = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("file")
        if file and allowed_file(file.filename):
            filename = f"{int(time.time())}_{file.filename}"  # Unique filename
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(file_path)

            disease, prescription = predict_disease(file_path)
            image_url = url_for('static', filename=f'uploads/{filename}')
        else:
            return render_template("predict.html", error="Invalid file format. Please upload PNG, JPG, or JPEG.")

    return render_template("predict.html", disease=disease, prescription=prescription, image_url=image_url)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if not specified
    app.run(host="0.0.0.0", port=port, debug=True)
