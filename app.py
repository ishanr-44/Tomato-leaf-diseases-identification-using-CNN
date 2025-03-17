from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = 'tomatomodel.h5'

# Load model
model = load_model(MODEL_PATH)

# Class names (update according to your model's classes)
CLASS_NAMES = ['Tomato___Bacterial_spot',
               'Tomato___Early_blight',
               'Tomato___Late_blight',
               'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
               'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_disease(img_path):
    """
    Preprocess image and make prediction
    """
    # Load and resize image
    img = Image.open(img_path).resize((256, 256))

    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(np.max(predictions[0]) * 100, 2)

    return predicted_class, confidence


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Make prediction
            prediction, confidence = predict_disease(file_path)

            return render_template('index.html',
                                   prediction=prediction,
                                   confidence=confidence,
                                   img_path=file_path)

    return render_template('index.html')


if __name__ == '__main__':
    # Create upload folder if not exists
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)