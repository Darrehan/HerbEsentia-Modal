# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('trained_models/medicinal_model.h5')

# Define image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get the image from the request
        image = request.files['image']
        image_path = 'temp.jpg'
        image.save(image_path)

        # Preprocess the image
        processed_image = preprocess_image(image_path)

        # Make prediction
        prediction = model.predict(processed_image)[0][0]

        # Return the result
        return jsonify({'result': 'medicinal' if prediction > 0.5 else 'non-medicinal'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
