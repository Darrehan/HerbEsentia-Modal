# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from amazon_api import call_aws_rekognition  # Import the AWS API call function

app = Flask(__name__)
CORS(app)
model = load_model(r'D:\HerbEsentia\HerbEsentia Modal\trained_models\plant_classification_model.h5')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        image = request.files['image']
        image_path = 'temp.jpg'
        image.save(image_path)
        processed_image = preprocess_image(image_path)
        prediction = model.predict(processed_image)[0][0]

        # Call AWS Rekognition for additional labels
        aws_labels = call_aws_rekognition(image_data=request.form['imageData'])
        # Combine the model prediction and AWS Rekognition labels
        result_labels = ['medicinal' if prediction > 0.5 else 'non-medicinal'] + aws_labels

        return jsonify({'result_labels': result_labels})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
