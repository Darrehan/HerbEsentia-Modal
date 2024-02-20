from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from amazon_api import call_aws_rekognition  # Import the AWS API call function

app = Flask(__name__)
CORS(app)
model_path = 'D:/HerbEsentia/HerbEsentia Modal/trained_models/plant_classification_model.h5'
model = load_model(model_path)

def preprocess_image(image):
    try:
        img = Image.open(image).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if 'image' in request.files:
            # Handling uploaded image
            uploaded_image = request.files['image']
            image_path = 'temp.jpg'
            uploaded_image.save(image_path)
            processed_image = preprocess_image(image_path)
        elif 'imageData' in request.form:
            # Handling captured image
            image_data = request.form['imageData']
            processed_image = preprocess_image(image_data)
        else:
            raise ValueError("No valid image data provided.")

        prediction = model.predict(processed_image)[0][0]

        # Call AWS Rekognition for additional labels
        aws_labels = call_aws_rekognition(image_data=image_data if 'imageData' in request.form else None)
        # Combine the model prediction and AWS Rekognition labels
        result_labels = ['medicinal' if prediction > 0.5 else 'non-medicinal'] + aws_labels

        return jsonify({'result_labels': result_labels})

    except ValueError as ve:
        return jsonify({'error': str(ve)})
    except Exception as e:
        return jsonify({'error': "An error occurred while processing the image. Please try again later."})

@app.route('/api/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            raise ValueError("No file provided in the request.")
        
        uploaded_file = request.files['file']
        # Handle the uploaded file as needed (save, process, etc.)
        # For demonstration purposes, this example just echoes the filename.
        return jsonify({'message': f'File uploaded: {uploaded_file.filename}'})

    except ValueError as ve:
        return jsonify({'error': str(ve)})
    except Exception as e:
        return jsonify({'error': "An error occurred while processing the uploaded file. Please try again later."})

if __name__ == '__main__':
    app.run(debug=True)
