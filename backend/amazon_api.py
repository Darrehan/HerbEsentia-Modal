# backend/aws_api.py
import boto3
import base64

def call_aws_rekognition(image_data):
    client = boto3.client('rekognition', aws_access_key_id='YOUR_ACCESS_KEY',
                          aws_secret_access_key='YOUR_SECRET_KEY', region_name='YOUR_REGION')

    # Convert the base64 image data to bytes
    image_bytes = base64.b64decode(image_data.split(',')[1])

    # Call AWS Rekognition
    response = client.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=5)

    # Extract labels from the response
    labels = [label['Name'] for label in response['Labels']]

    return labels
