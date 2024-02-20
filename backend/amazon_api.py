# backend/aws_api.py
import boto3
import base64
import os

def call_aws_rekognition(image_data):
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')

    if not all([aws_access_key_id, aws_secret_access_key, aws_region]):
        raise ValueError("AWS credentials and region are not provided.")

    client = boto3.client('rekognition', aws_access_key_id=aws_access_key_id,
                          aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

    try:
        # Convert the base64 image data to bytes
        image_bytes = base64.b64decode(image_data.split(',')[1])

        # Call AWS Rekognition
        response = client.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=5)

        # Extract labels from the response
        labels = [label['Name'] for label in response['Labels']]
        return labels

    except Exception as e:
        print(f"Error calling AWS Rekognition: {str(e)}")
        return []
