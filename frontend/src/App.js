import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Button, Container, Row, Col, Form, Alert } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
axios.defaults.maxBodyLength = 50 * 1024 * 1024;

const App = () => {
  const webcamRef = useRef(null);
  const [imageData, setImageData] = useState(null);
  const [resultLabels, setResultLabels] = useState(null);
  const [isCapturing, setIsCapturing] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [successMessage, setSuccessMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const capture = async () => {
    setSuccessMessage('');
    setErrorMessage('');

    if (isCapturing) {
      const imageSrc = webcamRef.current.getScreenshot();
      setImageData(imageSrc);

      try {
        // Save the image to the server
        const saveResponse = await axios.post('http://localhost:5000/api/save-image', { imageData: imageSrc });
        console.log(saveResponse.data);

        // Call ML model API and AWS Rekognition
        const mlResponse = await axios.post('http://localhost:5000/api/predict', { image: imageSrc, imageData: imageSrc });
        console.log(mlResponse.data);
        setResultLabels(mlResponse.data.result_labels);
        setSuccessMessage('Image captured and processed successfully!');
      } catch (error) {
        console.error(error);
        setErrorMessage('Error capturing and processing image. Please try again.');
      }
    } else {
      // Handle file upload logic here
      if (selectedFile) {
        try {
          const formData = new FormData();
          formData.append('file', selectedFile);

          // Replace 'http://localhost:5000/api/upload' with your server's file upload endpoint
          const uploadResponse = await axios.post('http://localhost:5000/api/upload', formData);

          console.log('File uploaded successfully:', uploadResponse.data);
          setSuccessMessage('File uploaded successfully!');
        } catch (error) {
          console.error('Error uploading file:', error);
          setErrorMessage('Error uploading file. Please try again.');
        }
      } else {
        console.warn('No file selected for upload.');
        setErrorMessage('No file selected for upload.');
      }
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  return (
    <Container className='gradient-background'>
      <h1 style={{ textAlign: "center", color: "yellow" }} className="mt-4">Medicinal Plant Recognition with Agrotech</h1>

      {successMessage && <Alert variant="success">{successMessage}</Alert>}
      {errorMessage && <Alert variant="danger">{errorMessage}</Alert>}

      <Row className="mt-4">
        <Col>
          {isCapturing ? (
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
            />
          ) : (
            <Form.Group controlId="formFile" className="mb-3">
              <Form.Label>Upload Image</Form.Label>
              <Form.Control type="file" onChange={handleFileChange} accept="image/*" />
            </Form.Group>
          )}

          <Button variant="primary" className="mt-2" onClick={capture}>
            {isCapturing ? 'Capture' : 'Upload'}
          </Button>
          <Button variant="secondary" className="mt-2 ms-2" onClick={() => setIsCapturing(!isCapturing)}>
            {isCapturing ? 'Switch to Upload' : 'Switch to Capture'}
          </Button>
        </Col>
        <Col>
          {/* this is where the captured image or uploaded image shows */}
          {imageData && <img src={imageData} alt="Captured" className="mb-2 rehansmodification" />}
          {/* This is the result response from the machine learning Model and AWS Rekognition */}
          {resultLabels && (
            <div>
              <p>Model Prediction: {resultLabels[0]}</p>
              <p>AWS Rekognition Labels: {resultLabels.slice(1).join(', ')}</p>
            </div>
          )}
        </Col>
      </Row>
    </Container>
  );
};

export default App;

