// src/App.js
import "./App.css"
import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Button, Container, Row, Col } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
axios.defaults.maxBodyLength = 50 * 1024 * 1024;

const App = () => {
  const webcamRef = useRef(null);
  const [imageData, setImageData] = useState(null);
  const [resultLabels, setResultLabels] = useState(null);

  const capture = async () => {
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
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <Container className='gradient-background'>
      <h1 style={{ textAlign: "center", color: "yellow" }} className="mt-4">MedicinalPlant Recognition with Agrotech</h1>

      <Row className="mt-4">
        <Col >
          <Webcam
            audio={false}
            ref={webcamRef}
            screenshotFormat="image/jpeg"
          />
          <Button variant="primary" className="mt-2" onClick={capture}>
            Capture
          </Button>
        </Col>
        <Col>
          {/* this is where the captured umage shows */}
          {imageData && <img src={imageData} alt="Captured" className="mb-2 rehansmodification" />}
          {/* This is the result response from machine learning Model and AWS Rekognition */}
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
