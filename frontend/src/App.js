// src/App.js
import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import { Button, Container, Row, Col } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

axios.defaults.maxBodyLength = 50 * 1024 * 1024; // 50 MB (change as needed)

const App = () => {
  const webcamRef = useRef(null);
  const [imageData, setImageData] = useState(null);
  const [result, setResult] = useState(null);

  const capture = async () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImageData(imageSrc);

    try {
      // Save the image to the server
      const saveResponse = await axios.post('http://localhost:5000/api/save-image', { imageData: imageSrc });
      console.log(saveResponse.data);

      // Call ML model API
      const mlResponse = await axios.post('http://localhost:5000/api/predict', { image: imageSrc });
      console.log(mlResponse.data);
      setResult(mlResponse.data.result);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <Container>
      <h1 className="mt-4">Medicinal Image Recognition</h1>
      <Row className="mt-4">
        <Col>
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
          {imageData && <img src={imageData} alt="Captured" className="mb-2" />}
          {result && <p>Result: {result}</p>}
        </Col>
      </Row>
    </Container>
  );
};

export default App;
