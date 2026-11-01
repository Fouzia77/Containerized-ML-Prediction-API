# Containerized ML Prediction API

A containerized image classification REST API built using **Python, TensorFlow/Keras, FastAPI, Docker, and GitHub Actions**. The application accepts JPEG and PNG images and returns the predicted class label along with class probabilities.

## 1. Project Overview

This project demonstrates how to develop and deploy a machine learning model through a REST API. Users can upload an image, and the API processes the image and returns a prediction in JSON format.

### Key Features

- Image classification using a Keras model.
- FastAPI-based REST API.
- Image preprocessing using Pillow.
- Model loaded once during application startup.
- Input validation and error handling.
- Docker containerization.
- Docker Compose configuration.
- Automated testing using pytest.
- GitHub Actions CI workflow.

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Backend programming |
| TensorFlow / Keras | Machine learning model |
| FastAPI | REST API development |
| NumPy | Numerical operations |
| Pillow | Image processing |
| Docker | Application containerization |
| Docker Compose | Container management |
| Pytest | Automated testing |
| GitHub Actions | Continuous integration |

## 3. Project Structure

```text
ml_prediction_api/
│
├── .github/
│   └── workflows/
│       └── main.yml
│
├── models/
│   ├── my_classifier.h5
│   └── README.md
│
├── predictions/
│   └── example.json
│
├── scripts/
│   └── create_model.py
│
├── src/
│   ├── __init__.py
│   ├── main.py
│   └── model.py
│
├── tests/
│   ├── __init__.py
│   └── test_api.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 4. Installation and Setup

### Prerequisites

- Python 3.11
- Docker Desktop
- Docker Compose
- Git

### Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ml_prediction_api
```

### Configure environment variables

```bash
cp .env.example .env
```

The environment configuration contains:

```env
MODEL_PATH=models/my_classifier.h5
LOG_LEVEL=INFO
```

## 5. Run the Application Using Docker

Build the Docker image and start the application:

```bash
docker compose up --build -d
```

Check the running container:

```bash
docker compose ps
```

View application logs:

```bash
docker compose logs -f ml_api
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Stop the application:

```bash
docker compose down
```

## 6. API Endpoints

### A. Health Check

**Endpoint:** `GET /health`

Checks whether the API is running.

Request:

```bash
curl -i http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok"
}
```

HTTP Status: `200 OK`

### B. Image Prediction

**Endpoint:** `POST /predict`

Accepts an image through multipart form-data.

- Parameter name: `file`
- Supported formats: JPEG and PNG

Request:

```bash
curl -i -X POST http://localhost:8000/predict \
  -F "file=@sample.png;type=image/png"
```

Example response:

```json
{
  "class_label": "automobile",
  "probabilities": [
    0.10,
    0.25,
    0.08,
    0.07,
    0.06,
    0.09,
    0.08,
    0.07,
    0.10,
    0.10
  ]
}
```

The response contains:
- `class_label`: Predicted class.
- `probabilities`: Probability values for the model's output classes.

The example values are illustrative; actual results depend on the model and input image.

## 7. Image Preprocessing

Before prediction, the uploaded image goes through the following steps:

1. Read the uploaded image.
2. Convert the image into RGB format.
3. Resize it to 64 × 64 pixels.
4. Convert image data into a NumPy array.
5. Normalize pixel values between 0 and 1.
6. Add a batch dimension.
7. Pass the processed image to the Keras model.

## 8. Error Handling

The API handles different types of errors:

| Situation | HTTP Status |
|---|---|
| Successful prediction | 200 |
| Unsupported file type | 400 |
| Missing file | 422 |
| Invalid or corrupt image | 422 |
| Unexpected prediction error | 500 |

Errors are returned as JSON responses.

## 9. Model Information

The model artifact is stored at:

```text
models/my_classifier.h5
```

The model accepts images with the shape:

```text
(None, 64, 64, 3)
```

It produces 10 output classes:

```text
(None, 10)
```

The current model is a randomly initialized demonstration model generated using:

```bash
python scripts/create_model.py
```

**Note:** This model validates the image-processing and prediction-serving pipeline. It is not trained on an image dataset, so its predictions should not be considered accurate or meaningful. A trained classifier should replace it for real-world classification.

## 10. Running Tests

Install the dependencies in a compatible Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
```

Run the test suite:

```bash
pytest -q tests/
```

The tests cover:

- Health endpoint.
- Successful mocked prediction.
- Unsupported image content type.
- Invalid image handling.
- Unexpected prediction errors.

The prediction tests mock model inference so that TensorFlow inference is not required during those tests.

## 11. Continuous Integration

The GitHub Actions workflow is located at:

```text
.github/workflows/main.yml
```

It runs when code is pushed to or a pull request targets the `main` branch.

The workflow performs:

1. Repository checkout.
2. Python 3.11 setup.
3. Dependency installation.
4. Pytest execution.
5. Docker image build.
6. Simulated image push step.

The workflow demonstrates automated testing and Docker image building. It does not publish an image to a container registry.

## 12. Configuration

The application supports the following environment variables:

| Variable | Description | Default |
|---|---|---|
| `MODEL_PATH` | Path to the Keras model | `models/my_classifier.h5` |
| `LOG_LEVEL` | Application logging level | `INFO` |

## 13. Future Enhancements

- Train the classifier using a real image dataset.
- Improve classification accuracy.
- Add model evaluation metrics.
- Deploy the API to a cloud platform.
- Add authentication and request-rate limiting.
- Add monitoring and logging dashboards.

## 14. Conclusion

This project demonstrates the development and containerized deployment of a machine learning prediction API. It integrates Keras model loading, image preprocessing, FastAPI endpoints, Docker, automated testing, and GitHub Actions into a single application.