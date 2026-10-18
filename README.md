# Plant Disease Classification — MLOps Project

## 1. Project Overview

Plant Disease Classification is a machine learning web application that predicts a possible plant condition based on four observed symptoms: leaf spots, leaf yellowing, leaf curling, and leaf wilting.

The project demonstrates an end-to-end MLOps workflow, including model training, web application development, automated testing, Docker containerization, source code management, and deployment on AWS EC2.

**Note:** This is an educational demonstration trained on a small synthetic dataset. Its predictions are not validated for real-world plant disease diagnosis.

## 2. Objectives

- Develop a machine learning classification model.
- Create a web interface using Flask.
- Test the application using pytest.
- Package the application using Docker.
- Store and manage source code using Git and GitHub.
- Deploy the containerized application on AWS EC2.

## 3. Technologies Used

- **Programming language:** Python 3.11
- **Machine learning:** Scikit-learn, Decision Tree Classifier
- **Data processing:** Pandas, NumPy
- **Web framework:** Flask
- **Model serialization:** Joblib
- **Testing:** Pytest
- **Containerization:** Docker
- **Cloud deployment:** AWS EC2 (Amazon Linux 2023)
- **Version control:** Git and GitHub
- **Web server:** Gunicorn

## 4. Project Structure

```text
PlantDisease-MLops/
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── test_app.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## 5. How It Works

1. The training script creates a small synthetic dataset and trains a Decision Tree classification model.
2. The trained model is saved as `model.pkl`.
3. The Flask application loads the model and displays a symptom input form.
4. The user selects whether each symptom is present or absent.
5. The model predicts a class based on the selected symptoms.
6. Pytest checks the home page, health endpoint, and prediction endpoint.
7. Docker packages the application and its dependencies into a container.
8. The Docker container runs on AWS EC2 and serves the web application over HTTP.

## 6. Run Locally

### Prerequisites

Install Python 3.11 and Docker if you want to run the containerized version.

### Install dependencies

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Train the model

```bash
python train_model.py
```

### Start the Flask application

```bash
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

## 7. Run Tests

Execute the automated tests with:

```bash
python -m pytest test_app.py -v
```

The tests cover the home page, health endpoint, and prediction endpoint.

## 8. Docker Deployment

Build the Docker image:

```bash
docker build -t plant-disease-ml .
```

Run the container locally:

```bash
docker run -d --name plant-disease-app -p 5002:5000 plant-disease-ml
```

Check running containers:

```bash
docker ps
```

Check application health:

```bash
curl http://10.44.47.152:5000/health
```

## 9. AWS EC2 Deployment

The application was deployed on an Amazon Linux 2023 EC2 instance.

The deployment process included:

1. Launching an EC2 instance.
2. Installing and starting Docker.
3. Cloning the project from GitHub.
4. Building the Docker image on EC2.
5. Running the container with host port 80 mapped to container port 5000.
6. Verifying the application using its health endpoint.
7. Accessing the application through the EC2 public IPv4 address.

The application health endpoint is available at `/health`.

## 10. Health Check

The `/health` endpoint reports the application's status and whether the machine learning model is loaded.

Example response:

```json
{
  "model_loaded": true,
  "status": "healthy"
}
```

## 11. Limitations

- The model uses a small synthetic dataset rather than a representative collection of real plant images or field observations.
- The classifier uses four binary symptom features.
- Predictions are for educational demonstration only and should not be treated as reliable agricultural advice.
- Real-world use would require representative data, model evaluation, validation, and monitoring.

## 12. Future Enhancements

- Train and evaluate the model using a larger, real-world plant disease dataset.
- Add model performance and data drift monitoring.
- Implement model versioning and experiment tracking.
- Add a CI/CD pipeline for automated testing and deployment.
- Improve the user interface and prediction explanations.

## 13. Repository

GitHub: https://github.com/vishnuvardhan999333/PlantDisease-MLops

## 14. Author

Developed as an academic MLOps project demonstrating machine learning deployment using Flask, Docker, GitHub, and AWS EC2.
