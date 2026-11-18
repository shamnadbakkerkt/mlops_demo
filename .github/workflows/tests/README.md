# Here is a summary of your project setup based on our steps so far, including how to structure and integrate this 
# Phase 2 Continuous Training & Drift Monitoring code cleanly.
Current Project Context
 * Core Task: Iris Dataset Classification (RandomForest) hosted via a FastAPI server.
 * Tracking & Packaging: MLflow for model registry/versioning (IrisRandomForest), Dockerized app deployment.
 * CI/CD: Automated build and VM deployment via GitHub Actions.
Project File Structure
Organize your workspace directory (Mlops-demo/) so that your original FastAPI app and the new Phase 2 scripts live cleanly side-by-side:
Mlops-demo/
├── .github/
│   └── workflows/
│       └── deploy.yml            # CI/CD pipeline script
├── mlruns/                        # MLflow tracking data directory
├── src/
│   ├── monitor_drift.py          # Evidently AI drift detection script
│   └── retrain_pipeline.py       # Automated MLflow retraining trigger
├── app.py                         # FastAPI serving script
├── train.py                       # Initial model training script
├── Dockerfile                     # Container build file
├── requirements.txt               # Updated dependency manifest
└── drift_report.html             # Generated visual drift dashboard

Step-by-Step Integration
1. Update requirements.txt
Append the required monitoring and training libraries to your environment configuration:
fastapi
uvicorn
scikit-learn
pandas
mlflow
pydantic
evidently
requests

Run installation locally:
pip install -r requirements.txt

2. Adapted Drift Script for Your Iris Setup (src/monitor_drift.py)
To align with your Iris dataset project rather than California Housing, use this adapted script:
import pandas as pd
from sklearn.datasets import load_iris
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def check_data_drift():
    # 1. Load reference data (Iris training set)
    iris = load_iris(as_frame=True)
    reference = iris.frame.sample(n=100, random_state=42)

    # 2. Simulate current production data with shifted distributions
    current = iris.frame.sample(n=100, random_state=99)
    current['sepal length (cm)'] = current['sepal length (cm)'] * 1.5  # Artificial drift

    # 3. Generate Evidently Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference, current_data=current)

    # 4. Save visual report and extract metric results
    drift_report.save_html("drift_report.html")
    report_dict = drift_report.as_dict()
    
    # Extract overall dataset drift flag
    dataset_drift = report_dict["metrics"][0]["result"]["dataset_drift"]
    print(f"Dataset Drift Detected: {dataset_drift}")
    
    return dataset_drift

if __name__ == "__main__":
    check_data_drift()

3. Adapted Continuous Retraining Trigger (src/retrain_pipeline.py)
This script executes model retraining and registers an updated version to your IrisRandomForest registry in MLflow whenever drift is triggered:
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from monitor_drift import check_data_drift

def automated_continuous_training():
    print("Checking production data for drift...")
    drift_detected = check_data_drift()

    if drift_detected:
        print("Drift threshold breached! Triggering automated retraining...")
        
        # Load dataset
        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42
        )

        mlflow.set_experiment("iris_classification")

        with mlflow.start_run(run_name="automated_retrain_run"):
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)

            # Log metadata and model artifact
            mlflow.log_param("trigger", "data_drift")
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(
                model, 
                "model", 
                registered_model_name="IrisRandomForest"
            )
            
            print(f"Retraining completed. New Model Accuracy: {acc:.4f}")
            print("Updated version registered to MLflow Model Registry.")
    else:
        print("No significant drift detected. Retraining skipped.")

if __name__ == "__main__":
    automated_continuous_training()

4. Execution Commands
Execute the pipeline check from your terminal:
python src/retrain_pipeline.py

Inspect the output HTML drift dashboard:
# Windows
start drift_report.html

# macOS/Linux
open drift_report.html

Inspect updated runs and model versions in the dashboard:
mlflow ui

Navigate to `


# Phase 3 Architecture Overview

Step 1: Implement Automated Retraining & Model Promotion

Update src/retrain_pipeline.py to handle full lifecycle management: checking drift, retraining on new data, comparing metric thresholds, logging to MLflow, and updating the Model Registry.

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from monitor_drift import check_data_drift

# Configure local tracking database
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-continuous-retraining")

MODEL_NAME = "IrisRandomForestModel"

def train_and_register_model():
    print("Initiating retraining process...")
    
    # 1. Load fresh dataset (incorporating new production records)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run() as run:
        # 2. Train model hyper-parameters
        n_estimators = 100
        max_depth = 5
        
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # 3. Evaluate candidate model
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # 4. Log params and metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        print(f"Candidate Model Trained. Accuracy: {acc:.4f}")

        # 5. Log and register the candidate model if it meets quality gate
        if acc >= 0.90:
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME
            )
            print(f"Model successfully logged and registered under URI: {model_info.model_uri}")
        else:
            print("Candidate model failed accuracy threshold (0.90). Promotion skipped.")

def run_pipeline():
    print("Checking production data for drift...")
    drift_detected = check_data_drift()

    if drift_detected:
        print("Data drift confirmed! Triggering retraining pipeline...")
        train_and_register_model()
    else:
        print("No significant drift detected. Model operating within normal bounds.")

if __name__ == "__main__":
    run_pipeline()


Step 2: Update Serving Endpoint (app.py)
Ensure your FastAPI serving layer dynamically loads the latest registered version from MLflow's Model Registry rather than reading a static serialized binary file:


import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Iris MLOps Level 2 API")

MODEL_NAME = "IrisRandomForestModel"

# Load latest model version from MLflow Registry
def load_latest_model():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model_uri = f"models:/{MODEL_NAME}/latest"
    try:
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Registry load warning ({e}); loading local fallback artifact...")
        return mlflow.sklearn.load_model("models/model.pkl")

model = load_latest_model()

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}


Step 3: CI/CD Workflow Trigger Setup
Update .github/workflows/deploy.yml so that whenever changes or automated retrains push updated model versions or core scripts to main, GitHub Actions triggers a build and redeploys the updated Docker container to your Linux VM:



name: Continuous Integration & Deployment

on:
  push:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Run Pipeline Validation & Tests
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pytest tests/

      - name: Build & Push Docker Image
        run: |
          docker build -t iris-mlops-api:latest .

      - name: Deploy Container to Remote Linux VM via SSH
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.VM_HOST }}
          username: ${{ secrets.VM_USER }}
          key: ${{ secrets.VM_SSH_KEY }}
          script: |
            docker pull iris-mlops-api:latest || true
            docker stop iris-api || true
            docker rm iris-api || true
            docker run -d -p 8000:8000 --name iris-api iris-mlops-api:latest

Phase 3 Execution Verification

Run your entry script to verify the automated pipeline execution end-to-end:
python src/retrain_pipeline.py

Expected Terminal Output Sequence:
 * Dataset Drift Detected: True
 * Initiating retraining process...
 * Candidate Model Trained. Accuracy: 1.0000
 * Model successfully logged and registered...
Would you like to draft unit/integration tests for pytest in step 3, or move directly to verifying deployment on the remote Linux VM?




Include tests folder in the root folder structure like below
-----------------

│     tests/
│   ├── test_app.py
      └──test_retrain.py


File contents 
---------------


test_app.py
----------------

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_check_endpoint():
    """Verify that the /health endpoint returns a 200 OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data


def test_predict_valid_input():
    """Verify inference pipeline with a valid Iris feature payload."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    
    # If the model is present, check 200 and schema
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1, 2]
    else:
        # Fallback handling if MLflow database model isn't populated yet
        assert response.status_code == 503
        assert response.json()["detail"] == "Model artifact unavailable."


def test_predict_invalid_schema():
    """Verify that Pydantic rejects payloads missing required fields."""
    invalid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        # Missing petal parameters
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity


test_retrain.py
------------------

import os
import sqlite3
import pytest
import mlflow
from src.monitor_drift import check_data_drift
from src.retrain_pipeline import train_and_register_model, run_pipeline, MODEL_NAME


def test_data_drift_detection():
    """Verify drift detection output interface."""
    drift_status = check_data_drift()
    assert isinstance(drift_status, bool)
    assert drift_status is True


def test_train_and_register_model_execution():
    """Test model training and verification of MLflow run logging."""
    # Ensure tracking URI matches pipeline setup
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Execute retraining step
    train_and_register_model()
    
    # Query MLflow tracking server for the experiment
    experiment = mlflow.get_experiment_by_name("iris-continuous-retraining")
    assert experiment is not None
    
    # Fetch runs associated with the experiment
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0
    
    # Verify logged metrics
    latest_run = runs.iloc[0]
    assert "metrics.accuracy" in latest_run
    assert latest_run["metrics.accuracy"] >= 0.90


def test_full_retrain_pipeline_trigger(capsys):
    """Verify end-to-end execution flow of the retrain pipeline."""
    run_pipeline()
    captured = capsys.readouterr()
    
    assert "Checking production data for drift..." in captured.out
    assert "Data drift confirmed! Triggering retraining pipeline..." in captured.out






    Here are the  terminal commands to test and verify the entire Phase 3 continuous training and deployment workflow end-to-end.

Step 1: Run the Continuous Retraining Pipeline

Execute your retraining script locally to verify drift detection, model training, and registration into MLflow:

python src/retrain_pipeline.py

Expected output: Checks drift, detects True, trains a new RandomForest model, and registers it to MLflow.

Step 2: Verify MLflow Experiments & Model Registry.

Launch the local MLflow dashboard to view logged parameters, metrics, and registered model versions:

mlflow ui --backend-store-uri sqlite:///mlflow.db

Open your browser and navigate to:
http://127.0.0.1:5000

Step 3: Test FastAPI Local Serving Endpoint
Start your FastAPI app locally to confirm it dynamically loads the new registered model version from MLflow:

uvicorn app:app --reload --port 8000

In a second terminal window, send a test payload using curl:

macOS / Linux Terminal:

curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

Windows Command Prompt (CMD):

curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}"

Expected JSON response: {"prediction": 0}


Step 4: Validate Containerization Locally
Test building and running the Docker container locally before triggering your GitHub Actions deployment:

# 1. Build Docker Image
docker build -t iris-mlops-api:latest .

# 2. Run Docker Container
docker run -d -p 8000:8000 --name iris-api-test iris-mlops-api:latest

# 3. Test Running Container

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method POST -ContentType "application/json" -Body '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

# 4. Clean Up Container
docker stop iris-api-test && docker rm iris-api-test

--------------TEST COMPLETED------------