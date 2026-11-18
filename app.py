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