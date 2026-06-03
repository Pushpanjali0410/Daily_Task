from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model once
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")


class WineInput(BaseModel):

    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315: float
    proline: float


@app.get("/")
def home():
    return {"message": "Wine Prediction API"}


@app.post("/predict")
def predict(data: WineInput):

    features = np.array([[
        data.alcohol,
        data.malic_acid,
        data.ash,
        data.alcalinity_of_ash,
        data.magnesium,
        data.total_phenols,
        data.flavanoids,
        data.nonflavanoid_phenols,
        data.proanthocyanins,
        data.color_intensity,
        data.hue,
        data.od280_od315,
        data.proline
    ]])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)

    return {
        "predicted_class": int(prediction[0])
    }