"""FastAPI service for the trained encroachment severity classifier."""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "encroachment_svm.joblib"

app = FastAPI(
    title="Road Reserve Encroachment Severity API",
    description="Predicts the rule-derived severity class from measured building/encroachment attributes.",
    version="0.1.0",
)
model_bundle = None


class EncroachmentRequest(BaseModel):
    distance_to_road: float = Field(ge=0, description="Distance from building centroid to road centreline, metres")
    encroachment_depth: float = Field(ge=0, le=50, description="Measured penetration depth used by the project's severity rule, metres")
    encroachment_area: float = Field(ge=0, description="Building area intersecting the ROW buffer, square metres")
    building_area: float = Field(gt=0, description="Total building footprint area, square metres")
    encroachment_percentage: float = Field(ge=0, le=100)
    building_type: str = Field(default="unknown")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class PredictionResponse(BaseModel):
    predicted_severity: Literal["Low", "Medium", "High", "Critical"]
    class_probabilities: dict[str, float]
    model_note: str


@app.on_event("startup")
def load_model():
    global model_bundle
    if MODEL_PATH.exists():
        model_bundle = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_bundle is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: EncroachmentRequest):
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Run the training script before starting the API."
        )
    row = pd.DataFrame([payload.model_dump()])
    pipeline = model_bundle["pipeline"]
    prediction = str(pipeline.predict(row)[0])
    probabilities = pipeline.predict_proba(row)[0]
    classes = pipeline.named_steps["classifier"].classes_
    return PredictionResponse(
        predicted_severity=prediction,
        class_probabilities={
            str(label): float(prob) for label, prob in zip(classes, probabilities)
        },
        model_note=(
            "This is an automated classification of measured encroachment attributes. "
            "Because the target severity is defined using encroachment depth, this output "
            "does not constitute a forecast of future encroachment."
        ),
    )
