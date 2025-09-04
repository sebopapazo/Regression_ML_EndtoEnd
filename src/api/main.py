from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
from src.inference_pipeline.inference import predict

app = FastAPI(title="Housing Regression API")

# ----------------------------
# Root endpoint (health check)
# ----------------------------
@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀"}

# ----------------------------
# Health endpoint
# ----------------------------
@app.get("/health")
def health():
    """Quick check if model + pipeline files exist."""
    model_path = Path("models/xgb_best_model.pkl")
    if model_path.exists():
        return {"status": "healthy", "model": str(model_path)}
    return {"status": "unhealthy", "error": "Model file not found"}

# ----------------------------
# Request schema for /predict
# ----------------------------
class HousingData(BaseModel):
    date: str
    city_full: str
    zipcode: int
    median_list_price: float
    price: float

# ----------------------------
# Predict endpoint
# ----------------------------
@app.post("/predict")
def predict_price(data: HousingData):
    """Run inference on one row of housing data."""
    df = pd.DataFrame([data.dict()])  # convert to DataFrame
    preds_df = predict(df)
    return {"predicted_price": float(preds_df["predicted_price"].iloc[0])}
