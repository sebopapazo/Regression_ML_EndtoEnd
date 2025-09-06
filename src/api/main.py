from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import pandas as pd

# Import inference and batch runner
from src.inference_pipeline.inference import predict
from src.batch.run_monthly import run_monthly_predictions

# ----------------------------
# Create FastAPI app
# ----------------------------
app = FastAPI(title="Housing Regression API")


# ----------------------------
# Root endpoint (sanity check)
# Purpose: quick test if API itself is alive
# ----------------------------
@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀"}


# ----------------------------
# Health endpoint
# Purpose: confirm if trained model exists on disk
# ----------------------------
@app.get("/health")
def health():
    model_path = Path("models/xgb_best_model.pkl")
    if model_path.exists():
        return {"status": "healthy", "model": str(model_path)}
    return {"status": "unhealthy", "error": "Model file not found"}


# ----------------------------
# Schema for row-level prediction
# Purpose: enforce structure when hitting /predict
# ----------------------------
class HousingData(BaseModel):
    date: str
    city_full: str
    zipcode: int
    median_list_price: float
    price: float


# ----------------------------
# Predict endpoint
# Purpose: run inference on one row of input
# ----------------------------
@app.post("/predict")
def predict_price(data: HousingData):
    df = pd.DataFrame([data.dict()])  # convert request to DataFrame
    preds_df = predict(df)
    return {"predicted_price": float(preds_df["predicted_price"].iloc[0])}


# ----------------------------
# Run Batch endpoint
# Purpose: trigger full batch predictions on holdout data
# Uses run_monthly_predictions() to generate monthly CSVs
# ----------------------------
@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": len(preds),
        "output_dir": "data/predictions/"
    }


# ----------------------------
# Latest Predictions endpoint
# Purpose: quickly inspect the most recent predictions file
# Helpful for debugging & validation
# ----------------------------
@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found."}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": len(df),
        "preview": df.head(limit).to_dict(orient="records")
    }
