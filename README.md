# Housing Regression MLE

## Overview
End-to-end **housing price regression project** built from notebooks into modular Python pipelines.  
Covers **feature engineering, training/tuning, inference, and batch predictions** with tests and reproducible outputs.

## Goal
Build a full **ML regression pipeline** (XGBoost) that is leakage-safe, experiment-tracked, and ready for deployment.

## Architecture
`Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve`

## Approach & Tools
- **Data Splits**: time-aware (train <2020, eval 2020–21, holdout ≥2022).  
- **Preprocessing**: city normalization, deduplication, outlier removal.  
- **Feature Engineering**: date parts, frequency encoding (`zipcode`), target encoding (`city_full`).  
- **Modeling**: XGBoost + Optuna tuning, tracked with MLflow.  
- **Inference**: consistent preprocessing + encoding applied to new data.  
- **Batch Logic**: monthly runs on holdout/future data (cron/EventBridge for scheduling).  
- **Testing**: PyTest for features/training; smoke test for inference.  
- **Future**: FastAPI for serving, Docker for packaging, GitHub Actions + AWS ECS for CI/CD, Streamlit UI.

## Fixes
- **Data leakage prevention**: Split data by date not randomly, fit encoders on train only, drop leakage columns, ensure strict schema alignment.
