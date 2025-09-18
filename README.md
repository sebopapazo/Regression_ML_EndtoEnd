# Housing Regression MLE

## Overview
End-to-end **housing price regression project** built from research notebooks into modular, production-ready pipelines.  
The system covers **feature engineering, training/tuning, inference, batch predictions, API serving, and a Streamlit dashboard** — all deployed on AWS with CI/CD.

## Goal
Build a full **ML regression pipeline** (XGBoost) that is:
- Leakage-safe
- Experiment-tracked
- Deployed with real-time and batch inference
- Backed by reproducible code and cloud infrastructure

## Architecture
Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve


### Components
- **Model training**: XGBoost regression tuned with Optuna, tracked in MLflow.
- **Inference**: FastAPI service for predictions, with strict schema alignment.
- **Batch logic**: monthly predictions triggered (via cron/EventBridge).
- **Serving**: Application Load Balancer (ALB) routing `/predict` → FastAPI, `/dashboard` → Streamlit.
- **Dashboard**: Streamlit app for visualizing holdout predictions vs actuals.
- **Storage**: Datasets + model artifacts uploaded and fetched from **Amazon S3**.
- **CI/CD**: GitHub Actions → build Docker images → push to ECR → redeploy ECS services.
- **Deployment**: AWS ECS Fargate cluster with ALB routing.

## Approach & Tools
- **Data Splits**: time-aware (train 2012-2019, eval 2020–21, holdout ≥2022).  
- **Preprocessing**: city normalization, deduplication, outlier removal.  
- **Feature Engineering**:  
  - Date parts (`year`, `month`)  
  - Frequency encoding (`zipcode`)  
  - Target encoding (`city_full`)  
- **Modeling**: XGBoost with Optuna tuning, logged in MLflow.  
- **Inference**: preprocessing + encodings applied consistently to API input.  
- **Batch Predictions**: automated monthly holdout/future runs with saved CSV outputs.  
- **Dashboard**: Streamlit explorer for interactive filtering & error metrics.  
- **Deployment**: FastAPI + Streamlit Dockerized, deployed via ECS with CI/CD.  
- **Data Access**: all key datasets and trained models stored/retrieved from S3.  
- **Testing**: PyTest for preprocessing & training, smoke tests for inference.

## Bottlenecks Solved
- **Data leakage prevention**:  
  - Split by date instead of random  
  - Fit encoders on train only  
  - Dropped leakage-prone columns  
  - Enforced schema alignment at inference  
- **Routing**: Configured ALB rules to direct `/predict` → API target group, `/dashboard` → Streamlit target group.  
- **502 Gateway Errors**: Solved by aligning ALB listener rules and Streamlit’s `--server.baseUrlPath=/dashboard`.  
- **Missing data in ECS**: Moved medium-sized CSVs and model files to S3, programmatically fetched at container startup.  
- **CI/CD mismatch**: Fixed by updating GitHub Actions to push images with `latest` tag to **ECR** and force ECS service redeployment.  

## Future Improvements
- Add **monitoring** with Prometheus/Grafana + Evidently for drift detection.  
- Extend CI/CD with staging/production environments.  
- Add authentication to API + dashboard.  
- Scale to larger datasets via Spark + AWS Glue.  
