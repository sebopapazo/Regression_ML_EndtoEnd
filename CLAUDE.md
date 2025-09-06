# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Housing Regression MLE is an end-to-end machine learning pipeline for predicting housing prices using XGBoost. The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization, and comprehensive testing.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Batch → Serve`

### Core Modules

- **`src/feature_pipeline/`**: Data loading, preprocessing, and feature engineering
  - `load.py`: Time-aware data splitting (train <2020, eval 2020-21, holdout ≥2022)
  - `preprocess.py`: City normalization, deduplication, outlier removal  
  - `feature_engineering.py`: Date features, frequency encoding (zipcode), target encoding (city_full)

- **`src/training_pipeline/`**: Model training and hyperparameter optimization
  - `train.py`: Baseline XGBoost training with configurable parameters
  - `tune.py`: Optuna-based hyperparameter tuning with MLflow integration
  - `eval.py`: Model evaluation and metrics calculation

- **`src/inference_pipeline/`**: Production inference
  - `inference.py`: Applies same preprocessing/encoding transformations using saved encoders

- **`src/batch/`**: Batch prediction processing
  - `run_monthly.py`: Generates monthly predictions on holdout data

- **`src/api/`**: FastAPI web service
  - `main.py`: REST API with endpoints for single predictions, batch runs, and health checks

### Data Leakage Prevention

The project implements strict data leakage prevention:
- Time-based splits (not random)
- Encoders fitted only on training data
- Leakage-prone columns dropped before training
- Schema alignment enforced between train/eval/inference

## Common Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync
```

### Testing
```bash
# Run all tests
pytest

# Run specific test modules  
pytest tests/test_features.py
pytest tests/test_training.py
pytest tests/test_inference.py

# Run with verbose output
pytest -v
```

### Data Pipeline
```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py

# 2. Preprocess splits
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
```

### Training Pipeline
```bash
# Train baseline model
python src/training_pipeline/train.py

# Hyperparameter tuning with MLflow
python src/training_pipeline/tune.py

# Model evaluation
python src/training_pipeline/eval.py
```

### Inference
```bash
# Single inference
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv

# Batch monthly predictions
python src/batch/run_monthly.py
```

### API Service
```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Using uv (recommended)
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build container
docker build -t housing-regression .

# Run container
docker run -p 8000:8000 housing-regression
```

### MLflow Tracking
```bash
# Start MLflow UI (view experiments)
mlflow ui
```

## Key Design Patterns

### Pipeline Modularity
Each pipeline component can be run independently with consistent interfaces. All modules accept configurable input/output paths for testing isolation.

### Encoder Persistence  
Frequency and target encoders are saved as pickle files during training and loaded during inference to ensure consistent transformations.

### Configuration Management
Model parameters, file paths, and pipeline settings use sensible defaults but can be overridden through function parameters or environment variables.

### Testing Strategy
- Unit tests for individual pipeline components
- Integration tests for end-to-end pipeline flows  
- Smoke tests for inference pipeline
- All tests use temporary directories to avoid touching production data

## File Structure Notes

- **`data/`**: Raw, processed, and prediction data (time-structured)
- **`models/`**: Trained models and encoders (pkl files)
- **`mlruns/`**: MLflow experiment tracking data
- **`configs/`**: YAML configuration files
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`tests/`**: Comprehensive test suite with sample data