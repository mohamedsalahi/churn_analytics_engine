# Predictive Customer Churn Analytics Engine

Serverless-ready ML inference engine that predicts customer churn probability using a lightweight Logistic Regression model. Designed to run in AWS Lambda with zero heavy ML dependencies, the model is serialized to JSON weights (~300 bytes).

## Tech Stack

- **Python** | Core inference and training logic
- **scikit-learn** | Model training (offline)
- **Pandas** | Data preprocessing and feature engineering
- **NumPy** | Numerical operations

## Key Features

- Real-time churn probability scoring with sub-millisecond inference
- Three-tier risk classification (HIGH / MEDIUM / LOW)
- Serverless-ready: JSON-serialized model weights eliminate scikit-learn dependency at runtime
- Zero external inference dependencies — pure Python math

## How to Run Locally

```bash
pip install -r requirements.txt
python src/inference.py
```

## How to Train the Model

```bash
python scripts/train_model.py
```

## Serverless Deployment

The `api/churn.py` handler is a standalone HTTP server that loads the pre-trained model weights from `model_weights.json` and serves predictions via POST requests. It requires **zero** ML libraries at runtime, no scikit-learn, no Pandas, making it ideal for AWS Lambda and other serverless platforms.
