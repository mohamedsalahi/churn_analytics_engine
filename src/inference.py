import json
import math
import os

def load_model(weights_path=None):
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'model_weights.json')
    with open(weights_path, 'r') as f:
        model = json.load(f)
    return model['features'], model['intercept']

def predict_churn(tenure, monthly_charges, total_charges, contract_type, weights, intercept):
    z = intercept
    z += tenure * weights['tenure']
    z += monthly_charges * weights['monthlyCharges']
    z += total_charges * weights['totalCharges']

    if contract_type == 'Month-to-month':
        z += weights['contract_Month-to-month']
    elif contract_type == 'One year':
        z += weights['contract_One year']
    elif contract_type == 'Two year':
        z += weights['contract_Two year']

    try:
        probability = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        probability = 0.0 if z < 0 else 1.0

    risk_level = 'LOW'
    if probability > 0.65:
        risk_level = 'HIGH'
    elif probability > 0.35:
        risk_level = 'MEDIUM'

    return probability, risk_level


if __name__ == '__main__':
    weights, intercept = load_model()
    prob, risk = predict_churn(
        tenure=12,
        monthly_charges=60,
        total_charges=720,
        contract_type='Month-to-month',
        weights=weights,
        intercept=intercept,
    )
    print(f"Churn Probability: {prob:.4f}")
    print(f"Risk Level: {risk}")
