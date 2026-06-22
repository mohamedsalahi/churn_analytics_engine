from http.server import BaseHTTPRequestHandler
import json
import math
import os

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except Exception:
                data = {}
        else:
            data = {}

        tenure = float(data.get('tenure', 12))
        monthly_charges = float(data.get('monthlyCharges', 60))
        total_charges = float(data.get('totalCharges', 720))
        contract_type = data.get('contractType', 'Month-to-month')
        
        # Load the pre-trained ML model weights (Logistic Regression)
        # This completely removes the need for a 250MB scikit-learn dependency in AWS Lambda
        weights_path = os.path.join(os.path.dirname(__file__), '..', 'model_weights.json')
        with open(weights_path, 'r') as f:
            model = json.load(f)
            
        weights = model['features']
        z = model['intercept']
        
        # Calculate the dot product of features and weights
        z += tenure * weights['tenure']
        z += monthly_charges * weights['monthlyCharges']
        z += total_charges * weights['totalCharges']
        
        # One-hot encoding resolution for categorical feature
        if contract_type == 'Month-to-month':
            z += weights['contract_Month-to-month']
        elif contract_type == 'One year':
            z += weights['contract_One year']
        elif contract_type == 'Two year':
            z += weights['contract_Two year']
            
        # Sigmoid activation function for Logistic Regression Probability
        try:
            probability = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            probability = 0.0 if z < 0 else 1.0
        
        risk_level = 'LOW'
        if probability > 0.65:
            risk_level = 'HIGH'
        elif probability > 0.35:
            risk_level = 'MEDIUM'
            
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            'churnProbability': probability,
            'riskLevel': risk_level
        }
        self.wfile.write(json.dumps(response).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
