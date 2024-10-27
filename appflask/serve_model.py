from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will allow all domains to access your API

# Load the Random Forest model
model_path = '../notebook/models/Fraud_Random_Forest.pkl'  # Update with your model's actual path
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print(f"Model file not found at {model_path}")

@app.route('/')
def home():
    return "Welcome to the Fraud Detection API!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json()  # Expecting JSON input
    
    try:
        # Ensure data is in the correct format for DataFrame
        if isinstance(data, dict):  # If single record, wrap in list
            data = [data]
        
        # Convert JSON input to a DataFrame
        input_df = pd.DataFrame(data)
        
        # Perform prediction
        predictions = model.predict(input_df)
        
        # Generate response message based on the prediction
        if predictions[0] == 0:
            result_message = "The User is Not Fraud Suspected"
        elif predictions[0] == 1:
            result_message = "The User is Fraud Suspected"
        else:
            result_message = "Unknown prediction result"

        response = {"message": result_message}
        
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
