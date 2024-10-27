# app.py
import pandas as pd
from flask import Flask, jsonify
from datetime import datetime

app = Flask(__name__)

# Load the fraud data from a CSV file
DATA_PATH = '../Data/ForApp.csv'
data = pd.read_csv(DATA_PATH)

@app.route('/api/summary', methods=['GET'])
def summary():
    total_transactions = len(data)
    total_fraud_cases = data[data['class'] == 1].shape[0]
    fraud_percentage = (total_fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0

    return jsonify({
        'total_transactions': total_transactions,
        'total_fraud_cases': total_fraud_cases,
        'fraud_percentage': fraud_percentage,
    })

@app.route('/api/fraud_trends', methods=['GET'])
def fraud_trends():
    data['purchase_time'] = pd.to_datetime(data['purchase_time'])
    trends = data[data['class'] == 1].groupby(data['purchase_time'].dt.to_period('M')).size().reset_index(name='fraud_cases')
    trends['purchase_time'] = trends['purchase_time'].dt.to_timestamp()

    return trends.to_json(date_format='iso', orient='records')
@app.route('/api/fraud_by_country', methods=['GET'])
def fraud_by_country():
    country_fraud = data[data['class'] == 1].groupby('country').size().reset_index(name='fraud_cases')
    return country_fraud.to_json(orient='records')
@app.route('/api/fraud_by_device_browser', methods=['GET'])
def fraud_by_device_browser():
    device_browser_fraud = data[data['class'] == 1].groupby(['device_id', 'browser']).size().reset_index(name='fraud_cases')
    top_devices = device_browser_fraud.nlargest(10, 'fraud_cases')  # Get top 5 devices
    return top_devices.to_json(orient='records') 

if __name__ == '__main__':
    app.run(debug=True)