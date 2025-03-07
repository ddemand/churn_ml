import requests
import json

url = 'http://localhost:5000/predict'
data = {
    'tenure': 3,
    'MonthlyCharges': 100.0,
    'TotalCharges': 300.0,
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'gender': 'Male',
    'SeniorCitizen': 1
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))