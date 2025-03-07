import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('./rf_churn_model.pkl')
scaler = joblib.load('./scaler.pkl')

# Define categorical columns (same as in predict.py)
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']

# Define numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Convert to DataFrame
        new_data = pd.DataFrame([data])

        # Apply one-hot encoding
        new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

        # Ensure all columns match the training data (add missing columns with zeros)
        X_train_cols = model.feature_names_in_
        for col in X_train_cols:
            if col not in new_data_encoded.columns:
                new_data_encoded[col] = 0

        # Reorder columns to match training data
        new_data_encoded = new_data_encoded[X_train_cols]

        # Scale numeric features
        new_data_encoded[numeric_cols] = scaler.transform(new_data_encoded[numeric_cols])

        # Predict churn and probabilities
        prediction = model.predict(new_data_encoded)[0]
        probability = model.predict_proba(new_data_encoded)[0]

        # Prepare response
        response = {
            'prediction': int(prediction),  # 0 = No Churn, 1 = Churn
            'probability_no_churn': float(probability[0]),
            'probability_churn': float(probability[1]),
            'action': ('This customer is at risk of churning. Consider offering a retention incentive!'
                       if prediction == 1 else 'This customer is likely to stay. No immediate action needed.')
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)