import pandas as pd
import numpy as np
import joblib

# ====== Load the saved model and scaler
model = joblib.load('./rf_churn_model.pkl')
scaler = joblib.load('./scaler.pkl')

# ====== Create a simulated, new customer (raw data before encoding)
# Likely churn example
new_customer = {
    'tenure': 3,              # Short tenure increases churn risk
    'MonthlyCharges': 100.0,   # Higher charges amplify dissatisfaction
    'TotalCharges': 300.0,     # tenure * MonthlyCharges
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',  # Expensive service, higher churn
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',       # No support increases risk
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',  # Flexible contract = high churn
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',  # Risky payment method
    'gender': 'Male',
    'SeniorCitizen': 1         # Price-sensitive senior
}
# Not likely churn example
# new_customer = {
#     'tenure': 12,
#     'MonthlyCharges': 70.0,
#     'TotalCharges': 840.0,
#     'Partner': 'No',
#     'Dependents': 'No',
#     'PhoneService': 'Yes',
#     'MultipleLines': 'No',
#     'InternetService': 'Fiber optic',
#     'OnlineSecurity': 'No',
#     'OnlineBackup': 'No',
#     'DeviceProtection': 'No',
#     'TechSupport': 'No',
#     'StreamingTV': 'No',
#     'StreamingMovies': 'No',
#     'Contract': 'Month-to-month',
#     'PaperlessBilling': 'Yes',
#     'PaymentMethod': 'Electronic check',
#     'gender': 'Male',
#     'SeniorCitizen': 0
# }

# ====== Convert to DataFrame
new_data = pd.DataFrame([new_customer])

# ====== Get the categorical columns from the original training data (excluding customerID, Churn)
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                   'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                   'PaperlessBilling', 'PaymentMethod']

# ====== Apply one-hot encoding (match training data)
new_data_encoded = pd.get_dummies(new_data, columns=categorical_cols, drop_first=True)

# ====== Ensure all columns match X_train (add missing columns with zeros)
X_train_cols = model.feature_names_in_  # Get feature names from the trained model
for col in X_train_cols:
    if col not in new_data_encoded.columns:
        new_data_encoded[col] = 0

# ====== Reorder columns to match X_train
new_data_encoded = new_data_encoded[X_train_cols]

# ====== Scale numeric features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
new_data_encoded[numeric_cols] = scaler.transform(new_data_encoded[numeric_cols])

# ====== Predict churn (0 = No, 1 = Yes)
prediction = model.predict(new_data_encoded)
probability = model.predict_proba(new_data_encoded)

# ====== Output results
print("New Customer Data:\n", new_data)
print("\nPrediction (0 = No Churn, 1 = Churn):", prediction[0])
print("Probability of No Churn:", probability[0][0])
print("Probability of Churn:", probability[0][1])

# ====== Business decision
if prediction[0] == 1:
    print("Action: This customer is at risk of churning. Consider offering a retention incentive!")
else:
    print("Action: This customer is likely to stay. No immediate action needed.")