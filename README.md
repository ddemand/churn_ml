# churn_ml
An example machine learning model to predict whether a customer will leave a service (churn) based on historical data.


## Hyperparameter Tuning with GridSearchCV:
- Hyperparameters: These are settings in the Random Forest model (e.g., number of trees, max depth) that you define before training, unlike parameters (e.g., feature weights) learned during training.
- GridSearchCV: A scikit-learn tool that tests multiple combinations of hyperparameters using cross-validation to find the best-performing set.

  - n_estimators: Number of trees in the forest.
  - max_depth: Maximum depth of each tree (controls overfitting).
  - min_samples_split: Minimum samples required to split a node.
  - min_samples_leaf: Minimum samples required at a leaf node.

# How This Works

    Load Model and Scaler: Retrieve the saved Random Forest model and scaler.
    New Data: Define a new customer as a dictionary (simulating real input).
    Preprocessing:
        One-hot encode categorical variables, matching the training data’s structure.
        Add missing columns (e.g., if InternetService_DSL isn’t in the new data, set it to 0).
        Scale numeric features using the same scaler.
    Prediction:
        predict() gives a binary outcome (0 or 1).
        predict_proba() gives probabilities for both classes (useful for decision-making).
    Actionable Output: Suggest a business response based on the prediction.

# Example output
[1 rows x 19 columns]

Prediction (0 = No Churn, 1 = Churn): 1
Probability of No Churn: 0.48494599220681905
Probability of Churn: 0.515054007793181
Action: This customer is at risk of churning. Consider offering a retention incentive!

# To run the app in command line as a manual test
1. Navigate to the local repository in a terminal
2. Activate the virtual environment using "source venv/bin/activate"
3. python app.py 
 (venv) daviddemand@Rachels-MBP churn_ml % python app.py
 * Serving Flask app 'app'
 * Debug mode: on
 WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://192.168.1.149:5000
 Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 684-238-830
4. python test_request.py
 {
  "action": "This customer is at risk of churning. Consider offering a retention incentive!",
  "prediction": 1,
  "probability_churn": 0.515054007793181,
  "probability_no_churn": 0.48494599220681905
 } 


