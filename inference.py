import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np

# Load the trained model
model = joblib.load('model/best_model.joblib')

# Create a Flask app
app = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json

    # Convert the input data to a DataFrame
    input_data = pd.DataFrame(data)

    # Preprocessing needed for the model to run
    preprocessed_data = preprocess_data(input_data)

    # Prediction
    predictions = model.predict(preprocessed_data)

    # Convert the predictions to a list
    predictions_list = predictions.tolist()

    # Return JSON response
    return jsonify({'predictions': predictions_list})

# Preprocessing function
def preprocess_data(data):
    preprocessed_data = data.copy()

    # Create new features
    preprocessed_data['Payment_Consistency'] = preprocessed_data[['X6', 'X7', 'X8', 'X9', 'X10', 'X11']].std(axis=1)
    preprocessed_data['Total_Bill_Amount'] = preprocessed_data[['X12', 'X13', 'X14', 'X15', 'X16', 'X17']].sum(axis=1)
    preprocessed_data['Credit_Utilization_Ratio'] = preprocessed_data['Total_Bill_Amount'] / preprocessed_data['X1']
    preprocessed_data['Total_Payment_Amount'] = preprocessed_data[['X18', 'X19', 'X20', 'X21', 'X22', 'X23']].sum(axis=1)
    median_payment_amount = preprocessed_data['Total_Payment_Amount'].median()
    preprocessed_data['Debt_to_Income_Ratio'] = preprocessed_data['X1'] / (preprocessed_data['Total_Payment_Amount'] + 1)
    preprocessed_data['Total_Delays'] = (preprocessed_data[['X6', 'X7', 'X8', 'X9', 'X10', 'X11']] > 0).sum(axis=1)
    preprocessed_data['Total_Payments_Done'] = (preprocessed_data[['X18', 'X19', 'X20', 'X21', 'X22', 'X23']] > 0).sum(axis=1)
    preprocessed_data['On_Time_Payments'] = (preprocessed_data[['X6', 'X7', 'X8', 'X9', 'X10', 'X11']] == -1).sum(axis=1)
    preprocessed_data['Payment_amount_to_median'] = preprocessed_data['Total_Payment_Amount'] / median_payment_amount

    # Drop unnecessary columns
    preprocessed_data.drop(columns=['Total_Payment_Amount', 'Total_Bill_Amount'], inplace=True)

    preprocessed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    preprocessed_data.fillna(preprocessed_data.mean(), inplace=True)

    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = ['Payment_Consistency', 'Credit_Utilization_Ratio', 'Debt_to_Income_Ratio', 'Total_Delays',
                       'Total_Payments_Done', 'On_Time_Payments', 'Payment_amount_to_median']
    preprocessed_data[scaled_features] = scaler.fit_transform(preprocessed_data[scaled_features])

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(preprocessed_data[scaled_features])
    preprocessed_data['Cluster_kmeans'] = kmeans.labels_

    # Normalize specific columns
    normalize_columns = ['X1', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23',
                         'Debt_to_Income_Ratio', 'Payment_amount_to_median']
    preprocessed_data[normalize_columns] = scaler.fit_transform(preprocessed_data[normalize_columns])

    return preprocessed_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
