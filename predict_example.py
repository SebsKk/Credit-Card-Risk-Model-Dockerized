import pandas as pd
import json
import numpy as np

data = pd.read_csv("C:\\Users\\kaczm\\Desktop\\default_of_credit_card_clients.csv")
data.replace([np.inf, -np.inf], np.nan, inplace=True)




input_data = data.to_dict(orient='records')



import requests

url = "http://localhost:5000/predict"
headers = {"Content-Type": "application/json"}
response = requests.post(url, json=input_data, headers=headers)

if response.status_code == 200:
    predictions = response.json()["predictions"]
    print("Predictions:", predictions)
else:
    print("Error:", response.status_code, response.text)