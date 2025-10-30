# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
with open('pipeline_v1.bin', 'rb') as f_in: # very important to use 'rb' here, it means read-binary 
    dict_vectorizer, model = pickle.load(f_in)


from flask import Flask, request, jsonify
app = Flask(__name__)
def predict_single(client, dict_vectorizer, model):
  X = dict_vectorizer.transform([client])  ## apply the one-hot encoding feature to the customer data 
  y_pred = model.predict_proba(X)[:, 1]
  return y_pred[0]
@app.route('/predict', methods=['POST'])  ## in order to send the customer information we need to post its data.
def predict():
    client = request.get_json()  ## web services work best with json frame, So after the user post its data in json format we need to access the body of json.
    prediction = predict_single(client, dict_vectorizer, model)
    churn = prediction >= 0.5

    result = {
        'churn_probability': float(prediction), ## we need to cast numpy float type to python native float type
        'churn': bool(churn),  ## same as the line above, casting the value using bool method
    }
    return jsonify(result)  ## send back the data in json format to the user