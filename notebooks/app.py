from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

## Import any other packages that are needed

app = Flask(__name__)

# 1. Load your model here
model = joblib.load('model.joblib')

# 2. Define a prediction function
def compute_prediction(data):
    df = pd.DataFrame(data).T
    return model.predict(df).tolist()

# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    <p>
        <strong>Example</strong>:<br />
        <code>curl http://18.236.211.135:5000/predict
            -d '{"data":[1,2,3,4,53,11,22,37,41,53,11,24,31,44,53,11,22,35,42,53,12,23,31,42,53]}'
            -H "Content-Type: application/json"
        </code>
    </p>
    """

# 4. define a new route which will accept POST requests and return model predictions
@app.route('/predict', methods=['POST'])
def rainfall_prediction():
    content = request.json  # this extracts the JSON content we sent
    prediction = compute_prediction(content['data'])
    results =  {
        "Input": content['data'],
        "Output": f"Ensemble model prediction: {prediction}!"
    }   
    return jsonify(results)


