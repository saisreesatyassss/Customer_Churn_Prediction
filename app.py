import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load your pre-trained model
with open('model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route("/")
def loadPage():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():
    
    # Define the input variables and collect data from the form
    input_features = [
        'Age', 'Gender', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB',
        'Location_Chicago', 'Location_Houston', 'Location_Miami', 'Location_New York',
        'Subscription_Length_Bin', 'Monthly_Bill_Bin'
    ]
    
    input_data = []
    
    for feature in input_features:
        input_data.append(float(request.form[feature]))
    
    # Predict using the loaded model
    prediction = loaded_model.predict([input_data])[0]
    probability = loaded_model.predict_proba([input_data])[0][1]
    
    result = "This customer is likely to churn." if prediction == 1 else "This customer is likely to continue."
    confidence = f"Confidence: {probability * 100:.2f}%"
    
    return render_template('home.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run()
