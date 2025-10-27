from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('heart_disease_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        result = "Positive" if prediction == 1 else "Negative"
        return render_template('index.html', prediction_text=f'Heart Disease Prediction: {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == '__main__':
    app.run(debug=True)
