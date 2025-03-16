from flask import Flask, request, jsonify
import xgboost as xgb
import numpy as np
import joblib

app = Flask(__name__)

# Load trained XGBoost model
model = joblib.load(r"C:\Users\ether\OneDrive\Documents\assignment_imago\xgb_model.pkl")

@app.route('/')
def home():
    return jsonify({"message": "XGBoost Model API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Ensure input format is correct
        if "features" not in data:
            return jsonify({"error": "Missing 'features' in request"}), 400
        
        # Convert features to numpy array
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        
        return jsonify({"prediction": prediction.tolist()})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
