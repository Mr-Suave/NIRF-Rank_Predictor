from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import os # Import os to handle file paths

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- Model Training ---
# Assuming TRAINING DATA.csv is in the same directory or accessible path
# Use os.path.join to handle file paths correctly
data_path = os.path.join(os.path.dirname(__file__), 'TRAINING DATA.csv')
try:
    df = pd.read_csv(data_path)

    # Optional: filter outliers as in your original code
    # Adjust threshold based on your data if needed
    initial_row_count = len(df)
    df = df[df['SS'].between(0, 30)] # Filter SS between 0 and 30 as an example bound
    if len(df) < initial_row_count:
        print(f"Filtered {initial_row_count - len(df)} rows based on SS <= 30.")


    # Features and target
    # Ensure these columns exist in your CSV
    required_cols = ['NT', 'NE', 'Np', 'SS']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in CSV. Need: {required_cols}, Found: {df.columns.tolist()}")

    X = df[['NT', 'NE', 'Np']]
    y = df['SS']

    # Choose Model: Polynomial Regression (degree 2)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)

    print("SS Score predictor model trained successfully.")
    model_trained = True

except FileNotFoundError:
    print(f"Error: TRAINING DATA.csv not found at {data_path}. Model not trained.")
    model_trained = False
except ValueError as ve:
     print(f"Error with data or columns: {ve}. Model not trained.")
     model_trained = False
except Exception as e:
    print(f"An unexpected error occurred during model training: {e}. Model not trained.")
    model_trained = False


@app.route('/predict', methods=['POST'])
def predict_ss():
    if not model_trained:
        return jsonify({"error": "Model not trained. Data file missing or has errors."}, 500) # Return 500 status

    try:
        data = request.get_json()

        # Validate input data exists
        if not all(key in data for key in ['nt', 'ne', 'np']):
            return jsonify({"error": "Missing one or more input parameters (nt, ne, np)."}, 400) # Return 400 status

        # Get and convert inputs to float
        try:
            value_nt = float(data['nt'])
            value_ne = float(data['ne'])
            value_np = float(data['np'])
        except ValueError:
             return jsonify({"error": "Invalid input. Please provide numeric values."}, 400)

        # Optional: Add server-side non-negative validation as well
        if value_nt < 0 or value_ne < 0 or value_np < 0:
             return jsonify({"error": "Input values cannot be negative."}, 400)


        # Create DataFrame for prediction
        x_pred = pd.DataFrame([[value_nt, value_ne, value_np]], columns=['NT', 'NE', 'Np'])

        # Predict SS Score
        y_pred = model.predict(x_pred)

        # Apply clipping and rounding as in your original code snippet
        # Clipping between 0 and 20, then rounding to nearest integer
        predicted_ss_score_rounded = round(y_pred[0], 2)


        return jsonify({"predicted_ss": predicted_ss_score_rounded}) # Access the single element and convert to a standard Python float

    except Exception as e:
        # Log the error for debugging on the server side
        print(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}, 500) # Generic error message for user

if __name__ == '__main__':
    # Use a specific port or the default (5000)
    # Render will likely use a PORT environment variable, so you might need:
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host="0.0.0.0", port=port)
    app.run(host="0.0.0.0", port=5000) # Example for local testing