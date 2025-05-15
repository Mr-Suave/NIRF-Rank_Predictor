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
# Assuming MEDIAN DATASET.csv is in the same directory or accessible path
data_path = os.path.join(os.path.dirname(__file__), 'MEDIAN DATASET.csv')
try:
    df = pd.read_csv(data_path)

    # Features and target from the GSM part of your code
    # Ensure these columns exist in your CSV
    required_cols = ['MEDIAN_SALARY', 'SCORE']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns in CSV. Need: {required_cols}, Found: {df.columns.tolist()}")

    X = df[['MEDIAN_SALARY']]
    y = df['SCORE']

    # Choose Model: Polynomial Regression (degree 2)
    # The user's code used degree=2 for the polynomial model
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)

    print("GSM predictor model trained successfully using Polynomial Regression.")
    model_trained = True

except FileNotFoundError:
    print(f"Error: MEDIAN DATASET.csv not found at {data_path}. Model not trained.")
    model_trained = False
except ValueError as ve:
     print(f"Error with data or columns: {ve}. Model not trained.")
     model_trained = False
except Exception as e:
    print(f"An unexpected error occurred during model training: {e}. Model not trained.")
    model_trained = False


@app.route('/predict', methods=['POST'])
def predict_gsm():
    if not model_trained:
        return jsonify({"error": "Model not trained. Data file missing or has errors."}, 500) # Return 500 status

    try:
        data = request.get_json()

        # Validate input data exists
        if 'median_salary' not in data:
            return jsonify({"error": "Missing input parameter 'median_salary'."}, 400) # Return 400 status

        # Get and convert input to float
        try:
            value_median_salary = float(data['median_salary'])
        except ValueError:
             return jsonify({"error": "Invalid input for median_salary. Please provide a numeric value."}, 400)

        # Add server-side non-negative validation
        if value_median_salary < 0:
             return jsonify({"error": "Median salary cannot be negative."}, 400)


        # Create DataFrame for prediction
        x_pred = pd.DataFrame([[value_median_salary]], columns=['MEDIAN_SALARY'])

        # Predict GSM Score
        y_pred = model.predict(x_pred)

        # Apply rounding and clipping (assuming score is on a 0-100 scale)
        # Round to nearest integer and clip between 0 and 100
        predicted_gsm_score_clipped_rounded = int(np.round(np.clip(y_pred[0], a_min=0, a_max=100)))


        return jsonify({"predicted_gsm_score": predicted_gsm_score_clipped_rounded})

    except Exception as e:
        # Log the error for debugging on the server side
        print(f"Prediction error: {e}")
        return jsonify({"error": "An error occurred during prediction."}, 500) # Generic error message for user

if __name__ == '__main__':
    # Use a specific port or the default (5000)
    # Render will likely use a PORT environment variable, so you might need:
    # port = int(os.environ.get("PORT", 5000))
    # app.run(host="0.0.0.0", port=port)
    app.run(host="0.0.0.0", port=5001) # Use a different port than 5000 if running locally with SS backend