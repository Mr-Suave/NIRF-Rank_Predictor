from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and train model
df = pd.read_csv('Engineering.csv')
df = df[['TLR(100)', 'RPC(100)', 'GO(100)', 'OI(100)', 'Perception(100)', 'Score', 'Ranking']]
X = df.drop(columns=['Ranking'])
y = df['Ranking']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = PoissonRegressor()
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        x_pred = pd.DataFrame([[data['tlr'], data['rpc'], data['go'], data['oi'], data['perception'], data['score']]], 
                              columns=X.columns)
        y_pred = model.predict(x_pred)
        return jsonify({"predicted_rank": int(y_pred[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
