
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)



# -------------------------------
# Load models and helper objects
# -------------------------------
model = joblib.load("crowd_model.pkl")
place_encoder = joblib.load("place_encoder.pkl")
crowd_encoder = joblib.load("crowd_encoder.pkl")
helper_averages = joblib.load("helper_averages.pkl")

# Month mapping
month_map = {
    'january':1, 'february':2, 'march':3, 'april':4,
    'may':5, 'june':6, 'july':7, 'august':8,
    'september':9, 'october':10, 'november':11, 'december':12
}

# -------------------------------
# Prediction function
# -------------------------------
def predict_crowd(month_name, place_name):
    month_name = month_name.lower()
    place_name = place_name.lower()

    # Convert month to number
    month_num = month_map.get(month_name, 1)

    # Encode place
    try:
        place_enc = place_encoder.transform([place_name])[0]
    except ValueError:
        # If place not in training data, use 0
        place_enc = 0

    place_month = month_num * place_enc

    # Get historical averages for helper features
    try:
        helper_vals = helper_averages.loc[(place_name, month_name)]
    except KeyError:
        helper_vals = helper_averages.mean()

    row = pd.DataFrame([{
        "Month_num": month_num,
        "Place_enc": place_enc,
        "Place_Month": place_month,
        "Monthly_Visitors": helper_vals["Monthly_Visitors"],
        "Estimated_Visitors": helper_vals["Estimated_Visitors"],
        "Temperature": helper_vals["Temperature"],
        "Rainfall": helper_vals["Rainfall"],
        "Has_Event": 1,      # Can adjust if needed
        "Has_Holiday": 1     # Can adjust if needed
    }])

    pred = model.predict(row)[0]
    return crowd_encoder.inverse_transform([pred])[0]

# -------------------------------
# Flask routes
# -------------------------------
@app.route("/")
def home():
    return "Crowd Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    month = data.get("month")
    place = data.get("place")

    if not month or not place:
        return jsonify({"error": "Please provide both 'month' and 'place'"}), 400

    prediction = predict_crowd(month, place)
    return jsonify({"crowd_level": prediction})

# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)


