import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

# Load the trained LogisticRegression model from the pickle file
try:
    with open("exg.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

app = Flask(__name__)

# Basic HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f7f6;
            margin: 0;
            padding: 40px;
            display: flex;
            justify-content: center;
        }
        .container {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            max-width: 400px;
            width: 100%;
        }
        h2 {
            margin-top: 0;
            color: #333;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #666;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 10px;
            background-color: #007bff;
            border: none;
            border-radius: 4px;
            color: white;
            font-size: 16px;
            cursor: pointer;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            background-color: #e9ecef;
            text-align: center;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Prediction Form</h2>
        <form method="POST" action="/predict">
            <div class="form-group">
                <label for="user_id">User ID:</label>
                <input type="number" id="user_id" name="user_id" required>
            </div>
            <div class="form-group">
                <label for="gender">Gender:</label>
                <select id="gender" name="gender" required>
                    <option value="1">Male</option>
                    <option value="0">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label for="age">Age:</label>
                <input type="number" id="age" name="age" required>
            </div>
            <div class="form-group">
                <label for="estimated_salary">Estimated Salary:</label>
                <input type="number" id="estimated_salary" name="estimated_salary" required>
            </div>
            <button type="submit">Predict</button>
        </form>

        {% if prediction_text %}
            <div class="result">
                {{ prediction_text }}
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction_text="Error: Model file not loaded properly.")

    try:
        # Extract features matching model expected columns: [User ID, Gender, Age, EstimatedSalary]
        user_id = float(request.form.get("user_id", 0))
        gender = float(request.form.get("gender", 0))
        age = float(request.form.get("age", 0))
        estimated_salary = float(request.form.get("estimated_salary", 0))

        # Format input array
        features = np.array([[user_id, gender, age, estimated_salary]])
        
        # Predict class
        prediction = model.predict(features)[0]
        
        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text=f"Prediction Output: {prediction}"
        )
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error processing input: {e}")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for JSON payload requests."""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True)
        user_id = data.get("User ID", 0)
        gender = data.get("Gender", 0)
        age = data.get("Age", 0)
        salary = data.get("EstimatedSalary", 0)

        features = np.array([[user_id, gender, age, salary]])
        prediction = model.predict(features)[0]

        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
