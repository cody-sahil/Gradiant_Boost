import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request

app = Flask(__name__)

# Load the trained GradientBoostingRegressor model
MODEL_PATH = "gradiant.pkl"
model = None

try:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
    else:
        print(f"Warning: {MODEL_PATH} not found in root directory.")
except Exception as e:
    print(f"Error loading model: {e}")

# HTML Template with Tailwind CSS and state persistence
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Cost Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-xl w-full bg-slate-800/80 backdrop-blur-xl border border-slate-700 rounded-2xl shadow-2xl overflow-hidden p-8">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <span class="bg-indigo-500/10 text-indigo-400 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider border border-indigo-500/20">Gradient Boosting Model</span>
            <h1 class="text-3xl font-bold mt-3 text-white tracking-tight">Insurance Cost Predictor</h1>
            <p class="text-slate-400 text-sm mt-1">Enter patient attributes below to generate real-time charges estimation.</p>
        </div>

        <!-- Error Banner -->
        {% if model_error %}
        <div class="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-sm text-center">
            {{ model_error }}
        </div>
        {% endif %}

        <!-- Form -->
        <form method="POST" class="space-y-5">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <!-- Age -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Age</label>
                    <input type="number" name="age" required min="1" max="120" value="{{ form_data.age if form_data else '30' }}"
                        class="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <!-- BMI -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">BMI</label>
                    <input type="number" step="0.1" name="bmi" required min="10" max="60" value="{{ form_data.bmi if form_data else '25.0' }}"
                        class="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <!-- Children -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Children / Dependents</label>
                    <input type="number" name="children" required min="0" max="10" value="{{ form_data.children if form_data else '0' }}"
                        class="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                </div>

                <!-- Sex -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Sex</label>
                    <select name="sex" class="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                        <option value="1" {% if form_data and form_data.sex == '1' %}selected{% endif %}>Male</option>
                        <option value="0" {% if form_data and form_data.sex == '0' %}selected{% endif %}>Female</option>
                    </select>
                </div>
            </div>

            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <!-- Smoker -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Smoker</label>
                    <select name="smoker" class="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                        <option value="1" {% if form_data and form_data.smoker == '1' %}selected{% endif %}>Yes</option>
                        <option value="0" {% if not form_data or form_data.smoker == '0' %}selected{% endif %}>No</option>
                    </select>
                </div>

                <!-- Region -->
                <div>
                    <label class="block text-xs font-medium text-slate-300 uppercase tracking-wider mb-1">Region</label>
                    <select name="region" class="w-full bg-slate-900/50 border border-slate-700 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                        <option value="0" {% if form_data and form_data.region == '0' %}selected{% endif %}>Northeast</option>
                        <option value="1" {% if form_data and form_data.region == '1' %}selected{% endif %}>Northwest</option>
                        <option value="2" {% if form_data and form_data.region == '2' %}selected{% endif %}>Southeast</option>
                        <option value="3" {% if form_data and form_data.region == '3' %}selected{% endif %}>Southwest</option>
                    </select>
                </div>
            </div>

            <!-- Submit Button -->
            <button type="submit" 
                class="w-full mt-2 bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-600 hover:to-purple-700 text-white font-semibold py-3.5 px-4 rounded-xl shadow-lg shadow-indigo-500/25 transition duration-200 transform active:scale-[0.98]">
                Calculate Prediction
            </button>
        </form>

        <!-- Result Box -->
        {% if prediction is not none %}
        <div class="mt-8 p-5 bg-indigo-950/60 border border-indigo-500/30 rounded-2xl text-center animate-fade-in">
            <p class="text-xs font-medium text-indigo-300 uppercase tracking-wider">Estimated Insurance Charges</p>
            <p class="text-3xl font-extrabold text-white mt-1">${{ "%.2f"|format(prediction) }}</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    model_error = None
    form_data = None

    if request.method == "POST":
        form_data = request.form
        try:
            age = float(request.form["age"])
            sex = float(request.form["sex"])
            bmi = float(request.form["bmi"])
            children = float(request.form["children"])
            smoker = float(request.form["smoker"])
            region = float(request.form["region"])

            # Feature vector matching model expectations: [age, sex, bmi, children, smoker, region]
            features = np.array([[age, sex, bmi, children, smoker, region]])

            if model is not None:
                prediction = model.predict(features)[0]
            else:
                model_error = "Model file ('gradient.pkl') could not be loaded."
        except Exception as e:
            model_error = f"Prediction error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE, 
        prediction=prediction, 
        model_error=model_error, 
        form_data=form_data
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)