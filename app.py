# --- Import necessary libraries ---
from flask import Flask, render_template, request
import numpy as np
import pickle

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- Load the Machine Learning Model ---
# This line loads your pre-trained model from the file.
# Make sure "diabetes_model.pkl" is in the same directory as your app.py.
try:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
except FileNotFoundError:
    print("Error: 'diabetes_model.pkl' not found. Make sure the model file is in the correct directory.")
    model = None # Set model to None if it fails to load

# --- Define Routes for the Web Pages ---

@app.route('/')
def home():
    """Renders the main welcome page."""
    return render_template("diabcare_home.html")

@app.route('/predict-home')
def predict_home():
    return render_template('predict_home.html')
@app.route('/predict-form')
def predict_form():
    """Renders the form page for user input."""
    return render_template("predict_form.html")

@app.route('/healthy-habits')
def healthy_habits():
    """Renders the healthy habits page (you'll need to create this HTML file)."""
    # Note: You will need to create a 'healthy_habits.html' file in your templates folder.
    return render_template("healthy_habits.html")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission, makes a prediction, and shows the result.
    """
    if model is None:
        return "Model not loaded. Please check the server logs."

    try:
        # 1. Get all form values and convert them to floating-point numbers.
        features = [float(x) for x in request.form.values()]
        
        # 2. Prepare the features for the model (it expects a 2D array).
        final_input = [np.array(features)]
        
        # 3. Make the prediction (0 for Not Diabetic, 1 for Diabetic).
        prediction = model.predict(final_input)[0]
        
        # 4. Determine the result text to display on the page.
        if prediction == 1:
            result_string = "Diabetic"
            prediction_message = "The model predicts a high risk of diabetes."
        else:
            result_string = "Not Diabetic"
            prediction_message = "The model predicts a low risk of diabetes."
            
        # 5. Render the result page with the prediction outcome.
        return render_template("result.html", 
                               prediction_text=prediction_message, 
                               result=result_string)
    except Exception as e:
        # Handle potential errors, like non-numeric input.
        return f"An error occurred: {e}"

# --- Run the App ---
if __name__ == "__main__":
    app.run(debug=True)
