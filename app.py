from flask import Flask, render_template, request
import joblib
from flask import flash, redirect, url_for
import os
app = Flask(__name__)

@app.get('/')
def home():
    return render_template('home.html')

@app.get('/test')
def test():
    return render_template('test.html')
@app.post('/')
def submit():
    # Load the model from a joblib file
    with open('model.joblib', 'rb') as model_file:
        model = joblib.load(model_file)
    # print(type(model))  # Pastikan ini adalah DecisionTreeClassifier atau model yang sesuai

    # Get form data
    form_data = request.form
    gender = float(form_data.get('gender'))
    age = float(form_data.get('age'))
    hypertension = float(form_data.get('hypertension'))
    heart_disease = float(form_data.get('heart_disease'))
    ever_married = float(form_data.get('ever_married'))
    work_type = float(form_data.get('work_type'))
    Residence_type = float(form_data.get('residence_type'))
    avg_glucose_level = float(form_data.get('avg_glucose_level'))
    weight = float(form_data.get('weight'))
    height = float(form_data.get('height'))
    bmi = weight / ((height / 100) ** 2)
    smoking_status = float(form_data.get('smoking_status'))

    # Prepare data for prediction
    input_data = [[gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]]

    # Make prediction
    prediction = model.predict(input_data)

    # Add prediction result to the context

    if prediction[0] == 1:
        result = {
            'status': '⚠️ Deteksi menunjukkan berpotensi stroke. ⚠️',
            'recommendations': [
                "Konsultasikan dengan dokter segera dan jaga pola hidup sehat!",
                "Jaga pola makan sehat (rendah garam, tinggi serat).",
                "Rutin berolahraga minimal 30 menit sehari.",
                "Hindari merokok dan konsumsi alkohol.",
                "Pantau tekanan darah dan kadar gula secara teratur."
            ]
        }
    else:
        result = {
            'status': '✅ Tidak terdeteksi berpotensi stroke.',
            'recommendations': [
                "Tetap jaga pola hidup sehat!"
            ]
        }
    # Handle form submission here
    flash(result)
    return redirect(url_for('home'))
if __name__ == '__main__':
    app.secret_key = os.urandom(24).hex()
    app.run(debug=True,port=3000)