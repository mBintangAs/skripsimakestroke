from flask import Flask, render_template, request
import joblib
from flask import flash, redirect, url_for
import os
from flask_login import LoginManager,login_user,current_user,login_required
from model import db
from model.user import User
from werkzeug.security import generate_password_hash,check_password_hash

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://django:root@localhost/anxiety'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.secret_key = 'INI_SECRET_KEY'

db.init_app(app)  # Inisialisasi db dengan app
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'SignIn'  # Nama endpoint login

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.get('/')
def home():
    # Cek apakah user sudah login
    if current_user.is_authenticated:
        user = User.query.filter_by(id=current_user.id).first()
        if user.age is None:
            return redirect(url_for('profile'))
    return render_template('home.html')
@app.get('/profile')
def profile():
    user = User.query.filter_by(id=current_user.id).first()
    return render_template('profile.html', user=user)
@app.post('/profile')
def profile_post():
    # Cek apakah user sudah login
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    user = User.query.filter_by(id=current_user.id).first()
    # Ambil data dari form
    user.age = request.form.get('age')
    user.gender = request.form.get('gender')
    user.weight = request.form.get('weight')
    user.height = request.form.get('height')
    user.avg_glucose_level = request.form.get('avg_glucose_level')
    user.smoking_status = request.form.get('smoking_status')
    user.hypertension = 1 if request.form.get('hypertension') == 'on' else 0
    user.heart_disease = 1 if request.form.get('heart_disease') == 'on' else 0
    user.ever_married = 1 if request.form.get('ever_married') == 'on' else 0
    user.work_type = request.form.get('work_type')
    user.residence_type = request.form.get('residence_type')
    db.session.add(user)
    db.session.commit()
    flash('Profile updated successfully!', 'success')
    return redirect(url_for('profile'))
@app.get('/register')
def register():
    return render_template('register.html')

@app.get('/logout')
def logout():
    if current_user.is_authenticated:
        from flask_login import logout_user
        logout_user()
    return redirect(url_for('home'))
@app.get('/login')
def login():
    return render_template('login.html')
@app.post('/login')
def login_post():
    username = request.form.get('username')
    password = request.form.get('password')

    user = User.query.filter_by(username=username).first()
    if user and check_password_hash(user.password, password):
        login_user(user)
        return redirect(url_for('home'))
    flash('Invalid username or password', 'error')
    return redirect(url_for('login'))

@app.post('/register')
def register_post():
    name = request.form.get('name')
    username = request.form.get('username')
    password = request.form.get('password')
   
    # Cek apakah username sudah ada
    existing_user = User.query.filter_by(username=username).first()
    if existing_user:
        flash('Username sudah terdaftar', 'error')
        return redirect(url_for('register'))

    # Buat user baru
    hashed_password = generate_password_hash(password)
    new_user = User(username=username, password=hashed_password, name=name)
    db.session.add(new_user)
    db.session.commit()

    flash('Registrasi berhasil! Silakan login.', 'success')
    return redirect(url_for('login'))
@app.get('/deteksi')
def deteksi():
    # Cek apakah user sudah login
    if not current_user.is_authenticated:
        return redirect(url_for('login')) 
    user = User.query.filter_by(id=current_user.id).first()
    
    return render_template('deteksi.html', user=user)

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
    hypertension = 1.0 if form_data.get('hypertension') == 'on' else 0.0
    heart_disease = 1.0 if form_data.get('heart_disease') == 'on' else 0.0
    ever_married = 1.0 if form_data.get('ever_married') == 'on' else 0.0
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
        return redirect(url_for('positive'))
        
    else:
        return redirect(url_for('negative'))
        
    # Handle form submission here
    flash(result)
    print(result)
    return redirect(url_for('home'))



@app.get('/negative')
def negative():
    return render_template('negative.html')
@app.get('/positive')
def positive():
    return render_template('positive.html')
if __name__ == '__main__':
    with app.app_context():
        # db.drop_all()
        db.create_all()  # Membuat tabel jika belum ada
    app.run(debug=True,port=3000)