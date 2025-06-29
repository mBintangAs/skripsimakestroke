
from . import db
from flask_login import UserMixin

class User(UserMixin , db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    gender = db.Column(db.Float)
    age = db.Column(db.Integer)
    hypertension = db.Column(db.Float, default=0)
    heart_disease = db.Column(db.Float, default=0)
    ever_married = db.Column(db.Float, default=0)
    work_type = db.Column(db.Float)
    residence_type = db.Column(db.Float)
    avg_glucose_level = db.Column(db.Float)
    weight = db.Column(db.Integer)
    height = db.Column(db.Integer)
    smoking_status = db.Column(db.Float)