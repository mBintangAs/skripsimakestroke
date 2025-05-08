import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Membaca Data
data = pd.read_csv('stroke_dataset_new.csv')

# Ambil kolom yang relevan
data = data[['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
             'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']]

# Tampilkan info awal
print("Informasi Dataset:")
print(data.info())
print("\nJumlah Missing Value Sebelum Penanganan:")
print(data.isnull().sum())
print("\n")

# Penanganan Missing Value

for col in ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type','Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']:
    if data[col].notna().sum() > 0:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        print(f"[WARNING] Kolom '{col}' kosong, median tidak bisa dihitung.")

# Konfirmasi setelah pemrosesan
print("Jumlah Missing Value Setelah Penanganan:")
print(data.isnull().sum())

# FPenanganan outlier menggunakan IQR dan mengganti dengan median
def handle_outliers_with_median(data, columns):
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1

        batas_bawah = Q1 - 1.5 * IQR
        batas_atas = Q3 + 1.5 * IQR
        
        # Ganti outlier dengan median
        median = data[col].median()
        data[col] = data[col].apply(lambda x: median if x < batas_bawah or x > batas_atas else x)
    
    return data

# Kolom-kolom numerik
columns_to_check = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']

# Penerapan fungsi
data_no_outlier = handle_outliers_with_median(data, columns_to_check)

# Hasil Penanganan Outlier
print("Data Setelah Penanganan Outlier:")
print(data_no_outlier.describe())

# Penerapan fungsi
data_no_outlier = handle_outliers_with_median(data, columns_to_check)

# Hasil Penanganan Outlier
print("Data Setelah Penanganan Outlier:")
print(data_no_outlier.describe())

# Bagi dataset menjadi train dan test
target = 'stroke'  # Menentukan kolom target
X = data.drop([target], axis=1)
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cek distribusi data sebelum SMOTE
print("\nDistribusi data sebelum SMOTE:")
print(y_train.value_counts())

# Terapkan SMOTE
smote = SMOTE(random_state=42,k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Cek distribusi data setelah SMOTE
print("\nDistribusi data setelah SMOTE:")
print(pd.Series(y_train_balanced).value_counts())

# Hyperparameter tuning dengan GridSearchCV
param_grid = {
    'n_estimators': [100, 150, 200, 250],
    'max_depth': [10, 15, 20, 25],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_balanced, y_train_balanced)
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
# grid_search.fit(X_train_balanced, y_train_balanced)

# Tampilkan hasil terbaik
print("Max Depth terbaik:", random_search.best_params_['max_depth'])
print("Estimator terbaik:", random_search.best_params_['n_estimators'])  # Perbaikan
print("Akurasi terbaik:", random_search.best_score_)  # Tetap gunakan ini untuk melihat skor terbaik


# Gunakan model terbaik
best_rf = random_search.best_estimator_
best_rf.fit(X_train_balanced, y_train_balanced)

# Prediksi di test set
y_pred_best_rf = best_rf.predict(X_test)

# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred_best_rf)
precision = precision_score(y_test, y_pred_best_rf)
recall = recall_score(y_test, y_pred_best_rf)
f1 = f1_score(y_test, y_pred_best_rf)
conf_matrix = confusion_matrix(y_test, y_pred_best_rf)

# Print hasil evaluasi
print("\nFinal Model Performance:")
print("Confusion Matrix:\n", conf_matrix)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Stroke', 'Stroke'],
            yticklabels=['No Stroke', 'Stroke'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
joblib.dump(best_rf, 'random_forest_model.pkl')


# Fungsi untuk menerima input dari pengguna dan melakukan prediksi
def predict_rf():
    try:
        print("Masukkan nilai untuk setiap fitur:")

        gender = float(input("GENDER (0 = Male, 1 = Female): "))
        age = float(input("AGE (0 = 1-14 tahun, 1= 15-24 tahun, 2 = 25-34 tahun, 3 = 35-44 tahun, 4 = 45-54 tahun, 5 = 55-64 tahun, 6 = 65-75 tahun, 7 =>75): "))
        hypertension = float(input("HYPERTENSION (0 = No, 1 = Yes): "))
        heart_disease = float(input("HEART DISEASE (0 = No, 1 = Yes): "))
        ever_married = float(input("EVER MARRIED (0 = No, 1 = Yes): "))
        work_type = float(input("WORK TYPE (1 = Children, 2 = Govt_job, 3 = Never Worked, 4 = Private, 5 = Self-Employed): "))
        Residence_type = float(input("RESIDENCE TYPE (1 = Rural, 2 = Urban): "))
        avg_glucose_level = float(input("GLUCOSE LEVEL (0 = <180.0, 1 = >180.0): "))

        # Input tinggi dan berat badan, lalu hitung BMI
        weight = float(input("BERAT BADAN (kg): "))
        height = float(input("TINGGI BADAN (cm): "))
        bmi = weight / ((height / 100) ** 2)

        smoking_status = float(input("SMOKING STATUS (1 = Formerly Smoked, 2 = Never Smoked, 3 = Smokes, 4 = Unknown): "))

        # Buat array fitur dari input pengguna
        user_input = [[gender, age, hypertension, heart_disease, ever_married, work_type,
                       Residence_type, avg_glucose_level, bmi, smoking_status]]

        # Lakukan prediksi menggunakan model Random Forest
        prediction = best_rf.predict(user_input)  # ✅ Ganti dari rf_model ke best_rf

        # Tampilkan hasil prediksi
        if prediction[0] == 1:
            print("\n⚠️ Deteksi menunjukkan berpotensi stroke. ⚠️")
            print("Rekomendasi:")
            print("- Konsultasikan dengan dokter segera dan jaga pola hidup sehat!")
            print("- Jaga pola makan sehat (rendah garam, tinggi serat).")
            print("- Rutin berolahraga minimal 30 menit sehari.")
            print("- Hindari merokok dan konsumsi alkohol.")
            print("- Pantau tekanan darah dan kadar gula secara teratur.")
        else:
            print("\n✅ Tidak terdeteksi berpotensi stroke. Tetap jaga pola hidup sehat!")

    except ValueError:
        print("Input tidak valid. Pastikan untuk memasukkan angka yang sesuai.")

# Panggil fungsi
predict_rf()
