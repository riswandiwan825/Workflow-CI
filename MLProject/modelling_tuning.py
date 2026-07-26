"""
Script Hyperparameter Tuning Model Klasifikasi Prediksi Diabetes dengan GridSearchCV & MLflow.
"""

import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

TRACKING_URI    = "http://127.0.0.1:5000"
NAMA_EKSPERIMEN = "Diabetes_Prediction_Classification"


def jalankan_tuning():
    """Jalankan pencarian hyperparameter terbaik menggunakan GridSearchCV."""
    if not os.path.exists('X_train.csv'):
        print("[ERROR] File 'X_train.csv' tidak ditemukan! Jalankan preprocessing terlebih dahulu.")
        sys.exit(1)

    X_latih = pd.read_csv('X_train.csv')
    y_latih = pd.read_csv('y_train.csv').values.ravel()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(NAMA_EKSPERIMEN)
    mlflow.autolog()

    grid_parameter = {
        'n_estimators': [50, 100, 150],
        'max_depth':    [3, 5, 7, 10],
        'criterion':    ['gini', 'entropy']
    }

    print("🔎 Memulai proses hyperparameter tuning (Diabetes Model)...")
    with mlflow.start_run(run_name="GridSearch_Hyperparameter_Tuning"):
        estimator_base = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            estimator=estimator_base,
            param_grid=grid_parameter,
            cv=4,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_latih, y_latih)

        print("\n✅ Tuning Selesai!")
        print(f"  • Parameter Terbaik : {grid_search.best_params_}")
        print(f"  • Akurasi CV Terbaik: {grid_search.best_score_:.4f}\n")


if __name__ == '__main__':
    jalankan_tuning()
