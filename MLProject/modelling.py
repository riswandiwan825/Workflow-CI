"""
Script Pelatihan Model Klasifikasi Prediksi Diabetes dengan Tracking MLflow.

Penggunaan:
    python modelling.py --n_estimators 150 --max_depth 6
"""

import os
import sys
import warnings
import argparse
import urllib.request
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# Abaikan peringatan yang tidak relevan
warnings.filterwarnings("ignore")

SERVER_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
NAMA_EKSPERIMEN = "Diabetes_Prediction_Classification"


def inisialisasi_mlflow():
    """Hubungkan ke MLflow tracking server jika aktif, atau gunakan pencatatan lokal."""
    try:
        urllib.request.urlopen(SERVER_URI, timeout=2)
        mlflow.set_tracking_uri(SERVER_URI)
        print(f"✅ Terhubung ke MLflow Tracking Server di {SERVER_URI}")
    except Exception:
        print(f"ℹ️ Tracking server tidak terjangkau di {SERVER_URI}. Menggunakan penyimpanan lokal ./mlruns")
    
    try:
        mlflow.set_experiment(NAMA_EKSPERIMEN)
    except Exception as err:
        print(f"Catatan eksperimen MLflow: {err}")


def hitung_metrik_evaluasi(y_asli: np.ndarray, y_pred: np.ndarray) -> dict:
    """Menghitung metrik performa klasifikasi."""
    return {
        "accuracy":  float(accuracy_score(y_asli, y_pred)),
        "precision": float(precision_score(y_asli, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_asli, y_pred, zero_division=0)),
        "f1_score":  float(f1_score(y_asli, y_pred, zero_division=0)),
    }


def muat_dataset_preprocessed():
    """Memuat data latih dan data uji dari CSV."""
    if not (os.path.exists('X_train.csv') and os.path.exists('y_train.csv')):
        print("[ERROR] Dataset tidak ditemukan! Jalankan notebook preprocessing terlebih dahulu.")
        sys.exit(1)

    X_latih = pd.read_csv('X_train.csv')
    X_uji   = pd.read_csv('X_test.csv')
    y_latih = pd.read_csv('y_train.csv').values.ravel()
    y_uji   = pd.read_csv('y_test.csv').values.ravel()

    return X_latih, X_uji, y_latih, y_uji


def latih_dan_catat_model(n_est: int, kedalaman: int):
    """Melatih RandomForestClassifier dan mencatat artefak ke MLflow."""
    inisialisasi_mlflow()
    X_tr, X_te, y_tr, y_te = muat_dataset_preprocessed()

    # Aktifkan pencatatan otomatis MLflow
    mlflow.autolog(log_models=True)

    with mlflow.start_run(run_name=f"RF_est{n_est}_depth{kedalaman}"):
        model_rf = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=kedalaman,
            random_state=42,
            n_jobs=-1
        )
        model_rf.fit(X_tr, y_tr)

        hasil_prediksi = model_rf.predict(X_te)
        metrik = hitung_metrik_evaluasi(y_te, hasil_prediksi)

        garis_sama = "═" * 50
        garis_strip = "─" * 50
        print(f"\n{garis_sama}")
        print("  Hasil Evaluasi Model Diabetes (Random Forest)")
        print(f"  • n_estimators : {n_est}")
        print(f"  • max_depth    : {kedalaman}")
        print(garis_strip)
        for k, v in metrik.items():
            print(f"  • {k:<12} : {v:.4f}")
        print(f"{garis_sama}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pelatihan Model Prediksi Diabetes")
    parser.add_argument("--n_estimators", type=int, default=100, help="Jumlah pohon")
    parser.add_argument("--max_depth",    type=int, default=5,   help="Kedalaman maksimal pohon")
    opsi = parser.parse_args()

    latih_dan_catat_model(opsi.n_estimators, opsi.max_depth)
