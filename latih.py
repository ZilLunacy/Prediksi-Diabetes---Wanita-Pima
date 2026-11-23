"""latih.py - Pelatihan, seleksi model (CV di train), evaluasi test SATU kali, simpan joblib.
Butuh fitur.py (fungsi tambah_fitur) di folder yang sama."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score, accuracy_score,
                             make_scorer)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import fitur

BASE = Path(__file__).resolve().parent
NAMA_CSV = "kasus07_pima_diabetes.csv"
# cari CSV di dataset/ (struktur folder tugas) atau di folder yang sama (Colab)
DATA = next((p for p in [BASE / "dataset" / NAMA_CSV, BASE / NAMA_CSV] if p.exists()), BASE / "dataset" / NAMA_CSV)
OUT = BASE / "models"
SEED = 42
TARGET_RECALL = 0.85  # kesalahan paling mahal: pasien diabetes terlewat (false negative)


def buat_pipeline(estimator):
    prep = ColumnTransformer([("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), fitur.FITUR_MODEL)])
    return Pipeline([
        ("fitur", FunctionTransformer(fitur.tambah_fitur, validate=False)),
        ("prep", prep),
        ("clf", estimator),
    ])


def pilih_threshold(y, proba):
    """Threshold dari prediksi out-of-fold di TRAIN: maksimalkan F2 dengan recall >= target."""
    best_t, best_f2 = 0.5, -1
    for t in np.arange(0.05, 0.96, 0.01):
        pred = (proba >= t).astype(int)
        if recall_score(y, pred) >= TARGET_RECALL:
            f2 = fbeta_score(y, pred, beta=2)
            if f2 > best_f2:
                best_t, best_f2 = float(t), f2
    if best_f2 < 0:  # target recall tak tercapai -> ambil F2 tertinggi
        ts = np.arange(0.05, 0.96, 0.01)
        best_t = float(ts[np.argmax([fbeta_score(y, (proba >= t).astype(int), beta=2) for t in ts])])
    return round(best_t, 2)


def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset tidak ditemukan: {DATA}\nUnduh kasus07_pima_diabetes.csv ke folder dataset/.")
    df = pd.read_csv(DATA)
    print("=== AUDIT ===")
    print("Ukuran:", df.shape)
    print("Tipe data:", df.dtypes.astype(str).to_dict())
    print("Duplikat:", int(df.duplicated().sum()))
    print("Missing:", df.isna().sum().to_dict())
    print("Nilai <= 0 (tidak wajar):", {c: int((df[c] <= 0).sum()) for c in fitur.KOLOM_TIDAK_BOLEH_NOL})
    print("Distribusi target:", df[fitur.TARGET].value_counts().to_dict())
    df = df.drop_duplicates().reset_index(drop=True)

    y = (df[fitur.TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)  # 1 = diabetes
    X = df[fitur.FITUR_MENTAH]

    # SPLIT DULU sebelum imputasi/scaling/encoding
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    kandidat = {
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                               class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall",
               "precision": "precision", "f2": make_scorer(fbeta_score, beta=2)}

    hasil_cv = {}
    for nama, est in kandidat.items():
        r = cross_validate(buat_pipeline(est), X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
        hasil_cv[nama] = {m: [round(float(r[f"test_{m}"].mean()), 4), round(float(r[f"test_{m}"].std()), 4)]
                          for m in scoring}
        print(f"[CV] {nama}: " + ", ".join(f"{m}={v[0]:.3f}±{v[1]:.3f}" for m, v in hasil_cv[nama].items()))

    terbaik = max(hasil_cv, key=lambda n: hasil_cv[n]["roc_auc"][0])  # seleksi hanya dari CV train
    print("Model terpilih:", terbaik)

    pipe = buat_pipeline(kandidat[terbaik])
    oof = cross_val_predict(pipe, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
    threshold = pilih_threshold(y_tr.values, oof)
    print("Threshold (dari OOF train):", threshold)

    pipe.fit(X_tr, y_tr)

    # EVALUASI TEST - SATU KALI
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    metrik_test = {
        "accuracy": accuracy_score(y_te, pred), "precision": precision_score(y_te, pred, zero_division=0),
        "recall": recall_score(y_te, pred), "specificity": tn / (tn + fp),
        "f1": f1_score(y_te, pred), "f2": fbeta_score(y_te, pred, beta=2),
        "roc_auc": roc_auc_score(y_te, proba), "pr_auc": average_precision_score(y_te, proba),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "n_test": int(len(y_te)),
    }
    metrik_test = {k: (round(float(v), 4) if not isinstance(v, (dict, int)) else v) for k, v in metrik_test.items()}
    print("=== TEST (sekali) ===")
    print(json.dumps(metrik_test, indent=2))

    X_ref = fitur.tambah_fitur(X_tr)
    metadata = {
        "kasus": "Kasus 7 - Pima Diabetes", "model": terbaik, "threshold": threshold,
        "fitur_mentah": fitur.FITUR_MENTAH, "fitur_model": fitur.FITUR_MODEL,
        "kelas": {0: "Tidak diabetes (No)", 1: "Diabetes (Yes)"},
        "median_train": X_tr.replace(0, np.nan).median().round(2).to_dict(),
        "rentang_train": {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in fitur.FITUR_MENTAH},
        "hasil_cv": hasil_cv, "metrik_test": metrik_test, "n_train": int(len(y_tr)),
        "kolom_dibuang": {"(tidak ada ID)": "dataset tidak memiliki kolom ID atau kolom pasca-kejadian"},
    }
    OUT.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "metadata": metadata}, OUT / "pima_pipeline.joblib")
    (OUT / "metrik.json").write_text(json.dumps(metadata, indent=2, default=str))
    print("Tersimpan di models/pima_pipeline.joblib")


if __name__ == "__main__":
    main()
