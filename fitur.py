import numpy as np
import pandas as pd

TARGET = "type"
FITUR_MENTAH = ["npreg", "glu", "bp", "skin", "bmi", "ped", "age"]
KOLOM_TIDAK_BOLEH_NOL = ["glu", "bp", "skin", "bmi"]
FITUR_MODEL = ["npreg", "glu", "bp", "skin", "bmi", "age", "log_ped",
               "preg_rate", "glu_x_bmi", "ped_x_age", "skin_bmi_ratio"]

def tambah_fitur(X):
    X = pd.DataFrame(X)[FITUR_MENTAH].astype(float).copy()
    for c in KOLOM_TIDAK_BOLEH_NOL:
        X.loc[X[c] <= 0, c] = np.nan
    X["preg_rate"] = X["npreg"] / X["age"]
    X["glu_x_bmi"] = X["glu"] * X["bmi"] / 100.0
    X["ped_x_age"] = X["ped"] * X["age"]
    X["skin_bmi_ratio"] = X["skin"] / X["bmi"]
    X["log_ped"] = np.log1p(X["ped"])
    X = X.drop(columns=["ped"])
    return X[FITUR_MODEL].replace([np.inf, -np.inf], np.nan)
