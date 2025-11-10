import os
import pandas as pd
import numpy as np

DEFAULT_COLUMNS = [
    "EmployeeID","Age","Department","Education","Gender","MonthlyIncome",
    "JobSatisfaction","YearsAtCompany","NumPromotions","ReviewsText","Attrition"
]

CATEGORICAL = ["Department","Education","Gender"]
NUMERIC = ["Age","MonthlyIncome","JobSatisfaction","YearsAtCompany","NumPromotions"]
TEXT = ["ReviewsText"]
TARGET = "Attrition"

def synthesize(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    departments = ["Sales","R&D","HR","Finance","Operations","IT","Marketing"]
    educ_levels = ["High School","Bachelors","Masters","PhD"]
    genders = ["Male","Female","Other"]

    df = pd.DataFrame({
        "EmployeeID": np.arange(1, n+1),
        "Age": rng.integers(21, 60, n),
        "Department": rng.choice(departments, n, replace=True),
        "Education": rng.choice(educ_levels, n, replace=True),
        "Gender": rng.choice(genders, n, replace=True),
        "MonthlyIncome": rng.normal(6000, 2000, n).clip(2500, 20000).round(0),
        "JobSatisfaction": rng.integers(1, 5, n),
        "YearsAtCompany": rng.integers(0, 20, n),
        "NumPromotions": rng.poisson(0.3, n).clip(0, 5),
        "ReviewsText": rng.choice([
            "Great manager and flexible hours.",
            "Too much overtime and unclear goals.",
            "Supportive team, decent pay, growth options.",
            "Toxic culture, considering leaving soon.",
            "Challenging projects and helpful mentorship."
        ], n, replace=True)
    })

    # latent probability of attrition
    p = (
        0.12
        + 0.15*(df["JobSatisfaction"]<=2).astype(float)
        + 0.10*(df["YearsAtCompany"]<1).astype(float)
        + 0.10*(df["MonthlyIncome"]<4000).astype(float)
    ).clip(0.02, 0.85)
    df["Attrition"] = np.where(rng.random(n) < p, "Yes", "No")
    return df

def load_csv(path):
    df = pd.read_csv(path)
    missing = [c for c in DEFAULT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def prepare(df: pd.DataFrame):
    # Basic cleaning
    df = df.copy()
    for c in CATEGORICAL + TEXT:
        if c in df.columns:
            df[c] = df[c].fillna("")
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[NUMERIC] = df[NUMERIC].fillna(df[NUMERIC].median(numeric_only=True))

    return df
