import argparse, json, os, joblib, warnings
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from .data_prep import synthesize, load_csv, prepare, CATEGORICAL, NUMERIC, TEXT, TARGET
from .explain import make_explainer

warnings.filterwarnings("ignore")

def build_preprocessor(use_text=True):
    transformers = []
    if NUMERIC:
        transformers.append(("num", StandardScaler(), NUMERIC))
    if CATEGORICAL:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL))
    if use_text and TEXT:
        transformers.append(("txt", TfidfVectorizer(max_features=5000, ngram_range=(1,2)), "ReviewsText"))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

def build_model():
    # Start with a robust baseline classifier
    clf = LogisticRegression(max_iter=200, n_jobs=None) if hasattr(LogisticRegression, "n_jobs") else LogisticRegression(max_iter=200)
    pipe = Pipeline(steps=[
        ("prep", build_preprocessor(use_text=True)),
        ("clf", clf)
    ])
    # Small, sensible grid
    param_grid = {
        "clf__C": [0.1, 1.0, 3.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }
    return pipe, param_grid

def fit_and_eval(df, out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    df = prepare(df)

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])  # Yes/No -> 1/0
    X = df.drop(columns=[TARGET])

    pipe, param_grid = build_model()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    gscv = GridSearchCV(pipe, param_grid, scoring="f1", cv=cv, n_jobs=-1, verbose=1)
    gscv.fit(X, y)

    best = gscv.best_estimator_
    print("Best params:", gscv.best_params_)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best.fit(X_train, y_train)
    preds = best.predict(X_test)
    proba = best.predict_proba(X_test)[:,1]

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, preds)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")

    print("AUC:", auc)
    print("ACC:", acc)
    print("F1:", f1)
    print(classification_report(y_test, preds, digits=3))

    # Persist artifacts
    dump(best.named_steps["prep"], os.path.join(out_dir, "preprocessor.pkl"))
    dump(best.named_steps["clf"], os.path.join(out_dir, "model.pkl"))
    dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    with open(os.path.join(out_dir, "feature_meta.json"), "w") as f:
        json.dump({
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,
            "text": TEXT,
            "target": "Attrition",
            "metrics": {"auc": float(auc), "accuracy": float(acc), "f1": float(f1)}
        }, f, indent=2)

    print(f"Saved artifacts to: {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, default="", help="Path to HR CSV.")
    ap.add_argument("--use_synthetic", action="store_true", help="Train on synthetic data")
    ap.add_argument("--out_dir", type=str, default="models")
    args = ap.parse_args()

    if args.use_synthetic:
        df = synthesize(n=1500)
    elif args.train_csv and os.path.exists(args.train_csv):
        df = load_csv(args.train_csv)
    else:
        raise SystemExit("Provide --use_synthetic or a valid --train_csv path. See README for schema.")

    fit_and_eval(df, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
