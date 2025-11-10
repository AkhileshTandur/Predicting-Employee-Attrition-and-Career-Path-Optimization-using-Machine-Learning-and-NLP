import os, joblib, pandas as pd
from .data_prep import prepare, CATEGORICAL, NUMERIC, TEXT

MODEL_DIR = os.getenv("MODEL_DIR", "models")

def load_artifacts(model_dir=MODEL_DIR):
    prep = joblib.load(os.path.join(model_dir, "preprocessor.pkl"))
    clf = joblib.load(os.path.join(model_dir, "model.pkl"))
    le = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return prep, clf, le

def predict(df: pd.DataFrame, model_dir=MODEL_DIR):
    prep, clf, le = load_artifacts(model_dir)
    df_clean = prepare(df)
    X = df_clean.drop(columns=["Attrition"], errors="ignore")
    X_trans = prep.transform(X)
    proba = clf.predict_proba(X_trans)[:,1]
    labels = (proba >= 0.5).astype(int)
    yhat = le.inverse_transform(labels)
    out = df.copy()
    out["Attrition_Prob"] = proba
    out["Attrition_Pred"] = yhat
    return out

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/infer.py path/to/new_data.csv")
        sys.exit(0)
    df = pd.read_csv(sys.argv[1])
    res = predict(df)
    res.to_csv("predictions.csv", index=False)
    print("Saved predictions.csv")
