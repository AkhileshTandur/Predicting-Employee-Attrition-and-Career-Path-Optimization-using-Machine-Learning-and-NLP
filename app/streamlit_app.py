import os, json, joblib, pandas as pd, numpy as np, streamlit as st
from pathlib import Path
from sklearn.metrics import roc_auc_score
from src.data_prep import prepare, synthesize, CATEGORICAL, NUMERIC, TEXT

MODEL_DIR = "models"

st.set_page_config(page_title="Attrition Risk Dashboard", layout="wide")

st.title("Employee Attrition Risk Dashboard")

colA, colB = st.columns([2,1])
with colB:
    st.markdown("#### Quick Start")
    st.write("1) Upload a CSV with employee records")
    st.write("2) Or generate synthetic data")
    use_synth = st.button("Generate synthetic demo data")

uploaded = st.file_uploader("Upload HR data CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_synth:
    df = synthesize(200)
else:
    st.info("Upload a CSV to begin, or click 'Generate synthetic demo data'.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head())

# Load artifacts
try:
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
except Exception as e:
    st.error("Models not found. Train first with `python src/train_pipeline.py --use_synthetic` or provide your CSV to train.")
    st.stop()

df_clean = prepare(df.copy())
X = df_clean.drop(columns=["Attrition"], errors="ignore")
X_trans = preprocessor.transform(X)
proba = model.predict_proba(X_trans)[:,1]
pred = (proba >= 0.5).astype(int)
labels = label_encoder.inverse_transform(pred)

st.subheader("Predictions")
res = df.copy()
res["Attrition_Prob"] = proba
res["Attrition_Pred"] = labels
st.dataframe(res.head())

st.download_button("Download predictions.csv", res.to_csv(index=False).encode("utf-8"), file_name="predictions.csv")

# Feature importances / coefficients (basic)
st.subheader("Model Insights")
try:
    if hasattr(model, "coef_"):
        st.write("Top positive coefficients (increase attrition risk):")
        import numpy as np
        feat_names = preprocessor.get_feature_names_out()
        coefs = model.coef_[0]
        idx = np.argsort(coefs)[-15:][::-1]
        top = pd.DataFrame({"feature": feat_names[idx], "coef": coefs[idx]})
        st.dataframe(top)
    elif hasattr(model, "feature_importances_"):
        feat_names = preprocessor.get_feature_names_out()
        imps = model.feature_importances_
        idx = np.argsort(imps)[-15:][::-1]
        top = pd.DataFrame({"feature": feat_names[idx], "importance": imps[idx]})
        st.dataframe(top)
    else:
        st.write("Model explainability not available for this estimator.")
except Exception as e:
    st.write("Could not compute feature importance:", e)
