import numpy as np
import shap

def make_explainer(model, background):
    try:
        explainer = shap.Explainer(model.predict_proba, background, check_additivity=False)
        return explainer
    except Exception:
        return None

def shap_values(explainer, X):
    if explainer is None:
        return None
    try:
        sv = explainer(X)
        return sv
    except Exception:
        return None
