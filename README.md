# Employee Attrition & Career Optimization (ML + NLP)

End-to-end **data science project** that predicts employee attrition and provides explainability. Includes:
- Data pipeline with numeric, categorical, and text features
- Model training with cross-validation and metrics
- Explainability via SHAP (optional)
- Streamlit app for interactive predictions

## Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Train on **synthetic data** to test end-to-end
python src/train_pipeline.py --use_synthetic

# 4) (Recommended) Train on your own HR CSV
#    Expecting columns similar to:
#    EmployeeID,Age,Department,Education,Gender,MonthlyIncome,JobSatisfaction,YearsAtCompany,NumPromotions,ReviewsText,Attrition
python src/train_pipeline.py --train_csv data/hr_data.csv

# 5) Launch the Streamlit app
streamlit run app/streamlit_app.py
```

## Data Format

Minimum expected columns (case-sensitive):
- `EmployeeID` (unique id)
- `Age` (int)
- `Department` (category)
- `Education` (category or numeric)
- `Gender` (category)
- `MonthlyIncome` (numeric)
- `JobSatisfaction` (1-5 or 1-10)
- `YearsAtCompany` (numeric)
- `NumPromotions` (numeric)
- `ReviewsText` (free text; can be empty)
- `Attrition` (target) values: `"Yes"` or `"No"`

> If you don't have text reviews, keep `ReviewsText` blank; the pipeline handles it.

## Artifacts

Training saves to `models/`:
- `preprocessor.pkl` – ColumnTransformer for data prep
- `model.pkl` – final classifier
- `label_encoder.pkl` – for 'Attrition' encoding
- `feature_meta.json` – metadata about features
- `vectorizer.pkl` (only when NLP enabled)

## Customization

- Edit model & search space in `src/train_pipeline.py`
- Add or remove features in `src/data_prep.py`
- Replace SHAP explainer in `src/explain.py`

## License
MIT
