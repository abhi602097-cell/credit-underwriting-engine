# Transparent Credit Underwriting & Dynamic Pricing Engine

## Setup
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Folder structure
```
credit_underwriting_engine/
├── app.py
├── requirements.txt
├── data/
│   ├── internal_bank_dataset.csv   # from Internal_Bank_Dataset.xlsx
│   ├── external_cibil_dataset.csv  # from External_Cibil_Dataset.xlsx
│   └── unseen_dataset.csv          # from Unseen_Dataset.xlsx (batch scoring demo)
```
Keep `app.py` and the `data/` folder in the same directory — the app reads
the two training CSVs from `data/` on load and trains CatBoost live
(cached after the first run via `st.cache_resource`).

## What it does
1. **Cleaning** — inner-joins the two datasets on `PROSPECTID`, converts the
   legacy `-99999` placeholder to `NaN` (no mean/median imputation — CatBoost's
   `nan_mode='Min'` treats missingness as a native signal), and hardcodes the
   business `EDUCATION` ordinal mapping.
2. **Feature selection** — Chi-Square (categorical vs. target, p≤0.05) →
   sequential VIF pruning (≤6) → ANOVA F-test (p≤0.05). Full report is on the
   "Model Diagnostics" tab.
3. **Model** — CatBoostClassifier, MultiClass, iterations=600, depth=5,
   l2_leaf_reg=5, trained on an 80/20 stratified split.
4. **Pricing** — simultaneous Physical Bank vs. Digital App channel quotes,
   with a 1.50% late-night liquidity-stress penalty (23:00–04:00).
5. **Explainability** — LIME's top-5 drivers per prediction, translated into
   plain-English factor names and direction tags.
6. **Batch scoring tab** — scores the bundled `unseen_dataset.csv` (or any
   uploaded file with a compatible schema) end-to-end and offers a CSV
   download of results.
