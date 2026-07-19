"""
Transparent Credit Underwriting and Dynamic Pricing Engine
============================================================
A production-style Streamlit application that:
  1. Ingests and programmatically cleans the internal bank + external CIBIL
     bureau datasets ("Informative Missingness" preserved as NaN).
  2. Runs a statistical feature-selection pipeline (Chi-Square -> VIF -> ANOVA).
  3. Trains a regularized, multi-class CatBoost risk model live on app start.
  4. Computes comparative dynamic pricing across a Physical Bank channel and
     a Digital App (NBFC) channel.
  5. Explains every prediction in plain English using LIME.

Run with:
    streamlit run app.py

Expected local files (see DATA_DIR below):
    data/internal_bank_dataset.csv
    data/external_cibil_dataset.csv
    data/unseen_dataset.csv   (optional - used for batch scoring demo)

Requirements:
    streamlit pandas numpy scipy scikit-learn catboost lime
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import lime
import lime.lime_tabular

# ----------------------------------------------------------------------------
# 0. GLOBAL CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Underwriting & Dynamic Pricing Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INTERNAL_CSV = os.path.join(DATA_DIR, "internal_bank_dataset.csv")
EXTERNAL_CSV = os.path.join(DATA_DIR, "external_cibil_dataset.csv")
UNSEEN_CSV = os.path.join(DATA_DIR, "unseen_dataset.csv")

TARGET_MAP = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
TIER_LABELS = {0: "P1", 1: "P2", 2: "P3", 3: "P4"}
TIER_NAMES = {
    0: "P1 — Excellent Risk",
    1: "P2 — Good Risk",
    2: "P3 — Moderate Risk",
    3: "P4 — High Risk",
}
TIER_COLORS = {
    0: "#1B7F3A",  # Green
    1: "#1F5FBF",  # Blue
    2: "#D97706",  # Orange
    3: "#C0392B",  # Red
}

EDUCATION_MAP = {
    "SSC": 1,
    "OTHERS": 1,
    "12TH": 2,
    "GRADUATE": 3,
    "UNDER GRADUATE": 3,
    "PROFESSIONAL": 3,
    "POST-GRADUATE": 4,
}

CHI2_CATEGORICAL_CANDIDATES = [
    "MARITALSTATUS",
    "GENDER",
    "last_prod_enq2",
    "first_prod_enq2",
]

BASE_RATES = {0: 9.75, 1: 11.50, 2: 16.00, 3: 28.00}

# Human-readable labels for the LIME transparency layer.
FEATURE_LABELS = {
    "NETMONTHLYINCOME": "Net Monthly Income Base",
    "Credit_Score": "CIBIL / Credit Bureau Score",
    "Age_Oldest_TL": "Length of Credit History (Oldest Line)",
    "Age_Newest_TL": "Recency of Newest Credit Line",
    "AGE": "Applicant Age",
    "EDUCATION": "Education Tier",
    "MARITALSTATUS": "Marital Status",
    "GENDER": "Gender",
    "Tot_Missed_Pmnt": "Total Missed Payments",
    "Total_TL": "Total Trade Lines Ever Opened",
    "Tot_Active_TL": "Currently Active Trade Lines",
    "Tot_Closed_TL": "Closed Trade Lines",
    "enq_L3m": "Credit Enquiries (Last 3 Months)",
    "enq_L6m": "Credit Enquiries (Last 6 Months)",
    "enq_L12m": "Credit Enquiries (Last 12 Months)",
    "tot_enq": "Total Lifetime Credit Enquiries",
    "PL_TL": "Personal Loan Trade Lines",
    "CC_TL": "Credit Card Trade Lines",
    "Time_With_Curr_Empr": "Time With Current Employer (months)",
    "last_prod_enq2": "Most Recent Product Enquiry",
    "first_prod_enq2": "First Ever Product Enquiry",
    "num_times_delinquent": "Number of Delinquent Episodes",
    "num_times_60p_dpd": "Times 60+ Days Past Due",
    "time_since_recent_payment": "Days Since Most Recent Payment",
    "time_since_recent_deliquency": "Days Since Most Recent Delinquency",
    "pct_active_tl": "Share of Active Trade Lines",
    "PL_utilization": "Personal Loan Utilization %",
    "CC_utilization": "Credit Card Utilization %",
    "PL_Flag": "Holds a Personal Loan (Flag)",
    "CC_Flag": "Holds a Credit Card (Flag)",
    "HL_Flag": "Holds a Home Loan (Flag)",
    "GL_Flag": "Holds a Gold Loan (Flag)",
}


def pretty_feature_name(raw_name: str) -> str:
    """Map a raw database column name to plain English, fall back gracefully."""
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    return raw_name.replace("_", " ").strip().title()


# ----------------------------------------------------------------------------
# 1. DATA INGESTION & PROGRAMMATIC CLEANING
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_clean_data(internal_path: str, external_path: str) -> pd.DataFrame:
    """Load, inner-join, and clean the internal + external bureau datasets."""
    internal_df = pd.read_csv(internal_path)
    external_df = pd.read_csv(external_path)

    df = pd.merge(internal_df, external_df, on="PROSPECTID", how="inner")

    # --- Informative Missingness: -99999 is a legacy "not available" code.
    # We intentionally DO NOT mean/median impute. CatBoost's native NaN
    # handling (nan_mode='Min') treats missingness itself as a signal.
    df = df.replace(-99999, np.nan)

    # --- Target cleaning: drop rows with no bureau outcome, map to ordinal.
    df = df.dropna(subset=["Approved_Flag"]).copy()
    df["Approved_Flag"] = df["Approved_Flag"].astype(str).str.strip()
    df = df[df["Approved_Flag"].isin(TARGET_MAP.keys())].copy()
    df["Approved_Flag_Num"] = df["Approved_Flag"].map(TARGET_MAP).astype(int)

    # --- Hardcoded business ordinal mapping for EDUCATION.
    df["EDUCATION"] = df["EDUCATION"].astype(str).str.strip().str.upper()
    df["EDUCATION"] = df["EDUCATION"].map(EDUCATION_MAP)
    df["EDUCATION"] = pd.to_numeric(df["EDUCATION"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["EDUCATION"]).copy()
    df["EDUCATION"] = df["EDUCATION"].astype(int)

    return df


# ----------------------------------------------------------------------------
# 2. STATISTICAL FEATURE SELECTION PIPELINE
# ----------------------------------------------------------------------------
def _fast_vif(data: pd.DataFrame, cols: list) -> dict:
    """
    Closed-form VIF via the diagonal of the inverse correlation matrix.
    Mathematically equivalent to the classic 1/(1-R^2) definition for
    standardized regressors, but avoids running N separate OLS fits per
    iteration -- essential to keep this responsive on ~50k rows.
    """
    X = data[cols].to_numpy(dtype=float)
    X = X - X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1e-9
    Xs = X / std
    corr = np.corrcoef(Xs, rowvar=False)
    corr = corr + np.eye(len(cols)) * 1e-10
    try:
        inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr)
    return {c: float(inv[i, i]) for i, c in enumerate(cols)}


@st.cache_data(show_spinner=False)
def run_feature_selection(df: pd.DataFrame):
    """
    a) Chi-Square test on categorical columns vs. Approved_Flag (p <= 0.05)
    b) Sequential VIF pruning on numeric columns (VIF <= 6)
    c) ANOVA F-test on VIF survivors across the 4 risk tiers (p <= 0.05)
    """
    report = {"chi2": {}, "vif_removed": [], "anova": {}}

    # --- (a) Chi-Square -------------------------------------------------
    chi2_selected = []
    for col in CHI2_CATEGORICAL_CANDIDATES:
        ct = pd.crosstab(df[col], df["Approved_Flag"])
        _, p_value, _, _ = chi2_contingency(ct)
        report["chi2"][col] = p_value
        if p_value <= 0.05:
            chi2_selected.append(col)

    # --- (b) VIF ----------------------------------------------------------
    exclude = {"PROSPECTID", "Approved_Flag_Num", "EDUCATION"}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]
    work = df[numeric_cols].fillna(df[numeric_cols].median())
    vif_cols = list(numeric_cols)

    while len(vif_cols) > 1:
        vifs = _fast_vif(work, vif_cols)
        worst_col = max(vifs, key=vifs.get)
        if vifs[worst_col] > 6:
            vif_cols.remove(worst_col)
            report["vif_removed"].append((worst_col, round(vifs[worst_col], 2)))
        else:
            break

    # --- (c) ANOVA ----------------------------------------------------------
    anova_selected = []
    for col in vif_cols:
        groups = [
            df.loc[df["Approved_Flag"] == tier, col].dropna()
            for tier in TARGET_MAP.keys()
        ]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            continue
        _, p_value = f_oneway(*groups)
        report["anova"][col] = p_value
        if p_value <= 0.05:
            anova_selected.append(col)

    final_categorical = chi2_selected  # e.g. MARITALSTATUS, GENDER, prod enqs
    final_numeric = anova_selected + ["EDUCATION"]  # EDUCATION always retained
    final_features = final_numeric + final_categorical

    return {
        "final_features": final_features,
        "final_categorical": final_categorical,
        "final_numeric": final_numeric,
        "report": report,
    }


# ----------------------------------------------------------------------------
# 3. MODEL TRAINING (CatBoost) + LIME EXPLAINER
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def train_pipeline(_df: pd.DataFrame, final_features: tuple, final_categorical: tuple):
    df = _df.copy()
    final_features = list(final_features)
    final_categorical = list(final_categorical)

    for c in final_categorical:
        df[c] = df[c].astype(str)

    X = df[final_features].copy()
    y = df["Approved_Flag_Num"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cat_feature_idx = [X_train.columns.get_loc(c) for c in final_categorical]

    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=600,
        depth=5,
        l2_leaf_reg=5,
        nan_mode="Min",
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train, cat_features=cat_feature_idx)

    y_pred = model.predict(X_test).flatten()
    test_accuracy = accuracy_score(y_test, y_pred)

    # --- Build a LIME-friendly numeric-encoded copy of the training data.
    lime_train = X_train.copy()
    label_encoders = {}
    categorical_names = {}
    for c in final_categorical:
        le = LabelEncoder()
        lime_train[c] = le.fit_transform(lime_train[c].astype(str))
        label_encoders[c] = le
        categorical_names[X_train.columns.get_loc(c)] = list(le.classes_)

    # Ensure numeric NaNs are filled only for LIME's internal statistics
    # (perturbation sampling) -- the underlying model still sees real NaNs.
    lime_train_filled = lime_train.fillna(lime_train.median(numeric_only=True))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=lime_train_filled.to_numpy(dtype=float),
        feature_names=list(X_train.columns),
        class_names=[TIER_LABELS[i] for i in range(4)],
        categorical_features=cat_feature_idx,
        categorical_names=categorical_names,
        mode="classification",
        discretize_continuous=True,
    )

    def predict_fn(numeric_array: np.ndarray) -> np.ndarray:
        """Decode LIME's numeric perturbations back into model-native dtypes."""
        temp = pd.DataFrame(numeric_array, columns=list(X_train.columns))
        for c in final_categorical:
            idx = np.clip(
                temp[c].round().astype(int), 0, len(label_encoders[c].classes_) - 1
            )
            temp[c] = label_encoders[c].inverse_transform(idx)
        for c in final_features:
            if c not in final_categorical:
                temp[c] = pd.to_numeric(temp[c], errors="coerce")
        temp = temp[final_features]
        return model.predict_proba(temp)

    return {
        "model": model,
        "explainer": explainer,
        "predict_fn": predict_fn,
        "label_encoders": label_encoders,
        "feature_order": list(X_train.columns),
        "cat_feature_idx": cat_feature_idx,
        "test_accuracy": test_accuracy,
        "X_train": X_train,
    }


# ----------------------------------------------------------------------------
# 4. DYNAMIC PRICING & CHANNEL ROUTING
# ----------------------------------------------------------------------------
def evaluate_channels(risk_tier: int, loan_amount: float, app_hour: int) -> dict:
    base_rate = BASE_RATES[risk_tier]
    time_penalty = 1.50 if (app_hour >= 23 or app_hour <= 4) else 0.0
    adjusted_base = base_rate + time_penalty

    # --- Channel A: Physical Bank Route
    physical_rate = round(adjusted_base - 0.25, 2)
    physical_decision = "APPROVED" if risk_tier in (0, 1) else "REJECTED"
    physical = {
        "channel": "Physical Bank Route",
        "decision": physical_decision,
        "rate": physical_rate,
        "lenders": "✅ RBI-Licensed Banks (e.g., SBI, HDFC, ICICI)",
        "badge": "REGULATED ENTITY",
    }

    # --- Channel B: Digital App Route
    digital_rate = round(adjusted_base + 0.75, 2)
    if risk_tier in (0, 1):
        digital_decision = "APPROVED"
    elif risk_tier == 2:
        digital_decision = "APPROVED" if loan_amount <= 50_000 else "REJECTED"
    else:
        digital_decision = "REJECTED"
    digital = {
        "channel": "Digital App Route",
        "decision": digital_decision,
        "rate": digital_rate,
        "lenders": "✅ RBI-Licensed NBFC Apps (e.g., Navi, KreditBee, Tata Capital)",
        "badge": "REGULATED ENTITY",
    }

    return {
        "base_rate": base_rate,
        "time_penalty": time_penalty,
        "physical": physical,
        "digital": digital,
    }


# ----------------------------------------------------------------------------
# 5. LIME -> PLAIN ENGLISH TEXT MAPPING
# ----------------------------------------------------------------------------
def get_plain_english_reasons(explainer, predict_fn, input_row: np.ndarray, predicted_class: int, top_k: int = 5):
    exp = explainer.explain_instance(
        input_row,
        predict_fn,
        num_features=top_k,
        labels=(predicted_class,),
    )
    raw = exp.as_list(label=predicted_class)

    rows = []
    for condition_text, weight in raw:
        # condition_text looks like "NETMONTHLYINCOME <= 25000.00" or
        # "GENDER=M" -- extract the leading raw feature token.
        token = condition_text.split(" ")[0].split("=")[0].strip()
        friendly = pretty_feature_name(token)
        direction = "🔼 Pushes toward this tier" if weight > 0 else "🔽 Works in applicant's favor"
        rows.append(
            {
                "Factor": friendly,
                "Raw Signal": condition_text,
                "Influence Strength": round(abs(weight), 4),
                "Direction": direction,
            }
        )
    rows.sort(key=lambda r: r["Influence Strength"], reverse=True)
    return rows


# ----------------------------------------------------------------------------
# 6. CUSTOM CSS
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
}
.main-header {
    background: linear-gradient(120deg, #0F172A 0%, #1E3A8A 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.25);
}
.main-header h1 {
    color: #FFFFFF;
    font-size: 30px;
    margin: 0;
    font-weight: 800;
}
.main-header p {
    color: #CBD5E1;
    margin-top: 6px;
    font-size: 15px;
}
.tier-block {
    padding: 22px 26px;
    border-radius: 14px;
    color: white;
    font-family: 'Sora', sans-serif;
    box-shadow: 0 6px 18px rgba(0,0,0,0.15);
}
.tier-block h2 {
    color: white !important;
    margin: 0 0 4px 0;
    font-size: 26px;
}
.tier-block p {
    margin: 0;
    opacity: 0.92;
    font-size: 14px;
}
.channel-card {
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    background: #FFFFFF;
}
.decision-badge {
    padding: 10px 14px;
    border-radius: 8px;
    font-weight: 700;
    font-size: 15px;
    margin: 10px 0 16px 0;
}
.decision-approved {
    background: #DCFCE7;
    color: #166534 !important;
}
.decision-rejected {
    background: #FEE2E2;
    color: #991B1B !important;
}
.badge-regulated {
    display: inline-block;
    background: #DCFCE7;
    color: #166534;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 4px 10px;
    border-radius: 999px;
    margin-top: 8px;
}
.rate-tag {
    font-size: 30px;
    font-weight: 800;
    font-family: 'Sora', sans-serif;
    color: #0F172A !important;
}
.section-title {
    font-family: 'Sora', sans-serif;
    font-weight: 700;
    font-size: 20px;
    margin-top: 14px;
    margin-bottom: 10px;
    padding: 6px 2px;
    color: #0F172A !important;
}
/* Force every element inside our custom cards to render dark text
   regardless of the viewer's light/dark theme -- these cards always
   sit on an explicit white/light background, so text color must
   never fall back to Streamlit's theme-driven default. */
.channel-card, .channel-card * {
    color: #0F172A !important;
}
</style>
"""


# ----------------------------------------------------------------------------
# 7. STREAMLIT UI
# ----------------------------------------------------------------------------
def render_header():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="main-header">
            <h1>🏦 Transparent Credit Underwriting &amp; Dynamic Pricing Engine</h1>
            <p>CatBoost risk scoring · statistically-selected features · LIME-explained decisions · live channel pricing</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_inputs(final_categorical, df_reference):
    st.sidebar.markdown("## 🧾 Applicant Profile")

    age = st.sidebar.slider("Age", 18, 70, 32)
    net_income = st.sidebar.number_input("Net Monthly Income (₹)", min_value=0, value=45000, step=1000)
    credit_score = st.sidebar.slider("CIBIL Score", 300, 900, 720)
    age_oldest_tl = st.sidebar.slider("Age of Oldest Trade Line (months)", 0, 400, 60)
    age_newest_tl = st.sidebar.slider("Age of Newest Trade Line (months)", 0, 200, 6)
    enq_l3m = st.sidebar.slider("Enquiries in Last 3 Months", 0, 20, 1)
    enq_l12m = st.sidebar.slider("Enquiries in Last 12 Months", 0, 40, 3)
    total_outstanding = st.sidebar.number_input("Total Outstanding Debt (₹)", min_value=0, value=120000, step=5000)

    st.sidebar.markdown("---")
    education = st.sidebar.selectbox("Education", list(EDUCATION_MAP.keys()), index=1)
    marital = st.sidebar.selectbox("Marital Status", ["Married", "Single"])
    gender = st.sidebar.selectbox("Gender", ["M", "F"])
    last_prod = st.sidebar.selectbox(
        "Last Product Enquiry", ["PL", "ConsumerLoan", "others", "AL", "CC", "HL"]
    )
    first_prod = st.sidebar.selectbox(
        "First Product Enquiry", ["PL", "ConsumerLoan", "others", "AL", "CC", "HL"], index=1
    )

    st.sidebar.markdown("---")
    app_hour = st.sidebar.slider("Application Time (24h clock)", 0, 23, 14)
    loan_amount = st.sidebar.number_input("Requested Loan Amount (₹)", min_value=1000, value=75000, step=5000)

    raw_inputs = {
        "AGE": age,
        "NETMONTHLYINCOME": net_income,
        "Credit_Score": credit_score,
        "Age_Oldest_TL": age_oldest_tl,
        "Age_Newest_TL": age_newest_tl,
        "enq_L3m": enq_l3m,
        "enq_L12m": enq_l12m,
        "EDUCATION": EDUCATION_MAP[education],
        "MARITALSTATUS": marital,
        "GENDER": gender,
        "last_prod_enq2": last_prod,
        "first_prod_enq2": first_prod,
        "_total_outstanding_debt": total_outstanding,  # informational, not modeled directly
    }
    return raw_inputs, app_hour, loan_amount


def build_model_input_row(raw_inputs: dict, final_features: list, feature_order: list, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching the model's trained feature schema.
    Any statistically-selected feature the sidebar does not directly capture
    is backfilled from the training data's median/mode so the app remains
    fully functional even though the UI only exposes the headline inputs.
    """
    row = {}
    for feat in feature_order:
        if feat in raw_inputs:
            row[feat] = raw_inputs[feat]
        elif feat in X_train.columns:
            col = X_train[feat]
            if col.dtype == object:
                row[feat] = col.mode().iloc[0] if not col.mode().empty else ""
            else:
                row[feat] = float(col.median())
        else:
            row[feat] = np.nan
    return pd.DataFrame([row])[feature_order]


def render_risk_block(tier: int, probability: float):
    color = TIER_COLORS[tier]
    st.markdown(
        f"""
        <div class="tier-block" style="background:{color};">
            <p>PREDICTED RISK PROFILE</p>
            <h2>{TIER_NAMES[tier]}</h2>
            <p>Model confidence: {probability * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    """
    Render a section header with a fully self-contained inline style
    (background + text color both set inline). This is deliberately NOT
    a shared CSS class: Streamlit's own theme stylesheet can be injected
    into the DOM after our <style> block and win the cascade even against
    !important rules of matching specificity. Inline styles on the element
    itself always beat external stylesheets, so this is immune to both
    light and dark viewer themes.
    """
    st.markdown(
        f"""
        <div style="
            font-family: 'Sora', sans-serif;
            font-weight: 700;
            font-size: 20px;
            margin-top: 14px;
            margin-bottom: 10px;
            padding: 8px 14px;
            border-radius: 8px;
            background: #F1F5F9;
            color: #0F172A;
            display: inline-block;
        ">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def render_channel_card(col, channel: dict):
    decision_class = "decision-approved" if channel["decision"] == "APPROVED" else "decision-rejected"
    decision_icon = "✅" if channel["decision"] == "APPROVED" else "⛔"
    # Built as ONE html string in ONE st.markdown call -- splitting an opening
    # and closing <div> across separate st.markdown()/st.write() calls does
    # NOT nest them in Streamlit's DOM, which silently drops the card's
    # background and breaks text contrast under a dark viewer theme.
    card_html = f"""
    <div class="channel-card">
        <h4 style="margin-top:0;">{channel['channel']}</h4>
        <div class="decision-badge {decision_class}">{decision_icon} Decision: {channel['decision']}</div>
        <div class="rate-tag">{channel['rate']:.2f}%</div>
        <p style="font-size:13px; color:#64748B !important; margin-top:4px;">
            Annualized interest rate (dynamic, incl. time-of-day adjustment)
        </p>
        <p style="margin-top:10px;">{channel['lenders']}</p>
        <span class="badge-regulated">REGULATED ENTITY</span>
    </div>
    """
    with col:
        st.markdown(card_html, unsafe_allow_html=True)


def render_batch_scoring(pipeline, final_features, final_categorical):
    section_title("📂 Batch Scoring — Unseen Applications")
    st.caption(
        "Score a full portfolio file at once. Defaults to the bundled unseen_dataset.csv; "
        "you may also upload your own file with a compatible schema."
    )

    uploaded = st.file_uploader("Upload a CSV/XLSX of new applications (optional)", type=["csv", "xlsx"])
    if uploaded is not None:
        unseen_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        source_label = uploaded.name
    elif os.path.exists(UNSEEN_CSV):
        unseen_df = pd.read_csv(UNSEEN_CSV)
        source_label = "unseen_dataset.csv (bundled)"
    else:
        st.info("No unseen dataset found. Upload a file to run batch scoring.")
        return

    st.write(f"Source: **{source_label}** · {unseen_df.shape[0]} applications, {unseen_df.shape[1]} columns")

    work = unseen_df.copy()

    # EDUCATION must go through the same ordinal business mapping.
    if "EDUCATION" in work.columns:
        work["EDUCATION"] = work["EDUCATION"].astype(str).str.strip().str.upper().map(EDUCATION_MAP)

    missing_cols = [c for c in final_features if c not in work.columns]
    if missing_cols:
        st.warning(
            f"{len(missing_cols)} model feature(s) are absent from this file and will be treated as "
            f"missing (NaN), which CatBoost handles natively: {', '.join(missing_cols)}"
        )
        for c in missing_cols:
            work[c] = np.nan

    for c in final_categorical:
        work[c] = work[c].astype(str)

    X_batch = work[final_features]

    with st.spinner("Scoring portfolio..."):
        proba = pipeline["model"].predict_proba(X_batch)
        preds = proba.argmax(axis=1)

    results = unseen_df.copy()
    results["Predicted_Tier"] = [TIER_LABELS[p] for p in preds]
    results["Confidence"] = proba.max(axis=1).round(3)

    # Apply pricing at a neutral, mid-day application time when not present in file.
    hour_series = work["time_since_recent_enq"].fillna(12).astype(int) % 24 if "time_since_recent_enq" in work.columns else pd.Series([12] * len(work))
    loan_series = work.get("NETMONTHLYINCOME", pd.Series([50000] * len(work))).fillna(50000) * 2

    phys_decision, phys_rate, dig_decision, dig_rate = [], [], [], []
    for tier, hr, amt in zip(preds, hour_series, loan_series):
        pricing = evaluate_channels(int(tier), float(amt), int(hr) % 24)
        phys_decision.append(pricing["physical"]["decision"])
        phys_rate.append(pricing["physical"]["rate"])
        dig_decision.append(pricing["digital"]["decision"])
        dig_rate.append(pricing["digital"]["rate"])

    results["Physical_Decision"] = phys_decision
    results["Physical_Rate_%"] = phys_rate
    results["Digital_Decision"] = dig_decision
    results["Digital_Rate_%"] = dig_rate

    st.dataframe(results, use_container_width=True, height=420)

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Scored Portfolio (CSV)",
        data=csv_bytes,
        file_name="scored_applications.csv",
        mime="text/csv",
    )


def main():
    render_header()

    if not (os.path.exists(INTERNAL_CSV) and os.path.exists(EXTERNAL_CSV)):
        st.error(
            "Training data not found. Please place `internal_bank_dataset.csv` and "
            f"`external_cibil_dataset.csv` inside `{DATA_DIR}`."
        )
        st.stop()

    with st.spinner("Loading and cleaning bureau + internal data..."):
        df = load_and_clean_data(INTERNAL_CSV, EXTERNAL_CSV)

    with st.spinner("Running statistical feature selection (Chi-Square → VIF → ANOVA)..."):
        selection = run_feature_selection(df)

    final_features = selection["final_features"]
    final_categorical = selection["final_categorical"]

    with st.spinner("Training CatBoost risk model + LIME explainer (first load only)..."):
        pipeline = train_pipeline(df, tuple(final_features), tuple(final_categorical))

    tab_single, tab_batch, tab_diagnostics = st.tabs(
        ["🎯 Single Application Underwriting", "📂 Batch Scoring", "🔬 Model Diagnostics"]
    )

    with tab_single:
        raw_inputs, app_hour, loan_amount = render_sidebar_inputs(final_categorical, df)

        input_row_df = build_model_input_row(
            raw_inputs, final_features, pipeline["feature_order"], pipeline["X_train"]
        )

        proba = pipeline["model"].predict_proba(input_row_df)[0]
        predicted_tier = int(np.argmax(proba))
        confidence = float(proba[predicted_tier])

        section_title("1️⃣ Risk Profile")
        render_risk_block(predicted_tier, confidence)

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("2️⃣ Market Comparison")
        pricing = evaluate_channels(predicted_tier, loan_amount, app_hour)

        if pricing["time_penalty"] > 0:
            st.info(
                f"⏱️ Liquidity-stress time penalty applied: +{pricing['time_penalty']:.2f}% "
                "(application submitted between 23:00–04:00)."
            )

        col1, col2 = st.columns(2)
        render_channel_card(col1, pricing["physical"])
        render_channel_card(col2, pricing["digital"])

        st.markdown("<br>", unsafe_allow_html=True)
        section_title("3️⃣ Transparency — Why This Decision?")

        # Encode the row for LIME the same way training data was encoded.
        lime_row = input_row_df.copy()
        for c in final_categorical:
            le = pipeline["label_encoders"][c]
            val = str(lime_row.at[0, c])
            if val not in le.classes_:
                val = le.classes_[0]
            lime_row[c] = le.transform([val])[0]
        for c in final_features:
            if c not in final_categorical:
                lime_row[c] = pd.to_numeric(lime_row[c], errors="coerce")
        lime_array = lime_row[pipeline["feature_order"]].to_numpy(dtype=float)[0]

        with st.spinner("Generating plain-English explanation (LIME)..."):
            reasons = get_plain_english_reasons(
                pipeline["explainer"], pipeline["predict_fn"], lime_array, predicted_tier, top_k=5
            )

        reasons_df = pd.DataFrame(reasons)[["Factor", "Direction", "Influence Strength", "Raw Signal"]]
        st.dataframe(reasons_df, use_container_width=True, hide_index=True)
        st.caption(
            "🔼 = increases predicted risk tier (pushes toward higher-risk classification). "
            "🔽 = decreases predicted risk tier (works in the applicant's favor)."
        )

    with tab_batch:
        render_batch_scoring(pipeline, final_features, final_categorical)

    with tab_diagnostics:
        section_title("Feature Selection Report")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Chi-Square (categorical)**")
            st.dataframe(
                pd.DataFrame(
                    [{"Feature": k, "p-value": round(v, 5), "Kept": v <= 0.05} for k, v in selection["report"]["chi2"].items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
        with c2:
            st.markdown(f"**VIF Removed ({len(selection['report']['vif_removed'])} cols)**")
            st.dataframe(
                pd.DataFrame(selection["report"]["vif_removed"], columns=["Feature", "VIF"]),
                hide_index=True,
                use_container_width=True,
            )
        with c3:
            st.markdown(f"**ANOVA Survivors ({len(selection['final_numeric']) - 1})**")
            anova_df = pd.DataFrame(
                [{"Feature": k, "p-value": v} for k, v in selection["report"]["anova"].items() if v <= 0.05]
            ).sort_values("p-value")
            st.dataframe(anova_df, hide_index=True, use_container_width=True, height=250)

        st.markdown("---")
        section_title("Model Performance")
        st.metric("Hold-out Test Accuracy", f"{pipeline['test_accuracy'] * 100:.2f}%")
        st.write(f"Final feature count: **{len(final_features)}** "
                 f"({len(selection['final_numeric'])} numeric incl. EDUCATION, {len(final_categorical)} categorical)")
        st.write("Categorical features fed to CatBoost natively:", ", ".join(final_categorical))


if __name__ == "__main__":
    main()
