#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
import plotly.graph_objects as go

st.set_page_config(page_title="Sepsis Prediction Model", layout="wide")

# ====== 1) Load model & features ======
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("model.cbm")
    with open("features.json", "r", encoding="utf-8") as f:
        FEATURES = json.load(f)["features"]
    try:
        with open("config.json", "r") as f:
            THRESH = json.load(f).get("threshold", None)
    except Exception:
        THRESH = None
    return model, FEATURES, THRESH

model, FEATURES, THRESH = load_model()

# ====== 2) Title ======
st.markdown("<h2 style='text-align:center;'>Prediction Model for Sepsis Outcome</h2>", unsafe_allow_html=True)
st.write("")

# ====== 3) Inputs ======

# Continuous inputs
c1, c2 = st.columns([1, 1])
with c1:
    bp_time = st.number_input(
        "Blood Purification Time (hours)",
        min_value=0.0, max_value=200.0, step=0.5, value=24.0
    )
with c2:
    pheonix = st.number_input(
        "Pheonix score",
        min_value=0.0, max_value=100.0, step=0.1, value=10.0
    )

# Binary categorical
prep_map = {"≤6 h": 1, ">6 h": 0}
prep_choice = st.selectbox(
    "preparation time (ICU to blood purification)",
    options=list(prep_map.keys()),
    index=0
)
preparation_time = prep_map[prep_choice]

nutri_map = {"PN": 0, "EN": 1}
nutri_choice = st.selectbox(
    "Nutritional Methods",
    options=list(nutri_map.keys()),
    index=1
)
Nutritional_Methods = nutri_map[nutri_choice]

mv_map = {"No": 0, "Yes": 1}
mv_choice = st.radio(
    "mechanical ventilation",
    options=list(mv_map.keys()),
    horizontal=True
)
mechanical_ventilation = mv_map[mv_choice]

# Blood glucose (single choice → one-hot flags)
glucose_choice = st.selectbox(
    "Blood glucose level",
    options=["<7.8 mmol/L", "7.8–11.1 mmol/L", "≥11.1 mmol/L"],
    index=0
)
if glucose_choice == "<7.8 mmol/L":
    blood_glucose0, blood_glucose1, blood_glucose2 = 0, 0, 1
elif glucose_choice == "7.8–11.1 mmol/L":
    blood_glucose0, blood_glucose1, blood_glucose2 = 1, 0, 0
else:  # ≥11.1 mmol/L
    blood_glucose0, blood_glucose1, blood_glucose2 = 0, 1, 0

# ====== 4) Assemble inputs in training order ======
raw_inputs = {
    "Blood Purification Time": bp_time,
    "Pheonix score": pheonix,
    "preparation time": preparation_time,
    "Nutritional Methods": Nutritional_Methods,
    "mechanical ventilation": mechanical_ventilation,
    "blood glucose0": blood_glucose0,
    "blood glucose1": blood_glucose1,
    "blood glucose2": blood_glucose2,
}

x = np.array([[raw_inputs[f] for f in FEATURES]], dtype=float)

# ====== 5) Prediction ======
if st.button("Start Predict", use_container_width=True):
    prob = float(model.predict_proba(x)[0, 1])
    st.divider()
    st.subheader("Predict result")
    st.markdown(f"**Predicted probability of poor outcome:** `{prob*100:.1f}%`")
    st.progress(min(1.0, prob))

    if THRESH is not None:
        label = int(prob >= THRESH)
        st.write(
            f"Using threshold = **{THRESH:.3f}**, predicted class = **{label}** "
            "(1 = adverse outcome, 0 = favourable outcome)."
        )

        # ====== 5) SHAP contribution bar (CatBoost native) ======
    try:
        pool = Pool(x, feature_names=FEATURES)
        shap_vals = model.get_feature_importance(pool, type="ShapValues")
        contrib = shap_vals[0, :-1]  # per-feature raw log-odds contributions
        df = pd.DataFrame({
            "feature": FEATURES,
            "value": [raw_inputs[f] for f in FEATURES],
            "contrib": contrib
        }).sort_values("contrib", ascending=True)

        colors = ["#1f77b4" if c < 0 else "#d62728" for c in df["contrib"]]
        fig = go.Figure(go.Bar(
            x=df["contrib"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            hovertext=[f"value={v}" for v in df["value"]],
            hoverinfo="text+x+y"
        ))
        fig.update_layout(
            title="Feature contributions (SHAP, raw score space)",
            xaxis_title="Contribution (±)",
            yaxis_title="Feature",
            height=420,
            margin=dict(l=120, r=40, t=50, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red = higher prediction, Blue = lower prediction. Values are SHAP contributions in raw log-odds space.")
    except Exception as e:
        st.warning(f"Failed to draw explanation chart: {e}")

# ====== 7) Notes ======
st.markdown("---")
st.markdown("### Variable Notes")
st.markdown(
    """
- **preparation time**: Time from ICU to initiation of blood purification (≤6 h vs >6 h).
- **Nutritional Methods**: PN or EN.
- **mechanical ventilation**: Yes / No.

"""
)


