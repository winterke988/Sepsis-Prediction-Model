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
            "(1 = positive, 0 = negative)."
        )

        # ====== 6) SHAP waterfall explanation ======
    try:
        pool = Pool(x, feature_names=FEATURES)
        shap_vals = model.get_feature_importance(pool, type="ShapValues")
        contrib = shap_vals[0, :-1]   # feature contributions
        base = shap_vals[0, -1]       # base value (raw logit)

        df = pd.DataFrame({
            "feature": FEATURES,
            "value": [raw_inputs[f] for f in FEATURES],
            "contrib": contrib
        })

        # 从 base value 开始逐步累加
        df = df.sort_values("contrib", key=abs, ascending=False).reset_index(drop=True)
        df["cum"] = base + df["contrib"].cumsum()
        x_labels = [f"{f} = {v}" for f, v in zip(df["feature"], df["value"])]

        # 构建 waterfall
        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=["absolute"] + ["relative"] * len(df),
            y=["Base value"] + x_labels,
            x=[base] + df["contrib"].tolist(),
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            text=[f"{c:+.2f}" for c in [base] + df["contrib"].tolist()],
        ))

        fig.update_layout(
            title="SHAP Waterfall Explanation",
            xaxis_title="Raw log-odds contribution",
            yaxis_title="",
            showlegend=False,
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Waterfall plot: starting from base value, each feature pushes the prediction up (red) or down (blue) until the final probability."
        )
    except Exception as e:
        st.warning(f"Failed to draw SHAP waterfall chart: {e}")

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


