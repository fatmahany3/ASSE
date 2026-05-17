import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="EduGuard — Student Success Tracker",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# DATA + MODEL
# ──────────────────────────────────────────────
@st.cache_resource
def load_everything():

    train_df = pd.read_csv("train_processed.csv")
    test_df  = pd.read_csv("test_processed.csv")

    X_train = train_df.drop(columns=["Target"])
    y_train = train_df["Target"]
    X_test  = test_df.drop(columns=["Target"])
    y_test  = test_df["Target"]

    feature_columns = list(X_train.columns)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    explainer = shap.LinearExplainer(model, X_train_scaled)

    preds = model.predict(X_test_scaled)

    acc = round(accuracy_score(y_test, preds) * 100, 1)
    rec = round(recall_score(y_test, preds) * 100, 1)
    f1  = round(f1_score(y_test, preds) * 100, 1)

    rows = []
    for i in range(len(X_test)):
        raw = X_test.iloc[i].to_dict()

        pf = float(model.predict_proba(X_test_scaled[i:i+1])[0][0])
        rl = "🔴 High" if pf >= 0.7 else ("🟡 Medium" if pf >= 0.4 else "🟢 Low")

        rows.append({
            "Student": f"Student {i+1:03d}",
            "Attendance": f"{raw['Attendance']:.0f}%",
            "Hours Studied / Week": f"{raw['Hours_Studied']:.0f}h",
            "Previous Score": f"{raw['Previous_Scores']:.0f}/100",
            "Chance of Failing": f"{pf*100:.1f}%",
            "Risk Level": rl,
        })

    teacher_df = pd.DataFrame(rows)

    return model, scaler, explainer, feature_columns, X_test, X_test_scaled, y_test, acc, rec, f1, teacher_df


(model, scaler, explainer, feature_columns,
 X_test, X_test_scaled, y_test,
 ACC, REC, F1, teacher_df) = load_everything()


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def run_prediction(raw):

    row = pd.DataFrame([raw])[feature_columns]
    scaled = scaler.transform(row)

    prob_fail = float(model.predict_proba(scaled)[0][0])
    shap_vals = explainer.shap_values(scaled)[0]

    risk = "HIGH" if prob_fail >= 0.7 else ("MEDIUM" if prob_fail >= 0.4 else "LOW")

    return prob_fail, shap_vals, risk


# ──────────────────────────────────────────────
# PLOTLY DASHBOARD FUNCTIONS
# ──────────────────────────────────────────────

def plot_risk_distribution():
    risk_counts = teacher_df["Risk Level"].value_counts()

    fig = go.Figure(go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.6
    ))

    fig.update_layout(title="Risk Distribution")
    return fig


def plot_study_hours():
    fig = px.histogram(X_test, x="Hours_Studied", nbins=15,
                       title="Study Hours Distribution")
    return fig


def plot_attendance_vs_scores():

    df = X_test.copy()

    df["Risk"] = [
        "High" if p >= 0.7 else "Medium" if p >= 0.4 else "Low"
        for p in model.predict_proba(X_test_scaled)[:, 0]
    ]

    fig = px.scatter(
        df,
        x="Attendance",
        y="Previous_Scores",
        color="Risk",
        title="Attendance vs Scores"
    )

    return fig


def plot_feature_importance():

    shap_all = explainer.shap_values(X_test_scaled)
    mean_abs = np.abs(shap_all).mean(axis=0)

    df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": mean_abs
    }).sort_values("Importance").tail(10)

    fig = px.bar(df, x="Importance", y="Feature",
                 orientation="h",
                 title="Top Features")

    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    view = st.radio("Who are you?", ["Teacher", "Student"])


# ──────────────────────────────────────────────
# TEACHER VIEW
# ──────────────────────────────────────────────
if "Teacher" in view:

    st.title("📊 Teacher Analytics Dashboard")

    st.write("Interactive insights about students")

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(plot_risk_distribution(), use_container_width=True)

    with col2:
        st.plotly_chart(plot_study_hours(), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(plot_attendance_vs_scores(), use_container_width=True)

    with col4:
        st.plotly_chart(plot_feature_importance(), use_container_width=True)


# ──────────────────────────────────────────────
# STUDENT VIEW (kept minimal placeholder)
# ──────────────────────────────────────────────
else:

    st.title("🎓 Student View")

    hours = st.slider("Hours studied", 0, 40, 10)
    attendance = st.slider("Attendance", 0, 100, 75)
    score = st.slider("Previous score", 0, 100, 60)

    raw = {
        "Hours_Studied": hours,
        "Attendance": attendance,
        "Previous_Scores": score,
        "Motivation_Level": 1,
        "Peer_Influence": 1,
        "Access_to_Resources": 1,
        "Parental_Involvement": 1,
        "Gender_Male": 0,
        "Extracurricular_Activities_Yes": 1,
        "Learning_Disabilities_Yes": 0,
        "Sleep_Hours": 7,
        "Tutoring_Sessions": 1,
        "Physical_Activity": 2,
        "Stress_Proxy": 0.3,
    }

    prob, shap_vals, risk = run_prediction(raw)

    st.metric("Risk of Failing", f"{prob*100:.1f}%")
    st.write("Risk Level:", risk)
