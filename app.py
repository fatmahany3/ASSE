# app.py — EduPulse Phase 4 Dashboard

import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="EduPulse — Smart Student Monitoring",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CLEAN CSS
# =====================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background:#f5f7fb;
}

.block-container {
    padding-top:2rem;
    padding-bottom:2rem;
}

section[data-testid="stSidebar"] {
    background:#ffffff;
    border-right:1px solid #e5e7eb;
}

.dashboard-title {
    font-size:2.3rem;
    font-weight:800;
    color:#111827;
}

.dashboard-sub {
    color:#6b7280;
    margin-bottom:2rem;
}

.kpi-card {
    background:white;
    border-radius:20px;
    padding:1.5rem;
    border:1px solid #edf0f7;
    box-shadow:0 2px 15px rgba(0,0,0,0.04);
}

.kpi-label {
    color:#6b7280;
    font-size:0.9rem;
    font-weight:600;
}

.kpi-value {
    font-size:2rem;
    font-weight:800;
    color:#111827;
    margin-top:0.4rem;
}

.alert-box {
    background:#fff4f4;
    border-left:5px solid #ef4444;
    padding:1rem;
    border-radius:14px;
    margin-bottom:1rem;
}

.insight-card {
    background:white;
    border-radius:18px;
    padding:1.2rem;
    border:1px solid #edf0f7;
}

.high-risk {
    background:#fee2e2;
    color:#dc2626;
    padding:0.3rem 0.8rem;
    border-radius:999px;
    font-size:0.75rem;
    font-weight:700;
}

.medium-risk {
    background:#fef3c7;
    color:#d97706;
    padding:0.3rem 0.8rem;
    border-radius:999px;
    font-size:0.75rem;
    font-weight:700;
}

.low-risk {
    background:#dcfce7;
    color:#16a34a;
    padding:0.3rem 0.8rem;
    border-radius:999px;
    font-size:0.75rem;
    font-weight:700;
}

.stButton > button {
    background:#2563eb;
    color:white;
    border:none;
    border-radius:10px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA + MODEL
# =====================================================
@st.cache_resource

def load_model_and_data():

    train_df = pd.read_csv("train_processed.csv")
    test_df = pd.read_csv("test_processed.csv")

    X_train = train_df.drop(columns=["Target"])
    y_train = train_df["Target"]

    X_test = test_df.drop(columns=["Target"])
    y_test = test_df["Target"]

    feature_columns = list(X_train.columns)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    explainer = shap.LinearExplainer(model, X_train_scaled)

    preds = model.predict(X_test_scaled)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds) * 100, 1),
        "recall": round(recall_score(y_test, preds) * 100, 1),
        "f1": round(f1_score(y_test, preds) * 100, 1)
    }

    teacher_rows = []

    for i in range(len(X_test)):

        raw = X_test.iloc[i]

        fail_prob = float(model.predict_proba(X_test_scaled[i:i+1])[0][0])

        risk = (
            "HIGH" if fail_prob >= 0.7
            else "MEDIUM" if fail_prob >= 0.4
            else "LOW"
        )

        teacher_rows.append({
            "Student ID": f"S{i+1:03d}",
            "Attendance (%)": raw["Attendance"],
            "Hours Studied": raw["Hours_Studied"],
            "Previous Score": raw["Previous_Scores"],
            "Fail Prob (%)": round(fail_prob * 100, 1),
            "Risk Level": risk
        })

    teacher_df = pd.DataFrame(teacher_rows)

    return (
        model,
        scaler,
        explainer,
        feature_columns,
        X_test_scaled,
        metrics,
        teacher_df
    )


(
    model,
    scaler,
    explainer,
    feature_columns,
    X_test_scaled,
    metrics,
    teacher_df
) = load_model_and_data()

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.title("🎓 EduPulse")

    view = st.radio(
        "Navigation",
        [
            "👨‍🏫 Teacher Dashboard",
            "🎓 Student Self-Assessment"
        ]
    )

    st.markdown("---")

    st.subheader("📊 System Reliability")

    st.metric("Accuracy", f"{metrics['accuracy']}%")
    st.metric("Risk Detection", f"{metrics['recall']}%")
    st.metric("Balance Score", f"{metrics['f1']}%")

# =====================================================
# TEACHER DASHBOARD
# =====================================================
if "Teacher" in view:

    st.markdown(
        """
        <div class='dashboard-title'>
        🎓 EduPulse Teacher Dashboard
        </div>

        <div class='dashboard-sub'>
        Monitor student performance, identify risks early, and support students effectively.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =================================================
    # KPI SECTION
    # =================================================

    high = len(teacher_df[teacher_df["Risk Level"] == "HIGH"])
    medium = len(teacher_df[teacher_df["Risk Level"] == "MEDIUM"])
    low = len(teacher_df[teacher_df["Risk Level"] == "LOW"])
    total = len(teacher_df)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Total Students</div>
            <div class='kpi-value'>{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>High Risk</div>
            <div class='kpi-value'>{high}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Need Attention</div>
            <div class='kpi-value'>{medium}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Doing Well</div>
            <div class='kpi-value'>{low}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # =================================================
    # AI SUMMARY
    # =================================================

    st.subheader("📌 AI Summary")

    st.markdown(f"""
    <div class='alert-box'>

    <b>{high}</b> students are currently at high risk.

    The main issue affecting student performance is <b>low attendance</b>.

    Students with attendance below 60% are much more likely to fail.

    Recommended Actions:
    <ul>
        <li>Contact parents of absent students</li>
        <li>Provide additional learning resources</li>
        <li>Schedule one-on-one follow-up meetings</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

    # =================================================
    # CHARTS
    # =================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Risk Distribution")

        risk_counts = teacher_df["Risk Level"].value_counts()

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            hole=0.55
        )

        fig.update_layout(
            paper_bgcolor="#f5f7fb",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        st.subheader("📈 Attendance vs Performance")

        fig2 = px.scatter(
            teacher_df,
            x="Attendance (%)",
            y="Previous Score",
            color="Risk Level",
            size="Fail Prob (%)",
            hover_data=["Student ID"]
        )

        fig2.update_layout(
            paper_bgcolor="#f5f7fb",
            plot_bgcolor="white",
            height=400
        )

        st.plotly_chart(fig2, use_container_width=True)

    # =================================================
    # ALERTS
    # =================================================

    st.subheader("🚨 Today's Alerts")

    alerts = teacher_df[
        (teacher_df["Attendance (%)"] < 60) |
        (teacher_df["Fail Prob (%)"] > 70)
    ]

    for _, row in alerts.head(5).iterrows():

        st.markdown(f"""
        <div class='alert-box'>

        <b>{row['Student ID']}</b> may need attention.<br><br>

        • Attendance: {row['Attendance (%)']}%<br>
        • Risk Score: {row['Fail Prob (%)']}%

        </div>
        """, unsafe_allow_html=True)

    # =================================================
    # STUDENT OVERVIEW
    # =================================================

    st.subheader("👨‍🎓 Student Overview")

    risk_filter = st.selectbox(
        "Filter Students",
        ["All", "HIGH", "MEDIUM", "LOW"]
    )

    filtered = teacher_df.copy()

    if risk_filter != "All":
        filtered = filtered[filtered["Risk Level"] == risk_filter]

    for _, row in filtered.iterrows():

        risk = row["Risk Level"]

        badge = {
            "HIGH": "high-risk",
            "MEDIUM": "medium-risk",
            "LOW": "low-risk"
        }[risk]

        label = {
            "HIGH": "High Risk",
            "MEDIUM": "Needs Attention",
            "LOW": "On Track"
        }[risk]

        with st.expander(f"{row['Student ID']} — {label}"):

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Attendance", f"{row['Attendance (%)']}%")

            with c2:
                st.metric("Study Hours", f"{row['Hours Studied']} hrs")

            with c3:
                st.metric("Risk Score", f"{row['Fail Prob (%)']}%")

            st.markdown(
                f"<span class='{badge}'>{label}</span>",
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown("### Why this student may struggle")

            reasons = []

            if row["Attendance (%)"] < 60:
                reasons.append("Low attendance")

            if row["Hours Studied"] < 12:
                reasons.append("Limited study hours")

            if row["Previous Score"] < 60:
                reasons.append("Weak previous academic performance")

            if len(reasons) == 0:
                reasons.append("Student is currently performing well")

            for r in reasons:
                st.write("•", r)

            st.markdown("### Recommended Actions")

            if risk == "HIGH":
                st.error("Immediate intervention recommended")

            elif risk == "MEDIUM":
                st.warning("Monitor progress weekly")

            else:
                st.success("Student is progressing well")

            b1, b2, b3 = st.columns(3)

            with b1:
                st.button("📩 Send Alert", key=f"a{row['Student ID']}")

            with b2:
                st.button("📚 Assign Resources", key=f"b{row['Student ID']}")

            with b3:
                st.button("📞 Contact Parent", key=f"c{row['Student ID']}")

# =====================================================
# STUDENT VIEW
# =====================================================
else:

    st.markdown(
        """
        <div class='dashboard-title'>
        🎓 Student Academic Health Check
        </div>

        <div class='dashboard-sub'>
        Fill in your information to receive personalised academic insights.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("student_form"):

        c1, c2, c3 = st.columns(3)

        with c1:
            attendance = st.slider("Attendance (%)", 40, 100, 75)
            hours = st.slider("Study Hours / Week", 0, 40, 15)

        with c2:
            previous = st.slider("Previous Score", 40, 100, 70)
            sleep = st.slider("Sleep Hours", 4, 10, 7)

        with c3:
            motivation = st.selectbox(
                "Motivation Level",
                ["Low", "Medium", "High"]
            )

        submit = st.form_submit_button("Analyze My Academic Health")

    if submit:

        fail_prob = max(
            0.05,
            min(
                0.95,
                1 - (
                    attendance/100 * 0.4 +
                    previous/100 * 0.4 +
                    hours/40 * 0.2
                )
            )
        )

        pass_prob = 1 - fail_prob

        risk = (
            "HIGH" if fail_prob >= 0.7
            else "MEDIUM" if fail_prob >= 0.4
            else "LOW"
        )

        if risk == "HIGH":
            st.error("⚠️ You may need extra academic support")

        elif risk == "MEDIUM":
            st.warning("📘 You're doing okay, but there is room for improvement")

        else:
            st.success("✅ Great job! You are on track")

        # =============================================
        # RESULT METRICS
        # =============================================

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Risk Score", f"{fail_prob*100:.1f}%")

        with c2:
            st.metric("Success Chance", f"{pass_prob*100:.1f}%")

        with c3:
            st.metric("Attendance", f"{attendance}%")

        st.write("")

        # =============================================
        # CLASS COMPARISON
        # =============================================

        class_att = teacher_df["Attendance (%)"].mean()
        class_score = teacher_df["Previous Score"].mean()

        x1, x2 = st.columns(2)

        with x1:
            st.info(f"""
            📊 Your Attendance: {attendance}%

            Class Average: {class_att:.1f}%
            """)

        with x2:
            st.info(f"""
            📝 Your Score: {previous}

            Class Average: {class_score:.1f}
            """)

        # =============================================
        # PROGRESS BARS
        # =============================================

        st.subheader("📈 Learning Progress")

        st.write("Attendance")
        st.progress(attendance / 100)

        st.write("Study Consistency")
        st.progress(min(hours / 30, 1.0))

        st.write("Sleep Quality")
        st.progress(min(sleep / 8, 1.0))

        # =============================================
        # RECOMMENDATIONS
        # =============================================

        st.subheader("💡 Personalized Recommendations")

        recommendations = []

        if attendance < 70:
            recommendations.append("Improve attendance consistency")

        if hours < 15:
            recommendations.append("Increase weekly study hours")

        if sleep < 6:
            recommendations.append("Improve sleep schedule")

        if previous < 65:
            recommendations.append("Review previous weak subjects")

        if len(recommendations) == 0:
            recommendations.append("Keep maintaining your current performance")

        for rec in recommendations:
            st.success(rec)
