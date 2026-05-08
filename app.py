import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="ASSE — Student Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.main { background: #0a0e1a; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b27 100%);
    border-right: 1px solid #1e2d40;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #8b9ab5 !important;
    font-size: 0.95rem;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e2d40;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.metric-card.blue::before { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
.metric-card.green::before { background: linear-gradient(90deg, #10b981, #34d399); }
.metric-card.red::before { background: linear-gradient(90deg, #ef4444, #f87171); }
.metric-card.purple::before { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.metric-card.amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.metric-label {
    color: #6b7a99;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.3rem;
}
.metric-value {
    color: #e2e8f0;
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
    font-family: 'JetBrains Mono', monospace;
}
.metric-sub {
    color: #4a5568;
    font-size: 0.78rem;
    margin-top: 0.2rem;
}

/* Risk Badge */
.risk-high { 
    background: rgba(239,68,68,0.15); 
    color: #f87171; 
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.85rem;
}
.risk-medium { 
    background: rgba(245,158,11,0.15); 
    color: #fbbf24; 
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.85rem;
}
.risk-low { 
    background: rgba(16,185,129,0.15); 
    color: #34d399; 
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.85rem;
}

/* Student score ring */
.score-ring {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem 0;
}

/* Section header */
.section-header {
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid #1e2d40;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Recommendation card */
.rec-card {
    background: #111827;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
}
.rec-icon {
    font-size: 1.4rem;
    margin-top: 0.1rem;
}
.rec-title {
    color: #e2e8f0;
    font-weight: 600;
    font-size: 0.9rem;
}
.rec-topic {
    color: #6b7a99;
    font-size: 0.78rem;
    margin-top: 0.15rem;
}
.rec-link {
    color: #60a5fa;
    font-size: 0.78rem;
    text-decoration: none;
}

/* Teacher table */
.teacher-table th {
    background: #111827;
    color: #8b9ab5;
}

/* App header */
.app-header {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 1px solid #1e2d40;
    margin-bottom: 1.5rem;
}
.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.app-subtitle {
    color: #4a5568;
    font-size: 0.85rem;
}

/* Form styling */
.stSlider label { color: #8b9ab5 !important; font-size: 0.85rem !important; }
.stSelectbox label { color: #8b9ab5 !important; font-size: 0.85rem !important; }
.stNumberInput label { color: #8b9ab5 !important; font-size: 0.85rem !important; }

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 0.9rem;
    width: 100%;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(99,102,241,0.35);
}

/* dataframe */
.dataframe { background: #0d1117 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# Load Artifacts + Train Model
# ─────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
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

    explainer = shap.LinearExplainer(model, X_train_scaled, feature_perturbation="interventional")

    preds   = model.predict(X_test_scaled)
    probas  = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds) * 100, 1),
        "recall":   round(recall_score(y_test, preds) * 100, 1),
        "f1":       round(f1_score(y_test, preds) * 100, 1),
    }

    # Build teacher table
    shap_vals_all = explainer.shap_values(X_test_scaled)
    teacher_rows  = []
    for i in range(len(X_test)):
        raw   = X_test.iloc[i].to_dict()
        proba = float(model.predict_proba(X_test_scaled[i:i+1])[0][0])
        rl    = "HIGH" if proba >= 0.7 else ("MEDIUM" if proba >= 0.4 else "LOW")
        teacher_rows.append({
            "Student ID":    f"S{i+1:03d}",
            "Attendance (%)":     raw["Attendance"],
            "Hours Studied":      raw["Hours_Studied"],
            "Previous Score":     raw["Previous_Scores"],
            "Fail Prob (%)":      round(proba * 100, 1),
            "Risk Level":         rl,
        })
    teacher_df = pd.DataFrame(teacher_rows)

    return (model, scaler, explainer, feature_columns,
            X_train_scaled, X_test_scaled, X_test, y_test,
            metrics, teacher_df)

# ─────────────────────────────────────────
# Resources Catalog
# ─────────────────────────────────────────
RESOURCES = [
    {"id":"R01","title":"Khan Academy — Mathematics Fundamentals","topic":"Math","icon":"📐",
     "url":"https://www.khanacademy.org/math","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R02","title":"Coursera — Learning How to Learn","topic":"Study Skills","icon":"🧠",
     "url":"https://www.coursera.org/learn/learning-how-to-learn","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R03","title":"Pomodoro Technique — Study Timer App","topic":"Productivity","icon":"⏱️",
     "url":"https://pomofocus.io","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R04","title":"Anki — Spaced Repetition Flashcards","topic":"Memory","icon":"🃏",
     "url":"https://apps.ankiweb.net","motivation":0,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R05","title":"MIT OpenCourseWare — Science & Engineering","topic":"Science","icon":"🔬",
     "url":"https://ocw.mit.edu","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R06","title":"Crash Course — Science & Humanities Videos","topic":"General","icon":"🎬",
     "url":"https://www.youtube.com/crashcourse","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R07","title":"Quizlet — Interactive Study Sets","topic":"Study Skills","icon":"✏️",
     "url":"https://quizlet.com","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R08","title":"Sleep Foundation — Healthy Sleep Guide","topic":"Wellness","icon":"😴",
     "url":"https://www.sleepfoundation.org/teens-and-sleep","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"id":"R09","title":"Headspace — Student Mindfulness & Stress Relief","topic":"Wellness","icon":"🧘",
     "url":"https://www.headspace.com/students","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"id":"R10","title":"Calm — Sleep & Relaxation for Students","topic":"Wellness","icon":"🌙",
     "url":"https://www.calm.com","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"id":"R11","title":"TED-Ed — Motivational Student Talks","topic":"Motivation","icon":"🎤",
     "url":"https://ed.ted.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R12","title":"Growth Mindset — Carol Dweck","topic":"Motivation","icon":"💡",
     "url":"https://www.mindsetonline.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R13","title":"SMART Goals Worksheet for Students","topic":"Motivation","icon":"🎯",
     "url":"https://www.smartgoalsguide.com","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R14","title":"Project Gutenberg — Free Study Materials","topic":"General","icon":"📚",
     "url":"https://www.gutenberg.org","motivation":0,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R15","title":"YouTube EDU — Free Educational Library","topic":"General","icon":"▶️",
     "url":"https://www.youtube.com/education","motivation":0,"hours_studied":0,"sleep":0,"resources_access":1},
    {"id":"R16","title":"OpenStax — Free Peer-Reviewed Textbooks","topic":"Science","icon":"📖",
     "url":"https://openstax.org","motivation":0,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R17","title":"edX — University-Level Online Courses","topic":"General","icon":"🏛️",
     "url":"https://www.edx.org","motivation":1,"hours_studied":1,"sleep":1,"resources_access":1},
    {"id":"R18","title":"Brilliant.org — Problem-Solving & Critical Thinking","topic":"Math","icon":"⚡",
     "url":"https://brilliant.org","motivation":1,"hours_studied":1,"sleep":1,"resources_access":1},
    {"id":"R19","title":"Duolingo — Language Learning (Cognitive Boost)","topic":"Cognitive","icon":"🦜",
     "url":"https://www.duolingo.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R20","title":"Notion — Student Study Planner Template","topic":"Productivity","icon":"📋",
     "url":"https://www.notion.so/templates/student-planner","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
]

RESOURCE_VECTORS = np.array(
    [[r["motivation"], r["hours_studied"], r["sleep"], r["resources_access"]] for r in RESOURCES],
    dtype=float,
)

# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def build_student_vector(raw, shap_vals, feature_names):
    shap_s = pd.Series(shap_vals, index=feature_names)
    needs_motivation = 1 if (raw.get("Motivation_Level", 2) == 0 or shap_s.get("Motivation_Level", 0) < -0.1) else 0
    needs_study      = 1 if (raw.get("Hours_Studied", 20) < 15   or shap_s.get("Hours_Studied", 0) < -0.5)   else 0
    needs_sleep      = 1 if (raw.get("Sleep_Hours", 7) < 6        or shap_s.get("Sleep_Hours", 0) < -0.2)     else 0
    needs_resources  = 1 if (raw.get("Access_to_Resources", 2) == 0 or shap_s.get("Access_to_Resources", 0) < -0.3) else 0
    return np.array([needs_motivation, needs_study, needs_sleep, needs_resources], dtype=float)

def get_top_recommendations(raw, shap_vals, feature_names, n=3):
    vec = build_student_vector(raw, shap_vals, feature_names).reshape(1, -1)
    sims = cosine_similarity(vec, RESOURCE_VECTORS)[0]
    top_idx = np.argsort(sims)[::-1][:n]
    return [RESOURCES[i] for i in top_idx]

def predict_student(model, scaler, explainer, feature_columns, student_raw):
    row = pd.DataFrame([student_raw])[feature_columns]
    row_scaled = scaler.transform(row)
    prob_fail = float(model.predict_proba(row_scaled)[0][0])
    prob_pass = 1 - prob_fail
    shap_vals = explainer.shap_values(row_scaled)[0]
    risk = "HIGH" if prob_fail >= 0.7 else ("MEDIUM" if prob_fail >= 0.4 else "LOW")
    recs = get_top_recommendations(student_raw, shap_vals, feature_columns)
    return prob_fail, prob_pass, shap_vals, risk, recs, row_scaled

def plot_shap_waterfall(shap_vals, feature_names, base_value, student_data):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:8]
    vals  = shap_vals[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]
    raw_vals = [list(student_data.values())[i] if i < len(student_data) else "" for i in sorted_idx]

    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors, alpha=0.85, height=0.6, edgecolor="none")

    pretty = {
        "Motivation_Level": "Motivation",
        "Peer_Influence": "Peer Influence",
        "Access_to_Resources": "Resources Access",
        "Parental_Involvement": "Parental Involvement",
        "Gender_Male": "Gender (Male)",
        "Extracurricular_Activities_Yes": "Extracurricular",
        "Learning_Disabilities_Yes": "Learning Disability",
        "Hours_Studied": "Hours Studied",
        "Attendance": "Attendance %",
        "Sleep_Hours": "Sleep Hours",
        "Previous_Scores": "Previous Score",
        "Tutoring_Sessions": "Tutoring Sessions",
        "Physical_Activity": "Physical Activity",
        "Stress_Proxy": "Stress Level",
    }
    ylabels = [pretty.get(n, n) for n in names]

    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(ylabels, color="#94a3b8", fontsize=9)
    ax.axvline(0, color="#2d3748", linewidth=1.2)
    ax.tick_params(axis="x", colors="#4a5568", labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xlabel("SHAP Value (impact on failure risk)", color="#4a5568", fontsize=8)
    ax.set_title("Feature Impact on Your Risk Score", color="#94a3b8", fontsize=10, fontweight="bold", pad=10)

    for i, (bar, val) in enumerate(zip(bars, vals)):
        label = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        ax.text(val + (0.005 if val >= 0 else -0.005), i,
                label, va="center", ha="left" if val >= 0 else "right",
                color="#ef4444" if val > 0 else "#60a5fa", fontsize=8, fontfamily="monospace")

    red_patch  = mpatches.Patch(color="#ef4444", alpha=0.85, label="↑ Increases failure risk")
    blue_patch = mpatches.Patch(color="#3b82f6", alpha=0.85, label="↓ Decreases failure risk")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right",
              facecolor="#111827", edgecolor="#1e2d40", labelcolor="#94a3b8", fontsize=8)

    plt.tight_layout()
    return fig

def plot_beeswarm(explainer, X_test_scaled, feature_columns):
    shap_vals = explainer.shap_values(X_test_scaled)
    pretty = {
        "Motivation_Level": "Motivation",
        "Peer_Influence": "Peer Influence",
        "Access_to_Resources": "Resources Access",
        "Parental_Involvement": "Parental Involvement",
        "Gender_Male": "Gender (Male)",
        "Extracurricular_Activities_Yes": "Extracurricular",
        "Learning_Disabilities_Yes": "Learning Disability",
        "Hours_Studied": "Hours Studied",
        "Attendance": "Attendance %",
        "Sleep_Hours": "Sleep Hours",
        "Previous_Scores": "Previous Score",
        "Tutoring_Sessions": "Tutoring Sessions",
        "Physical_Activity": "Physical Activity",
        "Stress_Proxy": "Stress Level",
    }
    pretty_names = [pretty.get(c, c) for c in feature_columns]

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:10]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    y_positions = list(range(len(order)))
    for yi, fi in zip(y_positions, order):
        sv = shap_vals[:, fi]
        fv = X_test_scaled[:, fi]
        norm_fv = (fv - fv.min()) / (fv.ptp() + 1e-8)
        colors = plt.cm.coolwarm(norm_fv)
        jitter = np.random.uniform(-0.18, 0.18, len(sv))
        ax.scatter(sv, yi + jitter, c=colors, alpha=0.5, s=8, linewidths=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([pretty_names[i] for i in order], color="#94a3b8", fontsize=9)
    ax.axvline(0, color="#2d3748", lw=1.5)
    ax.tick_params(axis="x", colors="#4a5568", labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xlabel("SHAP Value", color="#4a5568", fontsize=8)
    ax.set_title("Global Feature Importance (SHAP Beeswarm)", color="#94a3b8", fontsize=10, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────
# Load Everything
# ─────────────────────────────────────────
with st.spinner("Loading model & data…"):
    (model, scaler, explainer, feature_columns,
     X_train_scaled, X_test_scaled, X_test, y_test,
     metrics, teacher_df) = load_model_and_data()

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0; border-bottom: 1px solid #1e2d40; margin-bottom: 1.2rem;">
        <div style="font-size:1.3rem; font-weight:800; color:#f1f5f9; letter-spacing:-0.02em;">🎓 ASSE</div>
        <div style="color:#4a5568; font-size:0.78rem; margin-top:0.2rem;">Student Success Engine</div>
    </div>
    """, unsafe_allow_html=True)

    view = st.radio("Navigate", ["👨‍🏫  Teacher Dashboard", "🎓  Student Self-Assessment"])

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Model KPIs</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{metrics['accuracy']}%</div>
    </div>
    <div class="metric-card green">
        <div class="metric-label">Recall</div>
        <div class="metric-value">{metrics['recall']}%</div>
    </div>
    <div class="metric-card purple">
        <div class="metric-label">F1 Score</div>
        <div class="metric-value">{metrics['f1']}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#2d3748; font-size:0.72rem; text-align:center;'>Phase 4 · ASSE Project</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TEACHER VIEW
# ═══════════════════════════════════════════
if "Teacher" in view:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">👨‍🏫 Teacher Dashboard</div>
        <div class="app-subtitle">Class-level overview · At-risk detection · Global feature analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row
    high   = len(teacher_df[teacher_df["Risk Level"] == "HIGH"])
    medium = len(teacher_df[teacher_df["Risk Level"] == "MEDIUM"])
    low    = len(teacher_df[teacher_df["Risk Level"] == "LOW"])
    total  = len(teacher_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total}</div>
            <div class="metric-sub">in test set</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card red">
            <div class="metric-label">🔴 High Risk</div>
            <div class="metric-value">{high}</div>
            <div class="metric-sub">{round(high/total*100,1)}% of class</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">🟡 Medium Risk</div>
            <div class="metric-value">{medium}</div>
            <div class="metric-sub">{round(medium/total*100,1)}% of class</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card green">
            <div class="metric-label">🟢 Low Risk</div>
            <div class="metric-value">{low}</div>
            <div class="metric-sub">{round(low/total*100,1)}% of class</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Filters + Table
    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
    with col_f1:
        risk_filter = st.selectbox("Filter by Risk", ["All", "HIGH", "MEDIUM", "LOW"])
    with col_f2:
        att_min = st.slider("Min Attendance (%)", 0, 100, 0)

    filtered = teacher_df.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["Risk Level"] == risk_filter]
    filtered = filtered[filtered["Attendance (%)"] >= att_min]

    st.markdown("<div class='section-header'>At-Risk Student Table</div>", unsafe_allow_html=True)

    def color_risk(val):
        if val == "HIGH":   return "color: #f87171; font-weight: 700"
        if val == "MEDIUM": return "color: #fbbf24; font-weight: 700"
        return "color: #34d399; font-weight: 700"

    def color_prob(val):
        if val >= 70: return "color: #f87171"
        if val >= 40: return "color: #fbbf24"
        return "color: #34d399"

    styled = filtered.style\
        .applymap(color_risk, subset=["Risk Level"])\
        .applymap(color_prob, subset=["Fail Prob (%)"])\
        .set_properties(**{"background-color": "#0d1117", "color": "#94a3b8", "border": "1px solid #1e2d40"})\
        .set_table_styles([{"selector": "th", "props": [("background-color", "#111827"), ("color", "#6b7a99"), ("font-size", "0.78rem"), ("text-transform", "uppercase"), ("letter-spacing", "0.06em")]}])

    st.dataframe(styled, use_container_width=True, height=320)

    # ── SHAP Beeswarm
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Global Feature Importance — SHAP Beeswarm</div>", unsafe_allow_html=True)
    fig_bee = plot_beeswarm(explainer, X_test_scaled, feature_columns)
    st.pyplot(fig_bee, use_container_width=True)
    plt.close()

    # ── Intervention Badges
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Recommended Interventions</div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.info("🧑‍🏫 **Counselor Referral**\nFor students with HIGH risk + low motivation scores.")
    with b2:
        st.warning("👨‍👩‍👦 **Parent Outreach**\nFor students with poor attendance < 60%.")
    with b3:
        st.success("📚 **Learning Resources**\nAssign top-3 personalized resources per student.")


# ═══════════════════════════════════════════
# STUDENT VIEW
# ═══════════════════════════════════════════
else:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🎓 Student Self-Assessment</div>
        <div class="app-subtitle">Enter your details to get your predicted score, risk level, SHAP explanation & personalized recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("student_form"):
        st.markdown("<div class='section-header'>📋 Your Academic Profile</div>", unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            hours_studied   = st.slider("Hours Studied per Week", 0, 44, 15)
            attendance      = st.slider("Attendance (%)", 40, 100, 75)
            sleep_hours     = st.slider("Sleep Hours per Night", 4, 10, 7)
        with r1c2:
            previous_scores = st.slider("Previous Exam Score", 40, 100, 65)
            tutoring        = st.slider("Tutoring Sessions (per month)", 0, 8, 1)
            physical        = st.slider("Physical Activity (hrs/week)", 0, 6, 2)
        with r1c3:
            motivation_raw  = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
            resources_raw   = st.selectbox("Access to Resources", ["Low", "Medium", "High"])
            peer_raw        = st.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
            parental_raw    = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            gender_male      = st.selectbox("Gender", ["Female", "Male"])
        with r2c2:
            extracurricular  = st.selectbox("Extracurricular Activities", ["No", "Yes"])
        with r2c3:
            learning_dis     = st.selectbox("Learning Disability", ["No", "Yes"])

        submitted = st.form_submit_button("🔍 Analyze My Profile")

    if submitted:
        encode = {"Low": 0, "Medium": 1, "High": 2,
                  "Negative": 0, "Neutral": 1, "Positive": 2,
                  "No": 0, "Yes": 1,
                  "Female": 0, "Male": 1}

        stress_proxy = round(1 - (
            (encode[motivation_raw] / 2) * 0.4 +
            (min(hours_studied, 30) / 30) * 0.3 +
            (min(sleep_hours, 9) / 9) * 0.3
        ), 4)

        student_raw = {
            "Motivation_Level":              encode[motivation_raw],
            "Peer_Influence":                encode[peer_raw],
            "Access_to_Resources":           encode[resources_raw],
            "Parental_Involvement":          encode[parental_raw],
            "Gender_Male":                   encode[gender_male],
            "Extracurricular_Activities_Yes": encode[extracurricular],
            "Learning_Disabilities_Yes":     encode[learning_dis],
            "Hours_Studied":                 float(hours_studied),
            "Attendance":                    float(attendance),
            "Sleep_Hours":                   float(sleep_hours),
            "Previous_Scores":               float(previous_scores),
            "Tutoring_Sessions":             float(tutoring),
            "Physical_Activity":             float(physical),
            "Stress_Proxy":                  stress_proxy,
        }

        prob_fail, prob_pass, shap_vals, risk, recs, row_scaled = predict_student(
            model, scaler, explainer, feature_columns, student_raw
        )

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

        # ── Result Header
        risk_class = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}[risk]
        risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk]
        risk_msg   = {"HIGH": "You're at high risk of failing. Act now.",
                      "MEDIUM": "Moderate risk detected. Stay focused.",
                      "LOW": "You're on track! Keep it up."}[risk]

        st.markdown(f"""
        <div class="metric-card" style="border-top: 3px solid {'#ef4444' if risk=='HIGH' else '#f59e0b' if risk=='MEDIUM' else '#10b981'}; margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
                <div>
                    <div class="metric-label">Risk Assessment Result</div>
                    <div style="font-size:1.5rem; font-weight:800; color:#f1f5f9; margin: 0.3rem 0;">
                        {risk_emoji} {risk_msg}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="metric-label">Failure Probability</div>
                    <div class="metric-value" style="color:{'#ef4444' if risk=='HIGH' else '#fbbf24' if risk=='MEDIUM' else '#34d399'};">
                        {prob_fail*100:.1f}%
                    </div>
                    <div class="metric-sub">Pass probability: {prob_pass*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Two columns: Score ring + SHAP
        col_left, col_right = st.columns([1, 1.6])

        with col_left:
            st.markdown("<div class='section-header'>Your Score Breakdown</div>", unsafe_allow_html=True)

            # Gauge via matplotlib
            fig_gauge, ax_g = plt.subplots(figsize=(4.5, 3.2), subplot_kw=dict(aspect="equal"))
            fig_gauge.patch.set_facecolor("#0d1117")
            ax_g.set_facecolor("#0d1117")

            val = prob_fail
            colors_g = ["#ef4444" if val > 0.7 else "#f59e0b" if val > 0.4 else "#10b981", "#1a2235"]
            wedges = [val, 1 - val]
            ax_g.pie(wedges, startangle=90, colors=colors_g,
                     wedgeprops=dict(width=0.35, edgecolor="#0d1117", linewidth=3))
            color_text = "#ef4444" if val > 0.7 else "#fbbf24" if val > 0.4 else "#34d399"
            ax_g.text(0, 0.08, f"{val*100:.1f}%", ha="center", va="center",
                      fontsize=22, fontweight="800", color=color_text, fontfamily="monospace")
            ax_g.text(0, -0.22, "Failure Risk", ha="center", va="center",
                      fontsize=9, color="#6b7a99")
            ax_g.text(0, -0.42, f"Risk Level: {risk}", ha="center", va="center",
                      fontsize=9, fontweight="700", color=color_text)
            plt.tight_layout(pad=0)
            st.pyplot(fig_gauge)
            plt.close()

            # Score cards
            st.markdown(f"""
            <div class="metric-card green" style="margin-top:0.8rem;">
                <div class="metric-label">Pass Probability</div>
                <div class="metric-value">{prob_pass*100:.1f}%</div>
            </div>
            <div class="metric-card blue">
                <div class="metric-label">Previous Score</div>
                <div class="metric-value">{previous_scores}</div>
                <div class="metric-sub">/ 100 points</div>
            </div>
            <div class="metric-card {'red' if attendance < 60 else 'green'}">
                <div class="metric-label">Attendance</div>
                <div class="metric-value">{attendance}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='section-header'>Why Are You Flagged? — SHAP Explanation</div>", unsafe_allow_html=True)
            fig_shap = plot_shap_waterfall(shap_vals, feature_columns, 0, student_raw)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()

        # ── Recommendations
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎯 Top-3 Personalized Learning Recommendations</div>", unsafe_allow_html=True)

        r_cols = st.columns(3)
        topic_colors = {
            "Math": "#3b82f6", "Study Skills": "#8b5cf6", "Productivity": "#f59e0b",
            "Memory": "#10b981", "Science": "#06b6d4", "General": "#6b7a99",
            "Wellness": "#34d399", "Motivation": "#f87171", "Cognitive": "#a78bfa",
        }
        for i, (col, rec) in enumerate(zip(r_cols, recs)):
            tc = topic_colors.get(rec["topic"], "#4a5568")
            rank_emoji = ["🥇", "🥈", "🥉"][i]
            with col:
                st.markdown(f"""
                <div style="background:#111827; border:1px solid #1e2d40; border-radius:14px; padding:1.2rem; height:200px; display:flex; flex-direction:column; justify-content:space-between;">
                    <div>
                        <div style="font-size:1.5rem; margin-bottom:0.4rem;">{rec['icon']} {rank_emoji}</div>
                        <div style="color:#e2e8f0; font-weight:600; font-size:0.88rem; line-height:1.4;">{rec['title']}</div>
                    </div>
                    <div>
                        <span style="background:{tc}22; color:{tc}; border:1px solid {tc}44; border-radius:20px; padding:2px 10px; font-size:0.72rem; font-weight:600;">{rec['topic']}</span>
                        <div style="margin-top:0.6rem;">
                            <a href="{rec['url']}" target="_blank" style="color:#60a5fa; font-size:0.78rem; text-decoration:none;">🔗 Visit Resource →</a>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Action plan
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📌 Personalized Action Plan</div>", unsafe_allow_html=True)
        ac1, ac2, ac3 = st.columns(3)

        top_neg_idx = np.argsort(shap_vals)[:3]
        top_neg_feat = [feature_columns[i] for i in top_neg_idx]

        pretty_map = {
            "Motivation_Level": "Boost your motivation — try TED-Ed talks daily",
            "Hours_Studied": f"Increase study time — you study {hours_studied}h/week, aim for 25+",
            "Sleep_Hours": f"Improve sleep — you get {sleep_hours}h/night, aim for 8h",
            "Attendance": f"Improve attendance — currently at {attendance}%, target 85%+",
            "Previous_Scores": "Focus on past weak topics to improve your baseline score",
            "Access_to_Resources": "Use free resources: Khan Academy, OpenStax, YouTube EDU",
            "Stress_Proxy": "Manage stress — try Headspace or Calm mindfulness apps",
            "Tutoring_Sessions": "Increase tutoring sessions — even 1 extra/week helps",
            "Physical_Activity": "Add more exercise — improves cognition and memory",
            "Peer_Influence": "Seek positive study groups and mentors",
        }
        tips = [pretty_map.get(f, f"Focus on: {f}") for f in top_neg_feat]
        for col, tip in zip([ac1, ac2, ac3], tips):
            with col:
                st.info(f"💡 {tip}")
