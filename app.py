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
# Custom CSS — Premium Redesign
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg:        #0f0f13;
    --bg2:       #16161d;
    --bg3:       #1c1c26;
    --border:    #2a2a3a;
    --border2:   #35354a;
    --text:      #f0eee8;
    --text2:     #9896a4;
    --text3:     #5a5868;
    --accent:    #c8a96e;
    --accent2:   #e8c98e;
    --red:       #e05c5c;
    --amber:     #e0a43c;
    --green:     #4caf7d;
    --blue:      #5b8ff9;
    --purple:    #9b7ee8;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}

.main { background: var(--bg) !important; }
.block-container { padding-top: 1.5rem !important; }

/* ── Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text2) !important; }
section[data-testid="stSidebar"] .stRadio label { font-size: 0.9rem !important; }

/* ── App Header */
.app-header {
    padding: 1.8rem 0 1.2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.app-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    font-weight: 400;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1.1;
}
.app-title span { color: var(--accent); font-style: italic; }
.app-subtitle {
    color: var(--text3);
    font-size: 0.85rem;
    margin-top: 0.35rem;
    font-weight: 400;
}

/* ── Section header */
.section-header {
    font-family: 'DM Sans', sans-serif;
    color: var(--text3);
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}

/* ── Metric Cards */
.metric-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 18px 18px;
}
.metric-card.blue::after  { background: linear-gradient(90deg, var(--blue), transparent); }
.metric-card.green::after { background: linear-gradient(90deg, var(--green), transparent); }
.metric-card.red::after   { background: linear-gradient(90deg, var(--red), transparent); }
.metric-card.purple::after{ background: linear-gradient(90deg, var(--purple), transparent); }
.metric-card.amber::after { background: linear-gradient(90deg, var(--amber), transparent); }
.metric-card.gold::after  { background: linear-gradient(90deg, var(--accent), transparent); }

.metric-label {
    color: var(--text3);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.4rem;
}
.metric-value {
    color: var(--text);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    font-family: 'DM Mono', monospace;
}
.metric-sub {
    color: var(--text3);
    font-size: 0.76rem;
    margin-top: 0.25rem;
}

/* ── Risk Badges */
.risk-high {
    background: rgba(224,92,92,0.12);
    color: var(--red);
    border: 1px solid rgba(224,92,92,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-block;
}
.risk-medium {
    background: rgba(224,164,60,0.12);
    color: var(--amber);
    border: 1px solid rgba(224,164,60,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-block;
}
.risk-low {
    background: rgba(76,175,125,0.12);
    color: var(--green);
    border: 1px solid rgba(76,175,125,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-block;
}

/* ── Teacher — Plain Language Cards */
.student-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 1.2rem;
    transition: border-color 0.2s;
}
.student-card:hover { border-color: var(--border2); }
.student-card-id {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--text3);
    min-width: 44px;
}
.student-card-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text);
    flex: 1;
}
.student-card-meta {
    font-size: 0.78rem;
    color: var(--text2);
    margin-top: 0.12rem;
}
.mini-bar-wrap {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.mini-bar-label {
    font-size: 0.68rem;
    color: var(--text3);
    display: flex;
    justify-content: space-between;
}
.mini-bar-bg {
    height: 5px;
    background: var(--bg3);
    border-radius: 10px;
    overflow: hidden;
}
.mini-bar-fill {
    height: 5px;
    border-radius: 10px;
}

/* ── Insight cards for teacher */
.insight-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem;
    height: 100%;
}
.insight-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.insight-title {
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--text);
    margin-bottom: 0.4rem;
}
.insight-body {
    font-size: 0.82rem;
    color: var(--text2);
    line-height: 1.55;
}
.insight-action {
    margin-top: 0.8rem;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    display: inline-block;
}

/* ── Recommendation cards */
.rec-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.2rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* ── Form styling */
.stSlider label        { color: var(--text2) !important; font-size: 0.85rem !important; }
.stSelectbox label     { color: var(--text2) !important; font-size: 0.85rem !important; }
.stNumberInput label   { color: var(--text2) !important; font-size: 0.85rem !important; }
.stSlider div[data-baseweb="slider"] div { background: var(--accent) !important; }

/* ── Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #a07840 100%);
    color: #0f0f13;
    border: none;
    border-radius: 12px;
    padding: 0.65rem 2rem;
    font-weight: 700;
    font-size: 0.9rem;
    font-family: 'DM Sans', sans-serif;
    width: 100%;
    transition: all 0.2s;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 28px rgba(200,169,110,0.3);
}

/* ── Dataframe */
.dataframe { background: var(--bg2) !important; }

/* ── Selectbox / inputs dark */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
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
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#0f0f13")

    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:8]
    vals  = shap_vals[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]
    raw_vals = [list(student_data.values())[i] if i < len(student_data) else "" for i in sorted_idx]

    colors = ["#e05c5c" if v > 0 else "#5b8ff9" for v in vals]
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
    ax.set_yticklabels(ylabels, color="#9896a4", fontsize=9)
    ax.axvline(0, color="#2a2a3a", linewidth=1.2)
    ax.tick_params(axis="x", colors="#4a5568", labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xlabel("SHAP Value (impact on failure risk)", color="#5a5868", fontsize=8)
    ax.set_title("What's Helping & Hurting Your Result", color="#9896a4", fontsize=10, fontweight="bold", pad=10)

    for i, (bar, val) in enumerate(zip(bars, vals)):
        label = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        ax.text(val + (0.005 if val >= 0 else -0.005), i,
                label, va="center", ha="left" if val >= 0 else "right",
                color="#e05c5c" if val > 0 else "#5b8ff9", fontsize=8, fontfamily="monospace")

    red_patch  = mpatches.Patch(color="#e05c5c", alpha=0.85, label="↑ Increases failure risk")
    blue_patch = mpatches.Patch(color="#5b8ff9", alpha=0.85, label="↓ Decreases failure risk")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right",
              facecolor="#16161d", edgecolor="#2a2a3a", labelcolor="#9896a4", fontsize=8)

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
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#0f0f13")

    y_positions = list(range(len(order)))
    for yi, fi in zip(y_positions, order):
        sv = shap_vals[:, fi]
        fv = X_test_scaled[:, fi]
        norm_fv = (fv - fv.min()) / (fv.ptp() + 1e-8)
        colors = plt.cm.coolwarm(norm_fv)
        jitter = np.random.uniform(-0.18, 0.18, len(sv))
        ax.scatter(sv, yi + jitter, c=colors, alpha=0.5, s=8, linewidths=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([pretty_names[i] for i in order], color="#9896a4", fontsize=9)
    ax.axvline(0, color="#2a2a3a", lw=1.5)
    ax.tick_params(axis="x", colors="#4a5568", labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xlabel("SHAP Value", color="#5a5868", fontsize=8)
    ax.set_title("Global Feature Importance (SHAP Beeswarm)", color="#9896a4", fontsize=10, fontweight="bold", pad=10)
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
    <div style="padding: 1.2rem 0 1.6rem 0; border-bottom: 1px solid #2a2a3a; margin-bottom: 1.4rem;">
        <div style="font-family:'DM Serif Display',serif; font-size:1.5rem; font-weight:400; color:#f0eee8; letter-spacing:-0.01em;">
            <span style="color:#c8a96e; font-style:italic;">ASSE</span>
        </div>
        <div style="color:#5a5868; font-size:0.75rem; margin-top:0.25rem; letter-spacing:0.05em; text-transform:uppercase;">Student Success Engine</div>
    </div>
    """, unsafe_allow_html=True)

    view = st.radio("Navigate", ["👨‍🏫  Teacher Dashboard", "🎓  Student Self-Assessment"])

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>System Performance</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card blue">
        <div class="metric-label">Accuracy</div>
        <div class="metric-value">{metrics['accuracy']}%</div>
        <div class="metric-sub">Overall prediction accuracy</div>
    </div>
    <div class="metric-card green">
        <div class="metric-label">At-Risk Detection</div>
        <div class="metric-value">{metrics['recall']}%</div>
        <div class="metric-sub">Students caught before failing</div>
    </div>
    <div class="metric-card gold">
        <div class="metric-label">Balance Score</div>
        <div class="metric-value">{metrics['f1']}%</div>
        <div class="metric-sub">Precision vs recall balance</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#35354a; font-size:0.7rem; text-align:center; letter-spacing:0.05em;'>ASSE · Phase 4</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TEACHER VIEW — Plain Language, Premium Design
# ═══════════════════════════════════════════
if "Teacher" in view:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Good morning, <span>Teacher</span> 👋</div>
        <div class="app-subtitle">Here's a snapshot of how your class is doing — no technical knowledge needed.</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row — plain language
    high   = len(teacher_df[teacher_df["Risk Level"] == "HIGH"])
    medium = len(teacher_df[teacher_df["Risk Level"] == "MEDIUM"])
    low    = len(teacher_df[teacher_df["Risk Level"] == "LOW"])
    total  = len(teacher_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card blue">
            <div class="metric-label">👥 Your Class</div>
            <div class="metric-value">{total}</div>
            <div class="metric-sub">total students tracked</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card red">
            <div class="metric-label">🚨 Need Urgent Help</div>
            <div class="metric-value">{high}</div>
            <div class="metric-sub">{round(high/total*100,1)}% — at serious risk of failing</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card amber">
            <div class="metric-label">⚠️ Worth Watching</div>
            <div class="metric-value">{medium}</div>
            <div class="metric-sub">{round(medium/total*100,1)}% — may struggle soon</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card green">
            <div class="metric-label">✅ Doing Well</div>
            <div class="metric-value">{low}</div>
            <div class="metric-sub">{round(low/total*100,1)}% — on track</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── What's driving failure — plain language insight cards
    st.markdown("<div class='section-header'>📊 What's Affecting Your Class Most</div>", unsafe_allow_html=True)

    # Compute top factors from SHAP without exposing it
    shap_vals_all = explainer.shap_values(X_test_scaled)
    mean_impact   = np.abs(shap_vals_all).mean(axis=0)
    top3_idx      = np.argsort(mean_impact)[::-1][:3]
    factor_labels = {
        "Motivation_Level":               ("🔥 Low Motivation",          "Many students show low drive or engagement. Small wins and encouragement can make a big difference."),
        "Peer_Influence":                 ("👥 Peer Influence",           "The social circle matters. Students with negative peer groups are more likely to disengage."),
        "Access_to_Resources":            ("📚 Limited Resources",        "Some students don't have access to books, internet or study materials outside school."),
        "Parental_Involvement":           ("🏠 Parental Involvement",     "Students with lower parental support at home tend to struggle more academically."),
        "Gender_Male":                    ("⚖️ Gender Gap",              "There's a noticeable difference in outcomes between male and female students in this group."),
        "Extracurricular_Activities_Yes": ("🎭 Extracurricular Balance",  "Students in activities tend to do better — it builds routine and social connection."),
        "Learning_Disabilities_Yes":      ("🧩 Learning Differences",     "Students with learning disabilities need tailored support — consider extra check-ins."),
        "Hours_Studied":                  ("⏱️ Study Time",               "Students who study fewer hours per week are significantly more likely to underperform."),
        "Attendance":                     ("📅 Attendance",               "Missing classes is one of the strongest warning signs. Even a few absences add up quickly."),
        "Sleep_Hours":                    ("😴 Sleep Quality",            "Students who sleep less than 7 hours are noticeably less able to retain and apply knowledge."),
        "Previous_Scores":                ("📝 Past Performance",         "Students who struggled before are at elevated risk again — early follow-up helps."),
        "Tutoring_Sessions":              ("🎓 Tutoring Access",          "Students getting extra tutoring sessions show markedly better outcomes."),
        "Physical_Activity":              ("🏃 Physical Activity",        "Regular exercise is linked to better focus and academic resilience."),
        "Stress_Proxy":                   ("😰 Student Stress",           "High stress levels are quietly dragging down performance in a significant portion of students."),
    }
    action_map = {
        "Motivation_Level":               ("💬 Start a 5-min check-in conversation with disengaged students.", "#c8a96e22", "#c8a96e"),
        "Hours_Studied":                  ("📋 Share a simple weekly study plan template with the class.", "#5b8ff922", "#5b8ff9"),
        "Attendance":                     ("📞 Reach out to parents of students with 3+ absences this month.", "#e05c5c22", "#e05c5c"),
        "Sleep_Hours":                    ("🌙 Remind students about healthy sleep at the start of next class.", "#9b7ee822", "#9b7ee8"),
        "Previous_Scores":                ("📝 Schedule a brief one-on-one review with lower-scoring students.", "#4caf7d22", "#4caf7d"),
        "Peer_Influence":                 ("🔄 Try rearranging study groups to mix influences.", "#5b8ff922", "#5b8ff9"),
        "Access_to_Resources":            ("🔗 Share a list of free online resources with the class.", "#4caf7d22", "#4caf7d"),
        "Parental_Involvement":           ("✉️ Send a brief update email to parents of at-risk students.", "#c8a96e22", "#c8a96e"),
        "Tutoring_Sessions":              ("🎓 Refer struggling students to available tutoring support.", "#9b7ee822", "#9b7ee8"),
        "Stress_Proxy":                   ("🧘 Consider a 2-minute mindfulness break at the start of class.", "#4caf7d22", "#4caf7d"),
        "Physical_Activity":              ("🏃 Encourage short movement breaks during long sessions.", "#e0a43c22", "#e0a43c"),
        "Extracurricular_Activities_Yes": ("🎭 Support participation in at least one extracurricular activity.", "#5b8ff922", "#5b8ff9"),
        "Learning_Disabilities_Yes":      ("🧩 Coordinate with school counselor for personalised support plans.", "#e05c5c22", "#e05c5c"),
        "Gender_Male":                    ("⚖️ Review if any gender-specific barriers exist in your class.", "#9b7ee822", "#9b7ee8"),
    }

    ins_cols = st.columns(3)
    for col, fi in zip(ins_cols, top3_idx):
        fname = feature_columns[fi]
        icon_title, body = factor_labels.get(fname, (fname, "This factor significantly affects student outcomes."))
        action_tip, bg, border = action_map.get(fname, ("Follow up with affected students.", "#c8a96e22", "#c8a96e"))
        with col:
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">{icon_title}</div>
                <div class="insight-body">{body}</div>
                <div class="insight-action" style="background:{bg}; color:{border}; border:1px solid {border}44; margin-top:0.8rem;">
                    👉 {action_tip}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Student List — plain language, visual bars
    col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
    with col_f1:
        risk_filter = st.selectbox("Show students", ["All students", "Urgent (HIGH)", "Watch (MEDIUM)", "Doing well (LOW)"])
    with col_f2:
        att_min = st.slider("Minimum attendance (%)", 0, 100, 0)

    risk_map = {"All students": "All", "Urgent (HIGH)": "HIGH", "Watch (MEDIUM)": "MEDIUM", "Doing well (LOW)": "LOW"}
    filtered = teacher_df.copy()
    chosen_risk = risk_map[risk_filter]
    if chosen_risk != "All":
        filtered = filtered[filtered["Risk Level"] == chosen_risk]
    filtered = filtered[filtered["Attendance (%)"] >= att_min]

    st.markdown("<div class='section-header'>🧑‍🎓 Student Overview</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:var(--text3);font-size:0.8rem;margin-bottom:0.8rem;'>Showing {len(filtered)} of {total} students</div>", unsafe_allow_html=True)

    risk_class_map = {"HIGH": "risk-high", "MEDIUM": "risk-medium", "LOW": "risk-low"}
    risk_emoji_map = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}
    risk_label_map = {"HIGH": "Needs urgent help", "MEDIUM": "Worth watching", "LOW": "Doing well"}

    for _, row in filtered.iterrows():
        rl     = row["Risk Level"]
        att    = row["Attendance (%)"]
        hrs    = row["Hours Studied"]
        score  = row["Previous Score"]
        prob   = row["Fail Prob (%)"]
        att_color  = "#4caf7d" if att >= 75 else "#e0a43c" if att >= 60 else "#e05c5c"
        hrs_color  = "#4caf7d" if hrs >= 20 else "#e0a43c" if hrs >= 12 else "#e05c5c"
        prob_color = "#e05c5c" if prob >= 70 else "#e0a43c" if prob >= 40 else "#4caf7d"

        st.markdown(f"""
        <div class="student-card">
            <div class="student-card-id">{row['Student ID']}</div>
            <div style="flex:1;">
                <div class="student-card-name">{risk_emoji_map[rl]} {risk_label_map[rl]}</div>
                <div class="student-card-meta">Failure risk: <span style="color:{prob_color};font-weight:700;">{prob}%</span></div>
            </div>
            <div class="mini-bar-wrap" style="min-width:160px;">
                <div class="mini-bar-label"><span>Attendance</span><span style="color:{att_color};">{att}%</span></div>
                <div class="mini-bar-bg"><div class="mini-bar-fill" style="width:{att}%;background:{att_color};"></div></div>
                <div class="mini-bar-label" style="margin-top:4px;"><span>Study hours/week</span><span style="color:{hrs_color};">{hrs}h</span></div>
                <div class="mini-bar-bg"><div class="mini-bar-fill" style="width:{min(hrs/44*100,100):.0f}%;background:{hrs_color};"></div></div>
            </div>
            <div style="min-width:80px;text-align:right;">
                <span class="{risk_class_map[rl]}">{rl}</span>
                <div style="font-size:0.72rem;color:var(--text3);margin-top:0.4rem;">Prev. score: {score}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Suggested Actions
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💡 Suggested Next Steps</div>", unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("""
        <div class="insight-card" style="border-left:3px solid #e05c5c;">
            <div class="insight-icon">🧑‍💼</div>
            <div class="insight-title">Refer to a Counsellor</div>
            <div class="insight-body">Students marked <strong>Urgent</strong> with signs of low motivation should be connected with your school counsellor for a chat.</div>
        </div>
        """, unsafe_allow_html=True)
    with b2:
        st.markdown("""
        <div class="insight-card" style="border-left:3px solid #e0a43c;">
            <div class="insight-icon">📞</div>
            <div class="insight-title">Contact Parents</div>
            <div class="insight-body">For students with attendance below 60%, a brief call or message to parents can help turn things around quickly.</div>
        </div>
        """, unsafe_allow_html=True)
    with b3:
        st.markdown("""
        <div class="insight-card" style="border-left:3px solid #4caf7d;">
            <div class="insight-icon">📚</div>
            <div class="insight-title">Share Learning Resources</div>
            <div class="insight-body">Each student gets a personalised list of free resources based on their profile. Share these from the Student view to give extra support.</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════
# STUDENT VIEW
# ═══════════════════════════════════════════
else:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Your <span>Academic</span> Health Check 🎓</div>
        <div class="app-subtitle">Fill in your details below — we'll predict your risk level and give you personalised tips to improve.</div>
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
        <div class="metric-card" style="border-top: 3px solid {'#e05c5c' if risk=='HIGH' else '#e0a43c' if risk=='MEDIUM' else '#4caf7d'}; margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
                <div>
                    <div class="metric-label">Your Result</div>
                    <div style="font-family:'DM Serif Display',serif; font-size:1.5rem; font-weight:400; color:var(--text); margin: 0.3rem 0;">
                        {risk_emoji} {risk_msg}
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="metric-label">Chance of Not Passing</div>
                    <div class="metric-value" style="color:{'#e05c5c' if risk=='HIGH' else '#e0a43c' if risk=='MEDIUM' else '#4caf7d'};">
                        {prob_fail*100:.1f}%
                    </div>
                    <div class="metric-sub">Chance of passing: {prob_pass*100:.1f}%</div>
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
            fig_gauge.patch.set_facecolor("#0f0f13")
            ax_g.set_facecolor("#0f0f13")

            val = prob_fail
            colors_g = ["#e05c5c" if val > 0.7 else "#e0a43c" if val > 0.4 else "#4caf7d", "#1c1c26"]
            wedges = [val, 1 - val]
            ax_g.pie(wedges, startangle=90, colors=colors_g,
                     wedgeprops=dict(width=0.35, edgecolor="#0f0f13", linewidth=3))
            color_text = "#e05c5c" if val > 0.7 else "#e0a43c" if val > 0.4 else "#4caf7d"
            ax_g.text(0, 0.08, f"{val*100:.1f}%", ha="center", va="center",
                      fontsize=22, fontweight="800", color=color_text, fontfamily="monospace")
            ax_g.text(0, -0.22, "Failure Risk", ha="center", va="center",
                      fontsize=9, color="#5a5868")
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
            st.markdown("<div class='section-header'>What's Helping & Hurting Your Score</div>", unsafe_allow_html=True)
            fig_shap = plot_shap_waterfall(shap_vals, feature_columns, 0, student_raw)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()

        # ── Recommendations
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>🎯 Top-3 Personalized Learning Recommendations</div>", unsafe_allow_html=True)

        r_cols = st.columns(3)
        topic_colors = {
            "Math": "#5b8ff9", "Study Skills": "#9b7ee8", "Productivity": "#e0a43c",
            "Memory": "#4caf7d", "Science": "#06b6d4", "General": "#5a5868",
            "Wellness": "#4caf7d", "Motivation": "#e05c5c", "Cognitive": "#9b7ee8",
        }
        rank_labels = ["Top Pick", "2nd Pick", "3rd Pick"]
        for i, (col, rec) in enumerate(zip(r_cols, recs)):
            tc = topic_colors.get(rec["topic"], "#5a5868")
            with col:
                st.markdown(f"""
                <div class="rec-card">
                    <div>
                        <div style="font-size:1.6rem; margin-bottom:0.5rem;">{rec['icon']}</div>
                        <div style="color:var(--text); font-weight:600; font-size:0.88rem; line-height:1.45; margin-bottom:0.7rem;">{rec['title']}</div>
                    </div>
                    <div>
                        <span style="background:{tc}20; color:{tc}; border:1px solid {tc}40; border-radius:20px; padding:2px 10px; font-size:0.7rem; font-weight:600;">{rec['topic']}</span>
                        <div style="margin-top:0.7rem; font-size:0.7rem; color:var(--text3);">{rank_labels[i]}</div>
                        <a href="{rec['url']}" target="_blank" style="color:var(--accent); font-size:0.8rem; text-decoration:none; font-weight:500;">Open resource →</a>
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
