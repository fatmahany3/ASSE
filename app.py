import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches mpatches
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
    # هنا تم افتراض وجود الملفات لتجنب الـ Errors عند التشغيل التجريبي
    try:
        train_df = pd.read_csv("train_processed.csv")
        test_df  = pd.read_csv("test_processed.csv")
    except FileNotFoundError:
        # بيانات وهمية في حال لم تكن الملفات مرفوعة بعد لتجنب انهيار التطبيق
        cols = ["Attendance", "Hours_Studied", "Previous_Scores", "Motivation_Level", "Sleep_Hours", "Access_to_Resources", "Target"]
        train_df = pd.DataFrame(np.random.randint(10, 95, size=(100, 7)), columns=cols)
        train_df["Target"] = np.random.choice([0, 1], size=100)
        test_df = pd.DataFrame(np.random.randint(10, 95, size=(30, 7)), columns=cols)
        test_df["Target"] = np.random.choice([0, 1], size=30)

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
    teacher_rows  = []
    for i in range(len(X_test)):
        raw   = X_test.iloc[i].to_dict()
        proba = float(model.predict_proba(X_test_scaled[i:i+1])[0][1]) # التعديل هنا لـ Class 1 (Failure)
        rl    = "HIGH" if proba >= 0.7 else ("MEDIUM" if proba >= 0.4 else "LOW")
        teacher_rows.append({
            "Student ID":    f"S{i+1:03d}",
            "Attendance (%)":     raw.get("Attendance", 85),
            "Hours Studied":      raw.get("Hours_Studied", 15),
            "Previous Score":     raw.get("Previous_Scores", 75),
            "Fail Prob (%)":      round(proba * 100, 1),
            "Risk Level":          rl,
        })
    teacher_df = pd.DataFrame(teacher_rows)

    return (model, scaler, explainer, feature_columns,
            X_train_scaled, X_test_scaled, X_test, y_test,
            metrics, teacher_df)

# ─────────────────────────────────────────
# Resources Catalog
# ─────────────────────────────────────────
RESOURCES = [
    {"id":"R01","title":"Khan Academy — Mathematics Fundamentals","topic":"Math","icon":"📐", "url":"https://www.khanacademy.org/math","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R02","title":"Coursera — Learning How to Learn","topic":"Study Skills","icon":"🧠", "url":"https://www.coursera.org/learn/learning-how-to-learn","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R03","title":"Pomodoro Technique — Study Timer App","topic":"Productivity","icon":"⏱️", "url":"https://pomofocus.io","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R04","title":"Anki — Spaced Repetition Flashcards","topic":"Memory","icon":"🃏", "url":"https://apps.ankiweb.net","motivation":0,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R05","title":"MIT OpenCourseWare — Science & Engineering","topic":"Science","icon":"🔬", "url":"https://ocw.mit.edu","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R06","title":"Crash Course — Science & Humanities Videos","topic":"General","icon":"🎬", "url":"https://www.youtube.com/crashcourse","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R07","title":"Quizlet — Interactive Study Sets","topic":"Study Skills","icon":"✏️", "url":"https://quizlet.com","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R08","title":"Sleep Foundation — Healthy Sleep Guide","topic":"Wellness","icon":"😴", "url":"https://www.sleepfoundation.org/teens-and-sleep","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"id":"R09","title":"Headspace — Student Mindfulness & Stress Relief","topic":"Wellness","icon":"🧘", "url":"https://www.headspace.com/students","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"id":"R10","title":"Calm — Sleep & Relaxation for Students","topic":"Wellness","icon":"🌙", "url":"https://www.calm.com","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"id":"R11","title":"TED-Ed — Motivational Student Talks","topic":"Motivation","icon":"🎤", "url":"https://ed.ted.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R12","title":"Growth Mindset — Carol Dweck","topic":"Motivation","icon":"💡", "url":"https://www.mindsetonline.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R13","title":"SMART Goals Worksheet for Students","topic":"Motivation","icon":"🎯", "url":"https://www.smartgoalsguide.com","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"id":"R14","title":"Project Gutenberg — Free Study Materials","topic":"General","icon":"📚", "url":"https://www.gutenberg.org","motivation":0,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R15","title":"YouTube EDU — Free Educational Library","topic":"General","icon":"▶️", "url":"https://www.youtube.com/education","motivation":0,"hours_studied":0,"sleep":0,"resources_access":1},
    {"id":"R16","title":"OpenStax — Free Peer-Reviewed Textbooks","topic":"Science","icon":"📖", "url":"https://openstax.org","motivation":0,"hours_studied":1,"sleep":0,"resources_access":1},
    {"id":"R17","title":"edX — University-Level Online Courses","topic":"General","icon":"🏛️", "url":"https://www.edx.org","motivation":1,"hours_studied":1,"sleep":1,"resources_access":1},
    {"id":"R18","title":"Brilliant.org — Problem-Solving & Critical Thinking","topic":"Math","icon":"⚡", "url":"https://brilliant.org","motivation":1,"hours_studied":1,"sleep":1,"resources_access":1},
    {"id":"R19","title":"Duolingo — Language Learning (Cognitive Boost)","topic":"Cognitive","icon":"🦜", "url":"https://www.duolingo.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"id":"R20","title":"Notion — Student Study Planner Template","topic":"Productivity","icon":"📋", "url":"https://www.notion.so/templates/student-planner","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
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
    prob_fail = float(model.predict_proba(row_scaled)[0][1]) # تعديل الـ Index ليكون الـ Failure Class
    prob_pass = 1 - prob_fail
    shap_vals = explainer.shap_values(row_scaled)[0]
    risk = "HIGH" if prob_fail >= 0.7 else ("MEDIUM" if prob_fail >= 0.4 else "LOW")
    recs = get_top_recommendations(student_raw, shap_vals, feature_columns)
    return prob_fail, prob_pass, shap_vals, risk, recs, row_scaled

def plot_shap_waterfall(shap_vals, feature_names, student_data):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f0f13")
    ax.set_facecolor("#0f0f13")

    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:6]
    vals  = shap_vals[sorted_idx]
    names = [feature_names[i] for i in sorted_idx]

    colors = ["#e05c5c" if v > 0 else "#5b8ff9" for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors, alpha=0.85, height=0.5)

    pretty = {
        "Motivation_Level": "Motivation", "Access_to_Resources": "Resources Access",
        "Hours_Studied": "Hours Studied", "Attendance": "Attendance %",
        "Sleep_Hours": "Sleep Hours", "Previous_Scores": "Previous Score"
    }
    ylabels = [pretty.get(n, n) for n in names]

    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(ylabels, color="#9896a4", fontsize=9)
    ax.axvline(0, color="#2a2a3a", linewidth=1.2)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_title("Top Impact Factors (SHAP)", color="#9896a4", fontsize=10, fontweight="bold")
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
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# TEACHER VIEW
# ═══════════════════════════════════════════
if "Teacher" in view:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Good morning, <span>Teacher</span> 👋</div>
        <div class="app-subtitle">Here's a snapshot of how your class is doing — no technical knowledge needed.</div>
    </div>
    """, unsafe_allow_html=True)

    high   = len(teacher_df[teacher_df["Risk Level"] == "HIGH"])
    medium = len(teacher_df[teacher_df["Risk Level"] == "MEDIUM"])
    low    = len(teacher_df[teacher_df["Risk Level"] == "LOW"])
    total  = len(teacher_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card blue"><div class="metric-label">👥 Total Students</div><div class="metric-value">{total}</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card red"><div class="metric-label">🚨 Urgent Help</div><div class="metric-value">{high}</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card amber"><div class="metric-label">⚠️ Watch List</div><div class="metric-value">{medium}</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card green"><div class="metric-label">✅ Doing Well</div><div class="metric-value">{low}</div></div>', unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🧑‍🎓 Student Overview</div>", unsafe_allow_html=True)
    
    # فلترة سريعة
    risk_filter = st.selectbox("Filter by Risk", ["All", "HIGH", "MEDIUM", "LOW"])
    filtered = teacher_df.copy() if risk_filter == "All" else teacher_df[teacher_df["Risk Level"] == risk_filter]

    # عرض كروت الطلاب وإغلاق الـ HTML المكسور في كودك القديم
    for _, row in filtered.iterrows():
        rl = row["Risk Level"]
        badge_cls = "risk-high" if rl == "HIGH" else ("risk-medium" if rl == "MEDIUM" else "risk-low")
        
        st.markdown(f"""
        <div class="student-card">
            <div class="student-card-id">{row['Student ID']}</div>
            <div class="student-card-name">Student Analysis Overview
                <div class="student-card-meta">Attendance: {row['Attendance (%)']}% | Hours Studied: {row['Hours Studied']}h | Past Score: {row['Previous Score']}</div>
            </div>
            <div>
                <span class="{badge_cls}">{rl} RISK ({row['Fail Prob (%)']}%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════
# STUDENT VIEW
# ═══════════════════════════════════════════
else:
    st.markdown("""
    <div class="app-header">
        <div class="app-title">Empower Your <span>Journey</span> 🎓</div>
        <div class="app-subtitle">Fill out your daily metrics below to discover your personalized optimization roadmap.</div>
    </div>
    """, unsafe_allow_html=True)

    # نموذج إدخال بيانات الطالب تلقائياً بناءاً على الـ Feature Columns المتوفرة في الـ Model
    st.markdown("<div class='section-header'>📝 Enter Your Study & Lifestyle Metrics</div>", unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        att = st.slider("Your Attendance Rate (%)", 0, 100, 85)
        hrs = st.slider("Weekly Study Hours", 0, 50, 15)
        scores = st.slider("Last Exam Score", 0, 100, 75)
    with col_s2:
        sleep = st.slider("Average Sleep Hours", 4, 12, 7)
        motivation = st.selectbox("Motivation Level", ["Low", "Medium", "High"], index=1)
        resources = st.selectbox("Access to Study Materials", ["Limited", "Fully Available"], index=1)

    # تحويل الاختيارات النصية لأرقام متوافقة مع الـ Model الخاص بك
    mot_map = {"Low": 0, "Medium": 1, "High": 2}
    res_map = {"Limited": 0, "Fully Available": 1}

    # بناء القاموس المتوافق تماماً مع الـ Features الممررة للموديل
    student_raw = {}
    for col in feature_columns:
        if "Attendance" in col: student_raw[col] = att
        elif "Hours_Studied" in col: student_raw[col] = hrs
        elif "Previous_Scores" in col: student_raw[col] = scores
        elif "Sleep_Hours" in col: student_raw[col] = sleep
        elif "Motivation" in col: student_raw[col] = mot_map[motivation]
        elif "Resource" in col: student_raw[col] = res_map[resources]
        else: student_raw[col] = 0 # تعبئة باقي المدخلات كـ Zero-fill لتجنب الـ Shape mismatch

    if st.button("Analyze My Performance Blueprint"):
        prob_fail, prob_pass, shap_vals, risk, recs, _ = predict_student(
            model, scaler, explainer, feature_columns, student_raw
        )

        st.markdown("<div class='section-header'>🔮 Optimization Summary Blueprint</div>", unsafe_allow_html=True)
        
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            card_cls = "red" if risk == "HIGH" else ("amber" if risk == "MEDIUM" else "green")
            st.markdown(f"""
            <div class="metric-card {card_cls}">
                <div class="metric-label">Risk Assessment Profile</div>
                <div class="metric-value">{risk} RISK</div>
                <div class="metric-sub">Calculated probability of academic friction: {round(prob_fail*100, 1)}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # عرض الرسم البياني لتأثير العوامل (SHAP)
            fig = plot_shap_waterfall(shap_vals, feature_columns, student_raw)
            st.pyplot(fig)

        with c_res2:
            st.markdown("<div style='font-size:0.9rem; font-weight:700; margin-bottom:0.8rem;'>🎯 Core Recommendations Tailored for You:</div>", unsafe_allow_html=True)
            for r in recs:
                st.markdown(f"""
                <div class="rec-card" style="margin-bottom:0.6rem; border-left: 3px solid var(--accent);">
                    <div>
                        <span style="font-size:1.2rem; margin-right:0.5rem;">{r['icon']}</span>
                        <strong style="color:var(--text); font-size:0.85rem;">{r['title']}</strong>
                        <div style="color:var(--text2); font-size:0.75rem; margin-top:0.2rem;">Topic Classification: {r['topic']}</div>
                    </div>
                    <div style="margin-top:0.6rem; text-align:right;">
                        <a href="{r['url']}" target="_blank" style="color:var(--accent); text-decoration:none; font-size:0.8rem; font-weight:600;">Access Material ↗</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)
