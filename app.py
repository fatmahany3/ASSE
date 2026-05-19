import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

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
# GLOBAL CSS  — warm, light, friendly
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&family=Lora:ital,wght@0,500;0,600;1,500&display=swap');

:root {
    --cream:   #faf8f5;
    --white:   #ffffff;
    --border:  #ece8e1;
    --text:    #2d2926;
    --muted:   #8a8078;
    --green:   #2d9e6b;
    --green-l: #e8f7f1;
    --amber:   #d97706;
    --amber-l: #fef3c7;
    --red:     #dc2626;
    --red-l:   #fee2e2;
    --blue:    #2563eb;
    --blue-l:  #eff6ff;
    --purple:  #7c3aed;
    --purple-l:#f5f3ff;
    --teal:    #0d9488;
    --teal-l:  #f0fdfa;
    --shadow:  0 2px 12px rgba(0,0,0,0.07);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.10);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--cream) !important;
    font-family: 'Nunito', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }

[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.block-container { padding-top: 2rem !important; }

.kpi {
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    border: 1.5px solid var(--border);
    box-shadow: var(--shadow);
    text-align: center;
}
.kpi-icon  { font-size: 2rem; line-height: 1; margin-bottom: 0.4rem; }
.kpi-value { font-family: 'Lora', serif; font-size: 2rem; font-weight: 600; line-height: 1.1; }
.kpi-label { font-size: 0.8rem; font-weight: 700; text-transform: uppercase;
             letter-spacing: 0.07em; margin-top: 0.25rem; white-space: pre-line; }

.stSlider label, .stSelectbox label, .stNumberInput label {
    font-weight: 700 !important;
    color: var(--text) !important;
    font-size: 0.88rem !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2d9e6b, #0d9488) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2.5rem !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(45,158,107,0.35) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(45,158,107,0.45) !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--cream); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


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

    explainer = shap.LinearExplainer(model, X_train_scaled,
                                     feature_perturbation="interventional")

    preds = model.predict(X_test_scaled)
    acc   = round(accuracy_score(y_test, preds) * 100, 1)
    rec   = round(recall_score(y_test, preds)   * 100, 1)
    f1    = round(f1_score(y_test, preds)        * 100, 1)

    rows = []
    for i in range(len(X_test)):
        raw = X_test.iloc[i].to_dict()
        pf  = float(model.predict_proba(X_test_scaled[i:i+1])[0][0])
        rl  = "🔴 High" if pf >= 0.7 else ("🟡 Medium" if pf >= 0.4 else "🟢 Low")
        rows.append({
            "Student": f"Student {i+1:03d}",
            "Attendance": f"{raw['Attendance']:.0f}%",
            "Hours Studied / Week": f"{raw['Hours_Studied']:.0f}h",
            "Previous Score": f"{raw['Previous_Scores']:.0f}/100",
            "Chance of Failing": f"{pf*100:.1f}%",
            "Risk Level": rl,
        })
    teacher_df = pd.DataFrame(rows)

    return (model, scaler, explainer, feature_columns,
            X_train_scaled, X_test_scaled, X_test, y_test,
            acc, rec, f1, teacher_df)


(model, scaler, explainer, feature_columns,
 X_train_scaled, X_test_scaled, X_test, y_test,
 ACC, REC, F1, teacher_df) = load_everything()


# ──────────────────────────────────────────────
# RESOURCES
# ──────────────────────────────────────────────
RESOURCES = [
    {"title":"Khan Academy — Maths & Science",        "topic":"Study",       "icon":"📐","bg":"#eff6ff","ic":"#2563eb","url":"https://www.khanacademy.org","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"title":"Coursera — Learning How to Learn",      "topic":"Study Skills","icon":"🧠","bg":"#f5f3ff","ic":"#7c3aed","url":"https://www.coursera.org/learn/learning-how-to-learn","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"title":"Pomodoro Timer — Focus Technique",      "topic":"Productivity","icon":"⏱️","bg":"#fef3c7","ic":"#d97706","url":"https://pomofocus.io","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"title":"Anki — Smart Flashcard Memory App",     "topic":"Memory",      "icon":"🃏","bg":"#e8f7f1","ic":"#2d9e6b","url":"https://apps.ankiweb.net","motivation":0,"hours_studied":1,"sleep":0,"resources_access":0},
    {"title":"MIT OpenCourseWare — Free Lectures",    "topic":"Science",     "icon":"🔬","bg":"#eff6ff","ic":"#2563eb","url":"https://ocw.mit.edu","motivation":1,"hours_studied":1,"sleep":0,"resources_access":1},
    {"title":"Crash Course — Fun Educational Videos", "topic":"General",     "icon":"🎬","bg":"#fef3c7","ic":"#d97706","url":"https://www.youtube.com/crashcourse","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"title":"Quizlet — Practice with Flashcards",    "topic":"Study Skills","icon":"✏️","bg":"#f5f3ff","ic":"#7c3aed","url":"https://quizlet.com","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"title":"Sleep Foundation — Better Sleep Guide", "topic":"Wellness",    "icon":"😴","bg":"#f0fdfa","ic":"#0d9488","url":"https://www.sleepfoundation.org","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"title":"Headspace — Calm Your Mind",            "topic":"Wellness",    "icon":"🧘","bg":"#e8f7f1","ic":"#2d9e6b","url":"https://www.headspace.com/students","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"title":"Calm — Relax & Sleep Better",           "topic":"Wellness",    "icon":"🌙","bg":"#f0fdfa","ic":"#0d9488","url":"https://www.calm.com","motivation":0,"hours_studied":0,"sleep":1,"resources_access":0},
    {"title":"TED-Ed — Inspiring Student Talks",      "topic":"Motivation",  "icon":"🎤","bg":"#fee2e2","ic":"#dc2626","url":"https://ed.ted.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"title":"Growth Mindset — Carol Dweck",          "topic":"Motivation",  "icon":"💡","bg":"#fef3c7","ic":"#d97706","url":"https://www.mindsetonline.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"title":"SMART Goals Worksheet",                 "topic":"Motivation",  "icon":"🎯","bg":"#fee2e2","ic":"#dc2626","url":"https://www.smartgoalsguide.com","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
    {"title":"Project Gutenberg — Free Books",        "topic":"Reading",     "icon":"📚","bg":"#f5f3ff","ic":"#7c3aed","url":"https://www.gutenberg.org","motivation":0,"hours_studied":1,"sleep":0,"resources_access":1},
    {"title":"YouTube EDU — Free Video Lessons",      "topic":"General",     "icon":"▶️","bg":"#fee2e2","ic":"#dc2626","url":"https://www.youtube.com/education","motivation":0,"hours_studied":0,"sleep":0,"resources_access":1},
    {"title":"OpenStax — Free Textbooks",             "topic":"Science",     "icon":"📖","bg":"#eff6ff","ic":"#2563eb","url":"https://openstax.org","motivation":0,"hours_studied":1,"sleep":0,"resources_access":1},
    {"title":"edX — University Courses Online",       "topic":"General",     "icon":"🏛️","bg":"#f0fdfa","ic":"#0d9488","url":"https://www.edx.org","motivation":1,"hours_studied":1,"sleep":1,"resources_access":1},
    {"title":"Brilliant.org — Problem-Solving",       "topic":"Maths",       "icon":"⚡","bg":"#fef3c7","ic":"#d97706","url":"https://brilliant.org","motivation":1,"hours_studied":1,"sleep":1,"resources_access":1},
    {"title":"Duolingo — Language & Brain Training",  "topic":"Cognitive",   "icon":"🦜","bg":"#e8f7f1","ic":"#2d9e6b","url":"https://www.duolingo.com","motivation":1,"hours_studied":0,"sleep":0,"resources_access":0},
    {"title":"Notion — Student Planner Template",     "topic":"Productivity","icon":"📋","bg":"#f5f3ff","ic":"#7c3aed","url":"https://www.notion.so/templates/student-planner","motivation":1,"hours_studied":1,"sleep":0,"resources_access":0},
]
RES_VECTORS = np.array(
    [[r["motivation"], r["hours_studied"], r["sleep"], r["resources_access"]]
     for r in RESOURCES], dtype=float)

PRETTY = {
    "Motivation_Level":              "Motivation",
    "Peer_Influence":                "Peer Influence",
    "Access_to_Resources":           "Access to Resources",
    "Parental_Involvement":          "Parental Involvement",
    "Gender_Male":                   "Gender",
    "Extracurricular_Activities_Yes":"Extracurricular Activities",
    "Learning_Disabilities_Yes":     "Learning Disability",
    "Hours_Studied":                 "Hours Studied",
    "Attendance":                    "Attendance",
    "Sleep_Hours":                   "Sleep Hours",
    "Previous_Scores":               "Previous Score",
    "Tutoring_Sessions":             "Tutoring Sessions",
    "Physical_Activity":             "Physical Activity",
    "Stress_Proxy":                  "Stress Level",
}


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def student_vec(raw, shap_vals, feat):
    s  = pd.Series(shap_vals, index=feat)
    m  = 1 if (raw.get("Motivation_Level",2)==0    or s.get("Motivation_Level",0)<-0.1) else 0
    h  = 1 if (raw.get("Hours_Studied",20)<15       or s.get("Hours_Studied",0)<-0.5)   else 0
    sl = 1 if (raw.get("Sleep_Hours",7)<6            or s.get("Sleep_Hours",0)<-0.2)     else 0
    r  = 1 if (raw.get("Access_to_Resources",2)==0   or s.get("Access_to_Resources",0)<-0.3) else 0
    return np.array([m,h,sl,r], dtype=float)

def get_recs(raw, shap_vals, feat, n=3):
    v    = student_vec(raw, shap_vals, feat).reshape(1,-1)
    sims = cosine_similarity(v, RES_VECTORS)[0]
    idx  = np.argsort(sims)[::-1][:n]
    return [RESOURCES[i] for i in idx]

def run_prediction(raw):
    row       = pd.DataFrame([raw])[feature_columns]
    scaled    = scaler.transform(row)
    prob_fail = float(model.predict_proba(scaled)[0][0])
    shap_vals = explainer.shap_values(scaled)[0]
    risk      = "HIGH" if prob_fail>=0.7 else ("MEDIUM" if prob_fail>=0.4 else "LOW")
    recs      = get_recs(raw, shap_vals, feature_columns)
    return prob_fail, shap_vals, risk, recs


# ──────────────────────────────────────────────
# CHARTS
# ──────────────────────────────────────────────
def fig_donut(prob_fail, risk):
    cmap = {"HIGH":"#dc2626","MEDIUM":"#d97706","LOW":"#2d9e6b"}
    color = cmap[risk]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.pie([prob_fail, 1-prob_fail], colors=[color,"#f0ede8"],
           startangle=90, counterclock=False,
           wedgeprops=dict(width=0.32, edgecolor="white", linewidth=3))
    ax.text(0,  0.1, f"{prob_fail*100:.0f}%",  ha="center", va="center",
            fontsize=28, fontweight="800", color=color)
    ax.text(0, -0.22, "failure risk",           ha="center", va="center",
            fontsize=10, color="#8a8078")
    plt.tight_layout(pad=0)
    return fig

def fig_shap_simple(shap_vals, feature_names):
    order = np.argsort(np.abs(shap_vals))[::-1][:8]
    vals  = shap_vals[order]
    names = [PRETTY.get(feature_names[i], feature_names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_alpha(0); ax.set_facecolor("none")

    colors = ["#dc2626" if v>0 else "#2d9e6b" for v in vals]
    alphas = [min(1.0, 0.55 + abs(v)*2) for v in vals]
    for i,(v,c,a) in enumerate(zip(vals, colors, alphas)):
        ax.barh(i, v, color=c, alpha=a, height=0.55, linewidth=0, zorder=3)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10, color="#2d2926", fontweight="600")
    ax.axvline(0, color="#c9c2b9", lw=1.5, zorder=2)
    ax.tick_params(axis="x", colors="#c9c2b9", labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xlabel("← Helps you pass              Hurts your chances →",
                  color="#8a8078", fontsize=8.5, labelpad=6)
    ax.grid(axis="x", color="#ece8e1", lw=0.8, zorder=1)
    lim = max(abs(vals).max()+0.12, 0.2)
    ax.set_xlim(-lim, lim)

    rp = mpatches.Patch(color="#dc2626", alpha=0.8, label="🔴 Working against you")
    gp = mpatches.Patch(color="#2d9e6b", alpha=0.8, label="🟢 Working for you")
    ax.legend(handles=[gp,rp], loc="lower right", fontsize=8,
              framealpha=0, labelcolor="#2d2926")
    plt.tight_layout(pad=0.5)
    return fig

def fig_beeswarm():
    shap_all = explainer.shap_values(X_test_scaled)
    mean_abs = np.abs(shap_all).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:10]

    fig, ax = plt.subplots(figsize=(9,5))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    for yi, fi in enumerate(order):
        sv  = shap_all[:,fi]
        fv  = X_test_scaled[:,fi]
        nfv = (fv-fv.min())/(fv.ptp()+1e-8)
        clr = plt.cm.RdYlGn_r(nfv)
        jitter = np.random.uniform(-0.18, 0.18, len(sv))
        ax.scatter(sv, yi+jitter, c=clr, alpha=0.5, s=10, linewidths=0)

    labels = [PRETTY.get(feature_columns[i], feature_columns[i]) for i in order]
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=9.5, color="#2d2926", fontweight="600")
    ax.axvline(0, color="#c9c2b9", lw=1.8)
    ax.tick_params(axis="x", colors="#c9c2b9", labelsize=8)
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.set_xlabel("← Helps students pass          Increases failure risk →",
                  color="#8a8078", fontsize=9)
    ax.grid(axis="x", color="#ece8e1", lw=0.8)
    ax.set_title("Which factors matter most across your whole class?",
                 fontsize=11, color="#2d2926", fontweight="700", pad=10)
    plt.tight_layout(pad=0.5)
    return fig


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 1.5rem 0; border-bottom:1px solid #ece8e1; margin-bottom:1.2rem;'>
        <div style='font-size:1.5rem; font-weight:900; color:#2d2926;'>🌱 EduGuard</div>
        <div style='font-size:0.8rem; color:#8a8078; margin-top:2px; font-weight:600;'>
            Student Success Tracker</div>
    </div>
    """, unsafe_allow_html=True)

    view = st.radio("Who are you?",
                    ["👨‍🏫  I'm a Teacher", "🎓  I'm a Student"])

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.markdown("""<div style='font-size:0.72rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.08em; color:#c9c2b9; margin-bottom:0.8rem;'>
        How accurate is the model?</div>""", unsafe_allow_html=True)

    for label, val, color in [
        ("✅ Overall accuracy",             f"{ACC}%", "#2d9e6b"),
        ("📢 Catches at-risk students",     f"{REC}%", "#d97706"),
        ("⚖️ Balance score",                f"{F1}%",  "#2563eb"),
    ]:
        st.markdown(f"""
        <div style='background:white; border:1px solid #ece8e1; border-radius:12px;
                    padding:0.7rem 0.9rem; margin-bottom:0.5rem;
                    box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
            <div style='font-size:0.75rem; color:#8a8078; font-weight:700;'>{label}</div>
            <div style='font-size:1.3rem; font-weight:800; color:{color};
                        font-family:Lora,serif;'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem; color:#c9c2b9; text-align:center; font-weight:600;'>Phase 4 · ASSE Project</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  TEACHER VIEW
# ══════════════════════════════════════════════
if "Teacher" in view:

    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div style='font-family:Lora,serif; font-size:2rem; font-weight:600; color:#2d2926;'>
            👨‍🏫 Class Overview</div>
        <div style='color:#8a8078; font-size:0.9rem; margin-top:0.3rem;'>
            See which students need attention — and why.</div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    high   = len(teacher_df[teacher_df["Risk Level"].str.contains("High")])
    medium = len(teacher_df[teacher_df["Risk Level"].str.contains("Medium")])
    low    = len(teacher_df[teacher_df["Risk Level"].str.contains("Low")])
    total  = len(teacher_df)

    k1,k2,k3,k4 = st.columns(4)
    for col, icon, val, label, bg, fc in [
        (k1, "👥", total,  "Total Students",                               "#eff6ff","#2563eb"),
        (k2, "🔴", high,   f"Need Urgent Help\n({round(high/total*100)}% of class)",   "#fee2e2","#dc2626"),
        (k3, "🟡", medium, f"Need Some Support\n({round(medium/total*100)}% of class)","#fef3c7","#d97706"),
        (k4, "🟢", low,    f"On Track\n({round(low/total*100)}% of class)",             "#e8f7f1","#2d9e6b"),
    ]:
        with col:
            st.markdown(f"""
            <div class="kpi" style="background:{bg}; border-color:{fc}33;">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-value" style="color:{fc};">{val}</div>
                <div class="kpi-label" style="color:{fc}cc;">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # Filters
    f1c, f2c, _ = st.columns([2,2,3])
    with f1c:
        rf = st.selectbox("Show students at risk level:",
                          ["All levels","🔴 High only","🟡 Medium only","🟢 Low only"])
    with f2c:
        min_att = st.slider("Attendance at least (%)", 0, 100, 0)

    filtered = teacher_df.copy()
    if "High"   in rf: filtered = filtered[filtered["Risk Level"].str.contains("High")]
    elif "Medium" in rf: filtered = filtered[filtered["Risk Level"].str.contains("Medium")]
    elif "Low"   in rf: filtered = filtered[filtered["Risk Level"].str.contains("Low")]
    filtered = filtered[filtered["Attendance"].str.rstrip("%").astype(float) >= min_att]

    st.markdown(f"""
    <div style='font-family:Lora,serif; font-size:1.1rem; font-weight:600;
                color:#2d2926; margin-bottom:0.4rem; margin-top:0.3rem;'>
        Student List
        <span style='color:#8a8078; font-size:0.85rem; font-weight:400;'>
            — {len(filtered)} students shown</span>
    </div>""", unsafe_allow_html=True)

    def style_risk(v):
        if "High"   in str(v): return "color:#dc2626; font-weight:800"
        if "Medium" in str(v): return "color:#d97706; font-weight:800"
        return "color:#2d9e6b; font-weight:800"
    def style_prob(v):
        val = float(str(v).rstrip("%"))
        if val>=70: return "color:#dc2626; font-weight:700"
        if val>=40: return "color:#d97706; font-weight:700"
        return "color:#2d9e6b; font-weight:700"
    st.dataframe(
        filtered.style
        .map(style_risk, subset=["Risk Level"])
        .map(style_prob, subset=["Chance of Failing"])
        .set_properties(**{"font-size":"14px"}),
        use_container_width=True, height=330,
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # SHAP
    st.markdown("""
    <div style='font-family:Lora,serif; font-size:1.25rem; font-weight:600;
                color:#2d2926; margin-bottom:0.2rem;'>
        📊 What affects students most?</div>
    <div style='font-size:0.85rem; color:#8a8078; margin-bottom:1rem;'>
        Each dot is a student. Green = helps pass. Red = increases failure risk.</div>
    """, unsafe_allow_html=True)
    with st.spinner("Building chart…"):
        st.pyplot(fig_beeswarm(), use_container_width=True)
    plt.close()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Interventions
    st.markdown("""
    <div style='font-family:Lora,serif; font-size:1.25rem; font-weight:600;
                color:#2d2926; margin-bottom:0.2rem;'>🛠️ What can you do?</div>
    <div style='font-size:0.85rem; color:#8a8078; margin-bottom:1rem;'>
        Suggested actions based on the risk levels in your class.</div>
    """, unsafe_allow_html=True)

    i1,i2,i3 = st.columns(3)
    for col, icon, title, bg, fc, desc in [
        (i1,"🔴","Counsellor Referral","#fee2e2","#dc2626",
         "Students flagged as HIGH risk should speak with a school counsellor as soon as possible for a one-on-one conversation."),
        (i2,"📞","Parent Outreach","#fef3c7","#d97706",
         "Students with attendance below 60% benefit most from a parent check-in call to understand what's happening at home."),
        (i3,"📚","Share Learning Resources","#e8f7f1","#2d9e6b",
         "Each student gets a personalised list of 3 free resources matched to their needs — share them via the Student View."),
    ]:
        with col:
            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {fc}33; border-radius:16px;
                        padding:1.2rem; min-height:160px;'>
                <div style='font-size:1.5rem; margin-bottom:0.4rem;'>{icon}</div>
                <div style='font-weight:800; color:{fc}; font-size:0.95rem; margin-bottom:0.4rem;'>{title}</div>
                <div style='font-size:0.82rem; color:#2d2926; line-height:1.5;'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
#  STUDENT VIEW
# ══════════════════════════════════════════════
else:
    st.markdown("""
    <div style='margin-bottom:0.5rem;'>
        <div style='font-family:Lora,serif; font-size:2rem; font-weight:600; color:#2d2926;'>
            🎓 How am I doing?</div>
        <div style='color:#8a8078; font-size:0.9rem; margin-top:0.3rem;'>
            Fill in your details below — it only takes 2 minutes.
            We'll tell you how you're doing and what you can do to improve.</div>
    </div>
    <div style='height:0.5rem'></div>
    """, unsafe_allow_html=True)

    with st.form("student_form", border=False):

        # Section 1 — Study Habits
        st.markdown("""
        <div style='background:white; border-radius:20px; border:1px solid #ece8e1;
                    padding:1.5rem 1.5rem 1rem 1.5rem; margin-bottom:1rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
            <div style='font-family:Lora,serif; font-size:1.1rem; font-weight:600;
                        color:#2d2926; margin-bottom:1rem;'>📖 Your Study Habits</div>
        """, unsafe_allow_html=True)

        s1,s2,s3 = st.columns(3)
        with s1:
            hours_studied   = st.slider("How many hours do you study per week?", 0, 44, 15,
                                        help="Count all subjects combined")
        with s2:
            attendance      = st.slider("What is your attendance percentage?", 40, 100, 75,
                                        help="How often do you show up to class?")
        with s3:
            previous_scores = st.slider("What was your score on your last exam?", 40, 100, 65,
                                        help="Out of 100")

        s4,s5,s6 = st.columns(3)
        with s4:
            tutoring  = st.slider("How many tutoring sessions per month?", 0, 8, 1)
        with s5:
            sleep_hours = st.slider("How many hours do you sleep per night?", 4, 10, 7)
        with s6:
            physical  = st.slider("Hours of sport / exercise per week?", 0, 6, 2)

        st.markdown("</div>", unsafe_allow_html=True)

        # Section 2 — About You
        st.markdown("""
        <div style='background:white; border-radius:20px; border:1px solid #ece8e1;
                    padding:1.5rem 1.5rem 1rem 1.5rem; margin-bottom:1rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
            <div style='font-family:Lora,serif; font-size:1.1rem; font-weight:600;
                        color:#2d2926; margin-bottom:1rem;'>🙋 A Bit About You</div>
        """, unsafe_allow_html=True)

        a1,a2,a3 = st.columns(3)
        with a1:
            motivation_raw = st.selectbox(
                "How motivated do you feel about studying?",
                ["Low — I find it hard to start",
                 "Medium — I study when I have to",
                 "High — I genuinely enjoy learning"])
        with a2:
            resources_raw = st.selectbox(
                "How easily can you access study materials?",
                ["Low — I struggle to get books/internet",
                 "Medium — I have some access",
                 "High — I have everything I need"])
        with a3:
            peer_raw = st.selectbox(
                "Do your friends support your studies?",
                ["Negative — They distract me",
                 "Neutral — It doesn't affect me",
                 "Positive — They motivate me"])

        a4,a5,a6 = st.columns(3)
        with a4:
            parental_raw = st.selectbox(
                "Are your parents involved in your education?",
                ["Low — Not much", "Medium — Somewhat", "High — Very involved"])
        with a5:
            gender = st.selectbox("Gender", ["Female","Male"])
        with a6:
            extracurricular = st.selectbox(
                "Do you do extracurricular activities?", ["No","Yes"])

        a7,_,__ = st.columns(3)
        with a7:
            learning_dis = st.selectbox(
                "Do you have a diagnosed learning difficulty?", ["No","Yes"])

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Show My Results")

    # ── RESULTS ──────────────────────────────────
    if submitted:
        enc = {
            "Low — I find it hard to start":0,
            "Medium — I study when I have to":1,
            "High — I genuinely enjoy learning":2,
            "Low — I struggle to get books/internet":0,
            "Medium — I have some access":1,
            "High — I have everything I need":2,
            "Negative — They distract me":0,
            "Neutral — It doesn't affect me":1,
            "Positive — They motivate me":2,
            "Low — Not much":0,"Medium — Somewhat":1,"High — Very involved":2,
            "No":0,"Yes":1,"Female":0,"Male":1,
        }

        stress_proxy = round(1 - (
            (enc[motivation_raw]/2)*0.4 +
            (min(hours_studied,30)/30)*0.3 +
            (min(sleep_hours,9)/9)*0.3
        ), 4)

        raw = {
            "Motivation_Level":               enc[motivation_raw],
            "Peer_Influence":                 enc[peer_raw],
            "Access_to_Resources":            enc[resources_raw],
            "Parental_Involvement":           enc[parental_raw],
            "Gender_Male":                    enc[gender],
            "Extracurricular_Activities_Yes": enc[extracurricular],
            "Learning_Disabilities_Yes":      enc[learning_dis],
            "Hours_Studied":                  float(hours_studied),
            "Attendance":                     float(attendance),
            "Sleep_Hours":                    float(sleep_hours),
            "Previous_Scores":                float(previous_scores),
            "Tutoring_Sessions":              float(tutoring),
            "Physical_Activity":              float(physical),
            "Stress_Proxy":                   stress_proxy,
        }

        prob_fail, shap_vals, risk, recs = run_prediction(raw)

        # Hero banner
        hero_cfg = {
            "HIGH":   ("#fee2e2","#dc2626","🚨",
                       "You need some support right now",
                       "Your results suggest you're at high risk of not passing. Don't worry — this is exactly why we built this tool. Your personalised plan is below."),
            "MEDIUM": ("#fef3c7","#d97706","⚠️",
                       "You're getting there — stay focused",
                       "You're doing okay, but there are a few things to work on. Small changes can make a big difference."),
            "LOW":    ("#e8f7f1","#2d9e6b","🎉",
                       "You're on the right track!",
                       "Great job — your habits are working well. Keep it up and check the tips below to stay strong all the way to exam day."),
        }
        hbg, hfc, hemoji, htitle, hsub = hero_cfg[risk]

        st.markdown(f"""
        <div style='background:{hbg}; border:2px solid {hfc}33; border-radius:24px;
                    padding:2rem; text-align:center; margin-bottom:1.5rem;'>
            <div style='font-size:3rem; margin-bottom:0.4rem;'>{hemoji}</div>
            <div style='font-family:Lora,serif; font-size:1.8rem; font-weight:600;
                        color:{hfc}; margin-bottom:0.5rem;'>{htitle}</div>
            <div style='color:#2d2926; font-size:0.95rem; max-width:520px;
                        margin:0 auto; line-height:1.6;'>{hsub}</div>
        </div>""", unsafe_allow_html=True)

        # ── Left: donut + stats | Right: SHAP
        col_l, col_r = st.columns([1, 1.7])

        with col_l:
            st.markdown("""
            <div style='font-family:Lora,serif; font-size:1.05rem; font-weight:600;
                        color:#2d2926; margin-bottom:0.3rem;'>Your Risk Score</div>
            <div style='font-size:0.82rem; color:#8a8078; margin-bottom:0.6rem;'>
                The lower this number, the better.</div>
            """, unsafe_allow_html=True)

            st.pyplot(fig_donut(prob_fail, risk), use_container_width=False)
            plt.close()

            st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
            for label, val, bg, fc in [
                ("📅 Attendance",      f"{attendance}%",
                 "#e8f7f1" if attendance>=75 else "#fee2e2",
                 "#2d9e6b" if attendance>=75 else "#dc2626"),
                ("⏱️ Study Time",      f"{hours_studied}h/week",
                 "#e8f7f1" if hours_studied>=15 else "#fee2e2",
                 "#2d9e6b" if hours_studied>=15 else "#dc2626"),
                ("😴 Sleep",          f"{sleep_hours}h/night",
                 "#e8f7f1" if sleep_hours>=7 else "#fef3c7",
                 "#2d9e6b" if sleep_hours>=7 else "#d97706"),
                ("📝 Last Exam Score",f"{previous_scores}/100",
                 "#e8f7f1" if previous_scores>=65 else "#fee2e2",
                 "#2d9e6b" if previous_scores>=65 else "#dc2626"),
            ]:
                st.markdown(f"""
                <div style='background:{bg}; border-radius:12px; padding:0.6rem 0.9rem;
                            margin-bottom:0.45rem; display:flex; justify-content:space-between;
                            align-items:center;'>
                    <span style='font-size:0.82rem; font-weight:700; color:#2d2926;'>{label}</span>
                    <span style='font-size:0.85rem; font-weight:800; color:{fc};'>{val}</span>
                </div>""", unsafe_allow_html=True)

        with col_r:
            st.markdown("""
            <div style='font-family:Lora,serif; font-size:1.05rem; font-weight:600;
                        color:#2d2926; margin-bottom:0.3rem;'>
                What's affecting your chances?</div>
            <div style='font-size:0.82rem; color:#8a8078; margin-bottom:0.8rem;'>
                Green bars = things working in your favour.
                Red bars = things to improve.</div>
            """, unsafe_allow_html=True)

            st.pyplot(fig_shap_simple(shap_vals, feature_columns),
                      use_container_width=True)
            plt.close()

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        # ── Recommendations
        st.markdown("""
        <div style='font-family:Lora,serif; font-size:1.3rem; font-weight:600;
                    color:#2d2926; margin-bottom:0.2rem;'>
            🎯 Your Top 3 Recommended Resources</div>
        <div style='font-size:0.85rem; color:#8a8078; margin-bottom:1rem;'>
            Chosen specifically for you based on where you need the most help — all free.</div>
        """, unsafe_allow_html=True)

        rc1,rc2,rc3 = st.columns(3)
        rank_labels = ["Best match for you", "Great for your needs", "Also very helpful"]
        for col, rec, rl in zip([rc1,rc2,rc3], recs, rank_labels):
            with col:
                st.markdown(f"""
                <div style='background:white; border:1.5px solid #ece8e1; border-radius:18px;
                            padding:1.3rem; min-height:220px; display:flex; flex-direction:column;
                            justify-content:space-between;
                            box-shadow:0 2px 10px rgba(0,0,0,0.06);'>
                    <div>
                        <div style='background:{rec["bg"]}; width:44px; height:44px;
                                    border-radius:12px; display:flex; align-items:center;
                                    justify-content:center; font-size:1.4rem;
                                    margin-bottom:0.6rem;'>{rec["icon"]}</div>
                        <div style='font-weight:800; font-size:0.87rem; color:#2d2926;
                                    line-height:1.4; margin-bottom:0.3rem;'>{rec["title"]}</div>
                        <div style='font-size:0.72rem; color:#8a8078;'>{rl}</div>
                    </div>
                    <div>
                        <span style='background:{rec["bg"]}; color:{rec["ic"]};
                                     border-radius:20px; padding:2px 10px;
                                     font-size:0.72rem; font-weight:700;'>{rec["topic"]}</span>
                        <div style='margin-top:0.5rem;'>
                            <a href='{rec["url"]}' target='_blank'
                               style='color:#2563eb; font-size:0.78rem;
                                      font-weight:700; text-decoration:none;'>
                               Visit resource →</a>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        # ── Action Plan
        st.markdown("""
        <div style='font-family:Lora,serif; font-size:1.3rem; font-weight:600;
                    color:#2d2926; margin-bottom:0.2rem;'>
            📌 Your Personal Action Plan</div>
        <div style='font-size:0.85rem; color:#8a8078; margin-bottom:1rem;'>
            Three things you can start doing this week.</div>
        """, unsafe_allow_html=True)

        tip_map = {
            "Motivation_Level":  ("💪","Boost your motivation",
                                  "Watch a TED-Ed talk each morning for 5 minutes. Small inspiration goes a long way."),
            "Hours_Studied":     ("⏱️","Study more consistently",
                                  f"You study {hours_studied}h/week. Try adding just 30 min/day — that's 3.5 more hours per week!"),
            "Sleep_Hours":       ("😴","Improve your sleep",
                                  f"You get {sleep_hours}h/night. Aim for 8h — even one extra hour improves memory and focus."),
            "Attendance":        ("📅","Show up more often",
                                  f"You attend {attendance}% of classes. Missing class means missing explanations you can't easily replace."),
            "Previous_Scores":   ("📝","Revisit past topics",
                                  "Go back to the chapters you found hardest last term. A stronger base makes everything easier."),
            "Access_to_Resources":("📚","Use free online resources",
                                   "Khan Academy, OpenStax, and YouTube EDU are 100% free and cover almost every subject."),
            "Stress_Proxy":      ("🧘","Manage your stress",
                                   "Try 5 minutes of deep breathing before studying. Apps like Calm or Headspace are free for students."),
            "Tutoring_Sessions": ("🙋","Get extra help",
                                   "Even one tutoring session per week can dramatically improve your understanding."),
            "Physical_Activity": ("🏃","Add some exercise",
                                   "Exercise improves memory and reduces stress. A 20-minute walk before studying really helps!"),
            "Peer_Influence":    ("👯","Choose motivating study partners",
                                   "Study groups with positive peers can double your productivity and make it more enjoyable."),
        }

        bottom3 = [feature_columns[i] for i in np.argsort(shap_vals)[:3]]
        tips    = [tip_map.get(f, ("💡","Focus on improvement",
                                   f"Pay more attention to {PRETTY.get(f,f)}."))
                   for f in bottom3]

        t1,t2,t3 = st.columns(3)
        for col,(icon,title,desc),bg,fc in zip(
            [t1,t2,t3], tips,
            ["#eff6ff","#e8f7f1","#fef3c7"],
            ["#2563eb","#2d9e6b","#d97706"],
        ):
            with col:
                st.markdown(f"""
                <div style='background:{bg}; border:1.5px solid {fc}44; border-radius:16px;
                            padding:1.2rem; min-height:150px;
                            box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
                    <div style='font-size:1.3rem; margin-bottom:0.3rem;'>{icon}</div>
                    <div style='font-weight:800; font-size:0.9rem; color:#2d2926;
                                margin-bottom:0.25rem;'>{title}</div>
                    <div style='font-size:0.8rem; color:#2d2926;
                                line-height:1.45; opacity:0.85;'>{desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#f5f3ff; border:1px solid #c4b5fd; border-radius:14px;
                    padding:1rem 1.2rem; font-size:0.82rem; color:#5b21b6;
                    font-weight:600; text-align:center;'>
            ℹ️ This tool uses AI to <em>estimate</em> your risk level based on your answers.
            It is meant to <strong>guide</strong> you, not to label you.
            Talk to your teacher or counsellor if you need more support. You've got this! 🌱
        </div>""", unsafe_allow_html=True)
