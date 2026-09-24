# ============================================================
#  Student Career Success Prediction — Streamlit App
#  Single file: Backend (ML pipeline) + Frontend (Streamlit UI)
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── sklearn imports (backend)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error

# ════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Student Career Success Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS
st.markdown("""
<style>
    .main-header{font-size:2.2rem;font-weight:700;color:#1f2328;margin-bottom:0;}
    .sub-header{font-size:1rem;color:#57606a;margin-top:0;}
    .metric-card{background:#f7f8fa;border:1px solid #e5e7eb;border-radius:8px;
                 padding:1rem;text-align:center;}
    .placed-badge{background:#d4edda;color:#155724;padding:0.4rem 1.2rem;
                  border-radius:20px;font-weight:700;font-size:1.3rem;}
    .notplaced-badge{background:#f8d7da;color:#721c24;padding:0.4rem 1.2rem;
                     border-radius:20px;font-weight:700;font-size:1.3rem;}
    .section-title{font-size:1.25rem;font-weight:600;color:#1f2328;
                   border-bottom:2px solid #3b82d4;padding-bottom:4px;margin-bottom:1rem;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
#  BACKEND — CONSTANTS & HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════

CATEGORICAL_COLS = ["Gender", "University_Year", "Major", "Academic_Performance",
                    "GitHub_Profile", "Leadership_Experience", "LinkedIn_Profile",
                    "English_Proficiency"]

DROP_COLS = ["Student_ID", "Company_Tier", "Career_Field", "Placement_Mode"]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add three composite engineered features."""
    d = data.copy()
    d["Skill_Score"] = (
        d["Programming_Skill"] * 0.3 +
        d["Communication_Skills"] * 0.2 +
        d["Problem_Solving"] * 0.25 +
        d["Teamwork"] * 0.15 +
        d["Interview_Score"] * 0.1
    )
    d["Activity_Index"]     = d["Projects_Completed"] + d["Certifications"] + d["Hackathons"]
    d["Academic_Intensity"] = d["CGPA"] * d["Study_Hours_Per_Week"] / 10
    return d


@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    return pd.read_csv("student_career_success_dataset.csv")


@st.cache_resource(show_spinner="Training models — please wait (one-time only)...")
def train_models() -> dict:
    """Train placement classifier and salary regressor. Cached after first run."""
    df = load_data()

    # ── Shared transformers
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc",  StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    # ── Classification: Placement Status
    df_clf = df.drop(columns=DROP_COLS)
    df_clf = add_engineered_features(df_clf)
    df_clf["Target"] = (df_clf["Placement_Status"] == "Placed").astype(int)
    df_clf = df_clf.drop(columns=["Placement_Status"])

    X_clf  = df_clf.drop(columns=["Target"])
    y_clf  = df_clf["Target"]
    num_cols = [c for c in X_clf.columns if c not in CATEGORICAL_COLS]

    pre_clf = ColumnTransformer([("num", num_pipe, num_cols),
                                  ("cat", cat_pipe, CATEGORICAL_COLS)])

    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    clf_pipe = Pipeline([("pre", pre_clf),
                          ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))])
    clf_pipe.fit(X_tr_c, y_tr_c)

    y_pred_c = clf_pipe.predict(X_te_c)
    y_prob_c = clf_pipe.predict_proba(X_te_c)[:, 1]
    clf_acc  = accuracy_score(y_te_c, y_pred_c)
    clf_auc  = roc_auc_score(y_te_c, y_prob_c)

    # Feature importance
    rf_model  = clf_pipe.named_steps["clf"]
    ohe_names = list(clf_pipe.named_steps["pre"]
                     .named_transformers_["cat"]
                     .named_steps["ohe"]
                     .get_feature_names_out(CATEGORICAL_COLS))
    feat_imp  = pd.Series(rf_model.feature_importances_,
                           index=num_cols + ohe_names).nlargest(15)

    # ── Regression: Starting Salary (placed students only)
    df_placed = df[df["Placement_Status"] == "Placed"].copy()
    df_placed = df_placed.drop(columns=DROP_COLS + ["Placement_Status"])
    df_placed = add_engineered_features(df_placed)

    X_reg  = df_placed.drop(columns=["Starting_Salary_USD"])
    y_reg  = df_placed["Starting_Salary_USD"]

    cat_reg = [c for c in CATEGORICAL_COLS if c in X_reg.columns]
    num_reg = [c for c in X_reg.columns  if c not in cat_reg]
    pre_reg = ColumnTransformer([("num", num_pipe, num_reg),
                                  ("cat", cat_pipe, cat_reg)])

    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)

    reg_pipe = Pipeline([("pre", pre_reg),
                          ("reg", GradientBoostingRegressor(n_estimators=200, random_state=42))])
    reg_pipe.fit(X_tr_r, y_tr_r)

    y_pred_r = reg_pipe.predict(X_te_r)
    reg_r2   = r2_score(y_te_r, y_pred_r)
    reg_mae  = mean_absolute_error(y_te_r, y_pred_r)

    return {
        "clf_pipe":  clf_pipe,
        "reg_pipe":  reg_pipe,
        "clf_acc":   clf_acc,
        "clf_auc":   clf_auc,
        "reg_r2":    reg_r2,
        "reg_mae":   reg_mae,
        "feat_imp":  feat_imp,
        "num_cols":  num_cols,
    }


def predict_student(models: dict, student_dict: dict):
    """Run placement + salary prediction for a single student input dict."""
    row = pd.DataFrame([student_dict])
    row = add_engineered_features(row)
    placed_prob  = models["clf_pipe"].predict_proba(row)[0][1]
    placed_label = models["clf_pipe"].predict(row)[0]
    salary_pred  = models["reg_pipe"].predict(row)[0]
    return int(placed_label), float(placed_prob), float(salary_pred)


# ════════════════════════════════════════════════════════════
#  LOAD DATA & MODELS (cached)
# ════════════════════════════════════════════════════════════
df     = load_data()
models = train_models()

# ════════════════════════════════════════════════════════════
#  SIDEBAR — NAVIGATION
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎓 Career Success Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔮 Predict", "📊 Model Info"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Dataset: student_career_success_dataset.csv")
    st.caption("Model: Random Forest + Gradient Boosting")
    st.caption("50,000 records · 29 features")

# ════════════════════════════════════════════════════════════
#  PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("<h1 class='main-header'>🎓 Student Career Success Dashboard</h1>",
                unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Explore the dataset and understand key placement trends</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    placed_df   = df[df["Placement_Status"] == "Placed"]
    placement_r = len(placed_df) / len(df) * 100

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Students",    f"{len(df):,}")
    k2.metric("Placed",            f"{len(placed_df):,}")
    k3.metric("Placement Rate",    f"{placement_r:.1f}%")
    k4.metric("Avg CGPA (Placed)", f"{placed_df['CGPA'].mean():.2f}")
    k5.metric("Avg Starting Salary", f"${placed_df['Starting_Salary_USD'].mean():,.0f}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Placement Status Distribution</div>",
                    unsafe_allow_html=True)
        counts = df["Placement_Status"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(counts.index, counts.values, color=["#3b82d4", "#e74c3c"])
        for i, v in enumerate(counts.values):
            ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=9)
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("<div class='section-title'>Placement Rate by Major</div>",
                    unsafe_allow_html=True)
        major_p = df.groupby("Major")["Placement_Status"].apply(
            lambda x: (x == "Placed").sum() / len(x) * 100
        ).sort_values(ascending=True)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.barh(major_p.index, major_p.values, color="#3b82d4")
        ax2.set_xlabel("Placement Rate (%)")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='section-title'>CGPA Distribution by Placement</div>",
                    unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        for label, grp in df.groupby("Placement_Status"):
            ax3.hist(grp["CGPA"], bins=20, alpha=0.6, label=label)
        ax3.set_xlabel("CGPA")
        ax3.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

    with col4:
        st.markdown("<div class='section-title'>Median Salary by Company Tier</div>",
                    unsafe_allow_html=True)
        tier_sal = placed_df.groupby("Company_Tier")["Starting_Salary_USD"].median().sort_values(ascending=False)
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        ax4.bar(tier_sal.index, tier_sal.values, color=["#3b82d4", "#7c5cd8", "#27ae60"])
        ax4.set_ylabel("Median Salary (USD)")
        for i, v in enumerate(tier_sal.values):
            ax4.text(i, v + 200, f"${v:,.0f}", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")
    st.markdown("<div class='section-title'>Correlation Heatmap</div>", unsafe_allow_html=True)
    num_c = ["Age", "Attendance_Percentage", "Study_Hours_Per_Week", "CGPA",
             "Programming_Skill", "Projects_Completed", "Certifications",
             "Hackathons", "Internships", "Resume_Score",
             "Communication_Skills", "Interview_Score"]
    fig5, ax5 = plt.subplots(figsize=(12, 5))
    sns.heatmap(df[num_c].corr(), annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.4, ax=ax5, cbar_kws={"shrink": 0.8})
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()

    st.markdown("---")
    with st.expander("📋 View Raw Dataset Sample (first 100 rows)"):
        st.dataframe(df.head(100), use_container_width=True)


# ════════════════════════════════════════════════════════════
#  PAGE 2 — PREDICT
# ════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown("<h1 class='main-header'>🔮 Placement Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Enter student details to predict placement status and estimated salary</p>",
                unsafe_allow_html=True)
    st.markdown("---")

    with st.form("prediction_form"):
        st.markdown("#### 👤 Personal & Academic Information")
        c1, c2, c3 = st.columns(3)
        age       = c1.number_input("Age", 18, 35, 21)
        gender    = c2.selectbox("Gender", ["Male", "Female"])
        univ_year = c3.selectbox("University Year",
                                  ["Freshman", "Sophomore", "Junior", "Senior"])

        c4, c5, c6 = st.columns(3)
        major     = c4.selectbox("Major", ["Computer Science", "Information Technology",
                                            "Software Engineering", "Artificial Intelligence",
                                            "Business Analytics", "Data Science",
                                            "Cybersecurity", "Electrical Engineering"])
        cgpa      = c5.slider("CGPA", 2.0, 4.0, 3.0, 0.01)
        acad_perf = c6.selectbox("Academic Performance", ["Poor", "Average", "Good", "Excellent"])

        st.markdown("#### 📚 Study & Activity")
        c7, c8, c9 = st.columns(3)
        attend_pct    = c7.slider("Attendance (%)", 40, 100, 80)
        study_hrs     = c8.slider("Study Hours/Week", 5, 50, 20)
        projects      = c9.number_input("Projects Completed", 0, 20, 5)

        c10, c11, c12 = st.columns(3)
        certifications = c10.number_input("Certifications", 0, 10, 1)
        hackathons     = c11.number_input("Hackathons", 0, 15, 2)
        internships    = c12.number_input("Internships", 0, 10, 1)

        st.markdown("#### 💡 Skills & Profile")
        c13, c14, c15 = st.columns(3)
        prog_skill    = c13.slider("Programming Skill (1-10)", 1, 10, 7)
        comm_skill    = c14.slider("Communication Skills (1-10)", 1, 10, 7)
        problem_solve = c15.slider("Problem Solving (1-10)", 1, 10, 7)

        c16, c17, c18 = st.columns(3)
        teamwork      = c16.slider("Teamwork (1-10)", 1, 10, 7)
        interview_sc  = c17.slider("Interview Score (1-100)", 1, 100, 70)
        english       = c18.selectbox("English Proficiency", ["Basic", "Intermediate", "Advanced"])

        c19, c20 = st.columns(2)
        resume_score = c19.slider("Resume Score (1-100)", 1, 100, 80)
        leadership   = c20.selectbox("Leadership Experience", ["No", "Yes"])

        c21, c22 = st.columns(2)
        github   = c21.selectbox("GitHub Profile", ["No", "Yes"])
        linkedin = c22.selectbox("LinkedIn Profile", ["No", "Yes"])

        submitted = st.form_submit_button("🚀 Predict Now", use_container_width=True)

    if submitted:
        # Compute Employability_Score approximation
        emp_score = (prog_skill * 10 + comm_skill * 8 + problem_solve * 9 + teamwork * 7) / 3.4

        student_input = {
            "Age":                   age,
            "Gender":                gender,
            "University_Year":       univ_year,
            "Major":                 major,
            "Attendance_Percentage": attend_pct,
            "Study_Hours_Per_Week":  study_hrs,
            "CGPA":                  cgpa,
            "Academic_Performance":  acad_perf,
            "Programming_Skill":     prog_skill,
            "Projects_Completed":    projects,
            "Certifications":        certifications,
            "Hackathons":            hackathons,
            "GitHub_Profile":        github,
            "Internships":           internships,
            "Leadership_Experience": leadership,
            "LinkedIn_Profile":      linkedin,
            "Resume_Score":          resume_score,
            "Communication_Skills":  comm_skill,
            "Teamwork":              teamwork,
            "Problem_Solving":       problem_solve,
            "English_Proficiency":   english,
            "Interview_Score":       interview_sc,
            "Employability_Score":   emp_score,
        }

        placed_label, placed_prob, salary_pred = predict_student(models, student_input)

        st.markdown("---")
        st.markdown("### 📋 Prediction Results")

        r1, r2, r3 = st.columns(3)
        with r1:
            if placed_label == 1:
                st.markdown("<div style='text-align:center'>"
                            "<span class='placed-badge'>✅ PLACED</span></div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div style='text-align:center'>"
                            "<span class='notplaced-badge'>❌ NOT PLACED</span></div>",
                            unsafe_allow_html=True)
        with r2:
            st.metric("Placement Probability", f"{placed_prob * 100:.1f}%")
        with r3:
            if placed_label == 1:
                st.metric("Estimated Starting Salary", f"${salary_pred:,.0f} USD")
            else:
                st.metric("Estimated Starting Salary", "N/A (Not Placed)")

        # Probability gauge bar
        st.markdown("---")
        fig_g, ax_g = plt.subplots(figsize=(8, 1.2))
        color = "#27ae60" if placed_prob > 0.5 else "#e74c3c"
        ax_g.barh(["Placement Probability"], [placed_prob], color=color, height=0.5)
        ax_g.barh(["Placement Probability"], [1 - placed_prob],
                   left=[placed_prob], color="#e5e7eb", height=0.5)
        ax_g.set_xlim(0, 1)
        ax_g.axvline(0.5, color="gray", lw=1, linestyle="--")
        ax_g.text(placed_prob / 2, 0, f"{placed_prob*100:.1f}%",
                   ha="center", va="center", color="white", fontweight="bold", fontsize=12)
        ax_g.set_xticks([])
        plt.tight_layout()
        st.pyplot(fig_g)
        plt.close()

        # Personalized recommendations
        st.markdown("---")
        st.markdown("### 💡 Personalized Recommendations")
        recs = []
        if cgpa < 3.0:             recs.append("📚 Improve CGPA above 3.0 — it is a key differentiator.")
        if prog_skill < 7:         recs.append("💻 Strengthen programming skills (target ≥ 7/10).")
        if internships < 2:        recs.append("🏢 Complete at least 2 internships for practical exposure.")
        if projects < 5:           recs.append("🛠️ Build more projects (aim for 5+) to showcase skills.")
        if interview_sc < 70:      recs.append("🎤 Practice mock interviews — target Interview Score ≥ 70.")
        if github == "No":         recs.append("🐙 Create a GitHub profile to demonstrate your work.")
        if linkedin == "No":       recs.append("💼 Build a LinkedIn profile for networking opportunities.")
        if certifications == 0:    recs.append("📜 Earn industry certifications (AWS, Google, Coursera, etc.).")
        if hackathons < 2:         recs.append("🏆 Participate in hackathons to build competitive skills.")
        if attend_pct < 75:        recs.append("🎓 Improve class attendance above 75%.")
        if resume_score < 70:      recs.append("📄 Improve your resume — aim for a Resume Score ≥ 70.")

        if not recs:
            st.success("🌟 Excellent profile! You have a strong chance of placement. Keep it up!")
        else:
            for r in recs:
                st.info(r)


# ════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL INFO
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Info":
    st.markdown("<h1 class='main-header'>📊 Model Performance & Insights</h1>",
                unsafe_allow_html=True)
    st.markdown("---")

    clf_acc = models["clf_acc"]
    clf_auc = models["clf_auc"]
    reg_r2  = models["reg_r2"]
    reg_mae = models["reg_mae"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Classifier",    "Random Forest")
    m2.metric("Accuracy",      f"{clf_acc * 100:.2f}%")
    m3.metric("AUC-ROC",       f"{clf_auc:.4f}")
    m4.metric("Salary R²",     f"{reg_r2:.4f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-title'>Top 15 Feature Importances</div>",
                    unsafe_allow_html=True)
        fi = models["feat_imp"].sort_values()
        fig_fi, ax_fi = plt.subplots(figsize=(6, 5))
        ax_fi.barh(fi.index, fi.values, color="#3b82d4")
        ax_fi.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

    with col_b:
        st.markdown("<div class='section-title'>Model Architecture</div>",
                    unsafe_allow_html=True)
        st.markdown("""
| Component | Detail |
|-----------|--------|
| **Classification Model** | Random Forest (200 trees) |
| **Regression Model** | Gradient Boosting (200 estimators) |
| **Preprocessing** | StandardScaler + OneHotEncoder |
| **Missing Values** | SimpleImputer (median / mode) |
| **Train / Test Split** | 80% / 20% stratified |
| **Engineered Features** | Skill_Score, Activity_Index, Academic_Intensity |
| **Total Features** | ~40 (after one-hot encoding) |
| **Dataset Size** | 50,000 students |
        """)

    st.markdown("---")
    st.markdown("<div class='section-title'>Salary Prediction Error</div>", unsafe_allow_html=True)
    st.info(f"Mean Absolute Error (Salary): **${reg_mae:,.0f} USD**  |  R² Score: **{reg_r2:.4f}**")

    st.markdown("---")
    st.markdown("<div class='section-title'>About This Project</div>", unsafe_allow_html=True)
    st.markdown("""
This application predicts **student placement outcomes** using a machine learning pipeline trained on
50,000 student records. The system uses a **Random Forest Classifier** for binary placement prediction
and a **Gradient Boosting Regressor** for salary estimation.

**Key Features Used:**
- Academic: CGPA, Attendance %, Study Hours
- Technical: Programming Skill, Projects, Certifications, Hackathons
- Profile: GitHub, LinkedIn, Internships, Resume Score
- Soft Skills: Communication, Teamwork, Problem Solving, Interview Score
- Engineered: Skill Score (weighted), Activity Index, Academic Intensity
    """)
