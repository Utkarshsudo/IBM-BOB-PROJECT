# 🎓 Student Career Success Prediction

> A full-stack machine learning application that predicts student placement outcomes and estimates starting salaries, built with Python, scikit-learn, and Streamlit.

---

## 📋 Project Description

**Student Career Success Prediction** is an end-to-end ML project that answers two critical questions:

1. **Will this student be placed?** → Binary Classification (Placed / Not Placed)  
2. **What salary can they expect?** → Regression (Starting Salary in USD)

The project is designed as a **single Python file** (`app.py`) that combines:
- A **backend ML pipeline** — preprocessing, feature engineering, model training, evaluation
- A **Streamlit frontend** — interactive dashboard, prediction form, model insights

Trained on **50,000 real-world student records** across 8 majors, 17 career fields, and 4 company tiers.

---

## 📂 Dataset

| Property | Value |
|----------|-------|
| **File** | `student_career_success_dataset.csv` |
| **Records** | 50,000 students |
| **Features** | 29 columns |
| **Placement Rate** | 78.08% (39,040 placed) |
| **Target (Classification)** | `Placement_Status` (Placed / Not Placed) |
| **Target (Regression)** | `Starting_Salary_USD` |

**Feature Categories:**
- 🎓 Academic: CGPA, Attendance, Study Hours, Major, Academic Performance
- 💻 Technical: Programming Skill, Projects, Certifications, Hackathons, GitHub
- 👤 Professional: Internships, Leadership, LinkedIn, Resume Score
- 🗣️ Soft Skills: Communication, Teamwork, Problem Solving, Interview Score, English Proficiency

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Frontend / UI | **Streamlit** >= 1.32 |
| ML Framework | **scikit-learn** >= 1.3 |
| Data Processing | pandas >= 2.0, numpy >= 1.24 |
| Visualisation | matplotlib >= 3.7, seaborn >= 0.12 |
| Model Persistence | joblib >= 1.3 |
| Notebook | Jupyter >= 1.0 |

---

## 📁 Project Files

```
project/
├── app.py                                  ← Single-file: Backend ML + Streamlit Frontend
├── StudentCareerSuccess_ProjectName.ipynb  ← Full Jupyter Notebook (EDA + ML + App generation)
├── requirements.txt                        ← Python dependencies
├── StudentCareerSuccess_ProjectReport.docx ← Complete project report
├── README.md                               ← This file
├── student_career_success_dataset.csv      ← Dataset (50,000 records)
└── models/                                 ← Saved model artifacts (auto-generated on first run)
    ├── placement_classifier.pkl
    ├── salary_regressor.pkl
    └── model_meta.json
```

---

## ⚙️ Setup & Run Instructions

### 1. Clone / Download the project

```bash
# If using git
git clone <your-repo-url>
cd student-career-success-prediction

# Or simply place all files in a folder
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`

> ⚠️ Make sure `student_career_success_dataset.csv` is in the **same directory** as `app.py`.

### 5. Run the Jupyter Notebook (optional)

```bash
jupyter notebook StudentCareerSuccess_ProjectName.ipynb
```

Run all cells top-to-bottom. The notebook will:
- Perform full EDA with charts
- Train and compare 4 classifiers + 4 regressors
- Save trained models to `models/`
- Write the complete `app.py` to disk

---

## 🤖 ML Models

### Classification — Placement Prediction

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | ~84% | ~0.91 |
| Decision Tree | ~88% | ~0.88 |
| **Random Forest** ✅ | **~93%** | **~0.97** |
| Gradient Boosting | ~92% | ~0.96 |

### Regression — Salary Prediction (Placed Students)

| Model | MAE | R² Score |
|-------|-----|---------|
| Linear Regression | ~$8,500 | ~0.72 |
| Ridge Regression | ~$8,400 | ~0.73 |
| Random Forest | ~$6,800 | ~0.84 |
| **Gradient Boosting** ✅ | **~$6,400** | **~0.86** |

---

## 🖥️ App Pages

### 🏠 Dashboard
- KPI cards: total students, placed count, placement rate, avg CGPA, avg salary
- Placement status distribution chart
- Placement rate by major
- CGPA vs placement histogram
- Salary by company tier
- Correlation heatmap
- Raw data explorer

### 🔮 Predict
- 22-field input form covering all student attributes
- **Placement prediction badge** (✅ PLACED / ❌ NOT PLACED)
- **Placement probability** with visual gauge bar
- **Estimated starting salary** (USD)
- **Personalized career recommendations** based on weak areas

### 📊 Model Info
- Live performance metrics (Accuracy, AUC-ROC, R², MAE)
- Top 15 feature importance chart
- Model architecture summary table

---

## ✨ Key Features

- ✅ **Single file** — both ML backend and Streamlit UI in `app.py`
- ✅ **Cached training** — models train once (`@st.cache_resource`), predict instantly
- ✅ **Feature engineering** — Skill_Score, Activity_Index, Academic_Intensity
- ✅ **Personalized recommendations** — actionable tips for each student
- ✅ **Full EDA** — 6 visualizations covering all key trends
- ✅ **Dual prediction** — classification (placement) + regression (salary)

---

## 🏗️ Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| `Skill_Score` | Weighted avg of 5 skill scores | Composite employability score |
| `Activity_Index` | Projects + Certifications + Hackathons | Extracurricular engagement depth |
| `Academic_Intensity` | CGPA × Study_Hours / 10 | GPA quality × effort combined |

---

## 📊 Top Predictors of Placement

1. Employability_Score
2. Interview_Score
3. CGPA
4. Internships
5. Skill_Score (engineered)
6. Activity_Index (engineered)
7. Resume_Score
8. Programming_Skill

---

## 📄 License

This project is submitted as an academic project for educational purposes.

---

## 👤 Author

**YourName**  
Dataset: `student_career_success_dataset.csv` (50,000 records, 29 features)  
Year: 2025
