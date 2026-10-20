# MindCare AI — Mental Health Risk Screening and Wellness Support System

> **B.Tech Final-Year Capstone Project**  
> *Department of Computer Science & Engineering*  
> *An Educational Healthcare Machine Learning Web Application*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)](https://getbootstrap.com/)
[![License](https://img.shields.io/badge/Academic-Final--Year%20Project-green.svg)]()

---

## 1. Project Overview

**MindCare AI** is a full-stack, local-first web application designed to help university students and technology professionals evaluate self-reported wellbeing indicators and identify early mental health strain patterns.

The system uses a supervised **Random Forest Classifier** trained on the Open Sourcing Mental Illness (OSMI) Mental Health in Tech benchmark dataset ($N = 1,259$).

### Key Capabilities
- **Multi-Step Screening Wizard:** 5-step guided questionnaire with live validation, progress bar, and answer review.
- **AI-Assisted Screening Indicator:** Calibrated probabilistic risk tiering (Low, Moderate, Elevated) powered by an ensemble model.
- **Privacy-by-Design:** 100% anonymous screening by default (no names, emails, phone numbers, or external tracking).
- **Longitudinal Dashboard:** Interactive historical charts powered by Chart.js with explicit consent management and one-click data erasure.
- **Verified Wellness Resources:** Box-breathing tools, sleep hygiene guidance, and direct access to 24/7 verified crisis helplines (Tele-MANAS: **14416**, KIRAN: **1800-599-0019**).
- **Printable PDF Export:** Formatted `@media print` summary report for student records or professional consultation.
- **Admin & Model Insights:** Transparent model telemetry, Gini feature importance charts, and confusion matrix analytics.

> [!IMPORTANT]
> **Ethical & Medical Disclaimer:** MindCare AI is an educational screening prototype and **not a clinical diagnostic tool**. It does not diagnose mental illness or replace qualified psychiatric/psychological evaluation.

---

## 2. Technology Stack

- **Frontend:** HTML5, CSS3, JavaScript (ES6+), Bootstrap 5.3, Bootstrap Icons, Chart.js 4.4.
- **Backend:** Python 3.10+, Flask 3.1.
- **Machine Learning:** Scikit-Learn, Pandas, NumPy, Joblib.
- **Primary Model:** Random Forest Classifier (150 estimators, balanced class weights).
- **Database:** SQLite3 (parameterized SQL queries, PBKDF2 password hashing).
- **Test Suite:** Python standard `unittest` framework (17 automated unit and integration tests).

---

## 3. Machine Learning Architecture & Evaluation

The pipeline utilizes a scikit-learn `Pipeline` and `ColumnTransformer` to guarantee zero data leakage:

```
[Raw Inputs] 
     ├── Age (Numeric) ──────────> SimpleImputer(median) ──> StandardScaler() ──┐
     └── 7 Categoricals ─────────> SimpleImputer(const)   ──> OneHotEncoder()  ──┴──> RandomForestClassifier (150 trees)
```

### Actual Evaluated Metrics (Held-Out 25% Test Split)
- **Test Accuracy:** **$79.68\%$**
- **Test Precision:** **$76.54\%$**
- **Test Recall (Sensitivity):** **$86.16\%$**
- **Test F1-Score:** **$81.07\%$**
- **Test ROC-AUC:** **$0.8754$**
- **5-Fold Cross-Validation Accuracy:** **$82.84\% \pm 0.69\%$**

---

## 4. Project Directory Structure

```
MindCare-AI/
├── app.py                      # Flask web application & API routes
├── train_model.py              # ML preprocessing and training pipeline
├── database.py                 # SQLite database models & query interface
├── requirements.txt            # Python dependencies
├── README.md                   # Setup guide and viva documentation
├── PROJECT_REPORT.md           # Comprehensive B.Tech project report
├── .gitignore                  # Git ignore rules
├── data/
│   └── mental_health.csv       # OSMI benchmark survey dataset
├── models/
│   ├── mental_health_model.pkl # Serialized scikit-learn pipeline
│   └── model_metrics.json      # Evaluated metrics, confusion matrix & feature importances
├── database/
│   └── app.db                  # Local demonstration SQLite database
├── templates/
│   ├── base.html               # Master layout with responsive navbar & footer
│   ├── index.html              # Landing page with hero & feature pillars
│   ├── questionnaire.html      # 5-step interactive screening wizard
│   ├── result.html             # Calibrated result dashboard & PDF export
│   ├── dashboard.html          # User history & Chart.js trend tracking
│   ├── resources.html          # Wellness tools & verified emergency helplines
│   ├── admin.html              # Admin telemetry & feature importance chart
│   ├── login.html              # Optional account sign-in
│   ├── register.html           # Optional account registration
│   └── error.html              # Custom 404, 500, and missing-model handlers
├── static/
│   ├── css/
│   │   └── style.css           # Custom modern healthcare palette & print styles
│   └── js/
│       └── main.js             # Stepper navigation, validation, live review
└── tests/
    ├── test_model.py           # Unit tests for ML pipeline & inference
    └── test_routes.py          # Integration tests for Flask endpoints
```

---

## 5. Local Setup & Execution Guide (Windows PowerShell)

Follow these exact steps to run the application on your laptop:

### Step 1: Open Terminal and Navigate to the Directory
```powershell
cd "C:\Users\jagad\OneDrive\Documents\Mental-Health-Prediction\MindCare-AI"
```

### Step 2: Set Up Python Virtual Environment
*(If not already created)*
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

*(If you already have a virtual environment in the parent folder)*:
```powershell
& "C:\Users\jagad\OneDrive\Documents\Mental-Health-Prediction\venv\Scripts\Activate.ps1"
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Train the Machine Learning Model
Run the reproducible pipeline to generate `models/mental_health_model.pkl` and `models/model_metrics.json`:
```powershell
python train_model.py
```

### Step 5: Start the Flask Web Application
```powershell
python app.py
```

### Step 6: Open the Application in Your Browser
Open any web browser and visit:
```
http://127.0.0.1:5000
```

---

## 6. Running Automated Tests

To execute the test suite (17 unit and integration tests):
```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Expected output:
```
Ran 17 tests in 0.232s

OK
```

---

## 7. Troubleshooting Common Issues

| Issue / Error | Cause | Solution |
| :--- | :--- | :--- |
| `Model file not found` / `Model Not Ready` | `mental_health_model.pkl` has not been generated yet. | Run `python train_model.py` in your terminal. |
| `ModuleNotFoundError: No module named 'flask'` | Virtual environment is not activated or dependencies missing. | Run `.\venv\Scripts\Activate.ps1` followed by `pip install -r requirements.txt`. |
| PowerShell Execution Policy error on `Activate.ps1` | Windows restriction on running local scripts. | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in your PowerShell session. |
| Port 5000 in use | Another service is using port 5000. | Change the port in `app.py`: `app.run(port=5001)` or terminate the existing process. |

---

## 8. Verification & Viva Talking Points

When presenting this project to an external examiner or viva panel:
1. **Explain the Target Variable:** Clarify that the target `treatment` in the OSMI dataset represents *treatment-seeking behavior* in real-world surveys, not a psychiatric diagnosis.
2. **Explain Why Random Forest Was Chosen:** Bagging reduces variance; decision trees handle discrete categorical survey variables well; ensemble voting prevents overfitting; MDI provides interpretable feature importances.
3. **Data Leakage Prevention:** All scalers and encoders are encapsulated inside `sklearn.pipeline.Pipeline` and fit only on the $75\%$ training set.
4. **Segregation of Lifestyle Context:** Explain that questions about daily stress sliders (1-10) and sleep quality are presented for student self-reflection and explicitly segregated from the model features so no unmodeled inputs are fed to the ML algorithm.
5. **Privacy by Design:** Emphasize that the database uses guest session tokens (UUIDs) without harvesting personal identifiers (no names, phone numbers, or emails).

---

## 9. License & Academic Attribution
Developed as an academic final-year B.Tech Computer Science & Engineering capstone project.  
Dataset provided by Open Sourcing Mental Illness (OSMI).

