# 🎓 Student Placement Prediction System using Machine Learning

An end-to-end, production-grade Machine Learning and Flask web intelligence platform that analyzes student academic records, technical aptitude, and experiential profiles to predict campus placement outcomes (`Placed` vs. `Not Placed`) and benchmark **5 classification algorithms**.

---

## 📌 1. Project Overview & Objectives

In higher education institutions, campus placement success is a vital indicator of student readiness and academic excellence. This project builds a complete, data-driven machine learning system that:
1. **Predicts Placement Outcome**: Accurately forecasts whether an individual student will be placed or not placed based on comprehensive academic and skill parameters.
2. **Benchmarks 5 Classification Algorithms**: Trains and evaluates **Logistic Regression**, **Decision Tree Classifier**, **Random Forest Classifier**, **K-Nearest Neighbors (KNN)**, and **Support Vector Machine (SVM)** on the exact same holdout test cohort.
3. **Explains Predictions & Recommends Improvements**: Provides prediction confidence percentages, class probabilities, feature impact breakdowns, and actionable feedback.
4. **Delivers a Modern Full-Stack Web Application**: Features an interactive dashboard, dynamic input form generated directly from the dataset schema, benchmark analytics, and Matplotlib/Seaborn visualization galleries.

---

## 🛠️ 2. Technologies & Architecture

- **Backend / Web Framework**: Python 3.12, Flask 3.0
- **Machine Learning & Preprocessing**: Scikit-Learn, Pandas, NumPy, Joblib, SciPy, Imbalanced-Learn
- **Visualizations**: Matplotlib, Seaborn, Chart.js
- **Frontend Design**: HTML5, CSS3 (Modern Glassmorphism & Custom Palettes), Bootstrap 5.3, FontAwesome 6

---

## 📊 3. Dataset Description (`student_placement.csv`)

The machine learning pipeline operates strictly on the actual dataset (`student_placement.csv`), containing **1,000 student records** with **10 input dimensions** and **1 binary target variable**:

| Feature Name | Type | Min | Max | Mean | Description |
|---|---|---|---|---|---|
| `CGPA` | Numeric (float) | 5.52 | 9.80 | 7.61 | Cumulative Grade Point Average (0–10 scale) |
| `10th_Percentage` | Numeric (float) | 45.0% | 100.0% | 72.64% | Secondary School academic percentage |
| `12th_Percentage` | Numeric (float) | 45.0% | 100.0% | 70.33% | Higher Secondary School academic percentage |
| `Internships` | Numeric (int) | 0 | 3 | 1.10 | Number of completed industry internships |
| `Aptitude_Score` | Numeric (float) | 35.0 | 100.0 | 68.79 | Quantitative & logical aptitude test score |
| `Communication_Score` | Numeric (float) | 4.0 | 10.0 | 7.31 | Soft skills & verbal proficiency rating (0–10) |
| `Technical_Skill_Score` | Numeric (float) | 3.3 | 10.0 | 6.97 | Programming & core domain evaluation (0–10) |
| `Projects` | Numeric (int) | 0 | 5 | 2.61 | Completed software/hardware projects |
| `Certifications` | Numeric (int) | 0 | 6 | 2.68 | Professional credentials / certifications |
| `Backlogs` | Numeric (int) | 0 | 3 | 0.56 | Current active academic backlogs |
| **`Placement` (Target)** | Categorical | - | - | - | **`Placed` (737 / 73.7%)**, **`Not Placed` (263 / 26.3%)** |

---

## ⚙️ 4. Data Preprocessing & Pipeline Architecture

1. **Integrity & Cleaning**: Automated duplicate removal and missing value imputation.
2. **Target Encoding**: Mapped binary labels (`Placed` $\rightarrow$ 1, `Not Placed` $\rightarrow$ 0).
3. **Data Leakage Prevention**: `ColumnTransformer` and `StandardScaler` are fitted exclusively on the 80% training partition (`X_train_raw`) and persisted as `preprocessing_pipeline.pkl`.
4. **Stratified Split**: 80% Training (800 students) and 20% Holdout Testing (200 students) with exact class ratio preservation.

---

## 🤖 5. Machine Learning Algorithms & Benchmark Results

All five algorithms were trained on the exact same scaled training data and evaluated on the same 200-sample stratified test set and 5-Fold Cross-Validation:

| Rank | Algorithm | Accuracy | Precision | Recall | F1 Score | ROC AUC | 5-Fold CV Accuracy | Status |
|---|---|---|---|---|---|---|---|---|
| 🥇 | **Logistic Regression** | **85.50%** | **89.33%** | **91.16%** | **90.24%** | **0.9127** | **88.75% ± 1.8%** | **Selected Champion** |
| 🥈 | **Support Vector Machine (SVM)** | 84.00% | 88.08% | 90.48% | 89.26% | 0.8673 | 86.62% ± 2.3% | Tested |
| 🥉 | **K-Nearest Neighbors (KNN)** | 84.00% | 88.59% | 89.80% | 89.19% | 0.8706 | 85.25% ± 1.9% | Tested |
| 4 | **Random Forest Classifier** | 82.50% | 88.36% | 87.76% | 88.05% | 0.8931 | 88.62% ± 1.6% | Tested |
| 5 | **Decision Tree Classifier** | 80.50% | 86.49% | 87.07% | 86.78% | 0.8046 | 84.75% ± 1.8% | Tested |

> **Model Selection Rationale**: **Logistic Regression** achieved the highest F1 Score (90.24%), generalization accuracy (85.50%), and ROC AUC (0.9127), making it the champion default production model.

---

## 📁 6. Project Directory Structure

```
student Placement Prediction System using Machine Learning/
│
├── student_placement.csv          # Official dataset (1,000 records)
├── app.py                         # Flask web application & REST API server
├── train_model.py                 # Multi-model training & evaluation pipeline
├── preprocessing.py               # Data loading, validation, scaling & splitting
├── prediction.py                  # Model inference engine & student feedback
├── test_system.py                 # Automated unit and integration test suite
├── requirements.txt               # Python package dependencies
├── README.md                      # Comprehensive project documentation
│
├── models/                        # Serialized models & metadata (Joblib)
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── support_vector_machine.pkl
│   ├── k_nearest_neighbors.pkl
│   ├── decision_tree.pkl
│   ├── preprocessing_pipeline.pkl
│   └── metadata.json
│
├── outputs/                       # Evaluation reports & charts
│   ├── model_comparison.csv
│   ├── evaluation_metrics.json
│   ├── placement_distribution.png
│   ├── cgpa_distribution.png
│   ├── correlation_heatmap.png
│   ├── accuracy_comparison.png
│   ├── precision_comparison.png
│   ├── recall_comparison.png
│   ├── f1_comparison.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   ├── all_confusion_matrices.png
│   └── confusion_matrices/
│       ├── cm_logistic_regression.png
│       ├── cm_random_forest.png
│       ├── cm_support_vector_machine.png
│       ├── cm_k_nearest_neighbors.png
│       └── cm_decision_tree.png
│
├── templates/                     # Jinja2 HTML5 dynamic UI templates
│   ├── base.html
│   ├── index.html
│   ├── predict.html
│   ├── result.html
│   ├── comparison.html
│   ├── visualizations.html
│   ├── 404.html
│   └── 500.html
│
└── static/                        # Frontend assets
    ├── css/
    │   └── style.css              # Custom responsive stylesheet
    ├── js/
    │   └── dashboard.js           # Chart.js dashboards & form interactions
    └── images/                    # Synchronized high-resolution chart assets
```

---

## 🚀 7. Installation & Execution Guide

### Prerequisites
- Python 3.9+ (Python 3.12 recommended)
- Windows, macOS, or Linux

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train All Models & Generate Visualizations
```bash
python train_model.py
```

### Step 3: Run Automated Verification Tests
```bash
python test_system.py
```

### Step 4: Launch the Web Application
```bash
python app.py
```

### Step 5: Access the Web Interface
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🌐 8. Web Application Features & Walkthrough

1. **Executive Dashboard (`/`)**:
   - Live KPI counters (Total Students, Placed, Not Placed, Placement Rate).
   - Interactive Chart.js cohort distribution donut chart.
   - Quick model benchmark leaderboard with instant prediction action.
2. **Prediction Form (`/predict`)**:
   - Form fields dynamically constructed based on `student_placement.csv`.
   - **1-Click Profile Presets**: *Top Performer*, *Average Student*, and *At-Risk Profile* for instant testing.
   - Algorithm selector allowing user to test any of the 5 algorithms.
3. **Prediction Result & Report (`/result`)**:
   - Distinct **PLACED** (Green) or **NOT PLACED** (Red) visual result banner.
   - Confidence percentage and probability distribution.
   - Comparative student score breakdown vs cohort averages.
   - Actionable recommendations and printable summary view.
4. **Model Comparison Hub (`/comparison`)**:
   - Interactive multi-metric comparison bar & radar charts.
   - Confusion matrix metric breakdowns (TP, TN, FP, FN) for each classifier.
5. **Visual Analytics Gallery (`/visualizations`)**:
   - Full-resolution Matplotlib and Seaborn analytics suite with modal inspection.
6. **REST API Endpoints**:
   - `POST /api/predict`: JSON payload prediction.
   - `GET /api/metrics`: JSON model benchmark statistics.

---

## 🔒 9. Error Handling & Validation

- **Missing File Safeguards**: Gracefully detects missing dataset or models and auto-initiates training.
- **Input Validation**: Rejects empty, negative, or non-numeric inputs and provides user-friendly error banners.
- **Predictive Confidence Disclaimer**: Transparently reminds users that predictions represent statistical approximations rather than absolute hiring guarantees.

---

## 📜 10. License & Maintenance

Developed as an open-source educational machine learning system. Feel free to extend and integrate with institutional student management systems!
