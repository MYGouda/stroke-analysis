# Stroke Prediction – Classification Problem

## Overview

Supervised ML pipeline to predict stroke risk from routine clinical and demographic data. Directly relevant to the UAE, where the three strongest predictors found — elevated glucose, hypertension, and high BMI — are among the most prevalent health conditions. Data sourced from electronic health records originally compiled by McKinsey & Company.

**Dataset:** [Stroke Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

---

## At a Glance

| Metric | Value |
|---|---|
| Patient records | 5,110 |
| Features per record | 11 |
| Stroke cases | 249 (~4.9%) |
| Best AUC-ROC | ~0.78 (XGBoost) |

---

## Key Challenges

**Class imbalance** — 4,861 non-stroke vs 249 stroke cases. Accuracy alone is a useless metric; a model that always predicts "no stroke" scores 95%.

**Missing BMI values** — 201 records (~3.93%) have null BMI. Dropping them loses information; median imputation was applied instead.

---

## Methodology

### 1. EDA
Distribution of age, glucose, and BMI. Stroke rate by gender, smoking status, and work type. Pairplot of key numeric features coloured by stroke outcome.

### 2. Preprocessing
- Missing BMI imputed with median
- Categorical features (gender, smoking status, work type, marital status, residence type) one-hot encoded
- 60/40 train-test split (`random_state=42`)
- SMOTE applied to training set to address class imbalance

### 3. Modelling
| Model | Role |
|---|---|
| Random Forest | Baseline |
| XGBoost (`scale_pos_weight=20`, `n_estimators=200`, `max_depth=4`, `learning_rate=0.05`) | Improved |

### 4. Evaluation
Confusion matrix, classification report (precision, recall, F1), ROC-AUC score and curve.

---

## Model Comparison

### Baseline – Random Forest

| Metric | Result |
|---|---|
| True positives (stroke caught) | 1 |
| False negatives (missed) | 113 |
| AUC-ROC | — |
| Clinically safe? | No — 113 of 114 real stroke patients sent home undiagnosed |

### Improved – XGBoost ✓

| Metric | Result |
|---|---|
| True positives (stroke caught) | 67 |
| False negatives (missed) | 47 |
| AUC-ROC | ~0.78 |
| Clinically safe? | Significantly better |

---

## Key Findings

- **Age** is the single strongest predictor — mean age of stroke patients is ~68 years
- **Average glucose level** (mean ~133 for stroke cases) and **hypertension** are the next most influential features
- **XGBoost outperformed Random Forest** on recall and AUC-ROC by capturing non-linear feature interactions
- **SMOTE** was essential to make the minority class learnable

---

## Limitations

1. **Confidential data provenance** — originating institution undisclosed; representativeness cannot be fully assessed
2. **Geographic generalisability** — dataset population unspecified; UAE applicability requires validation on local clinical data
3. **Small positive class** — only 249 real stroke cases; SMOTE generates synthetic samples but cannot substitute for genuine patient data

---

## Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```
