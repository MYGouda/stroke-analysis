# Stroke Prediction – Classification Problem

**Course:** AI Programming  
**Student:** Marina Youssef Gouda Ramis (ID: 3059199)

## Overview

This project builds a machine learning pipeline to predict stroke risk from patient health records. It addresses the UAE's high prevalence of the three strongest stroke predictors identified in the analysis: elevated glucose levels, hypertension, and high BMI.

The dataset contains **5,110 patient records** with 11 features, sourced from electronic health records originally compiled by McKinsey & Company and available on Kaggle.

**Dataset:** [Stroke Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data)

---

## Key Challenges

| Challenge | Detail |
|---|---|
| **Class imbalance** | Only 249 stroke cases (~4.9%) vs 4,861 non-stroke cases — accuracy alone is a misleading metric |
| **Missing BMI values** | 201 records (~3.93%) have null BMI — dropping them loses information |

---

## Project Structure

```
3059199_Assignment.ipynb   # Main notebook with full analysis
```

---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Shape, data types, null value detection
- Stroke rate by age, glucose level, gender, smoking status, and work type
- Pairplot of key numeric features coloured by stroke outcome

### 2. Data Preprocessing
- **Missing BMI** imputed with median
- **Categorical features** (gender, smoking status, work type, marital status, residence type) encoded via one-hot encoding
- **Train/test split:** 60% train / 40% test (`random_state=42`)
- **Class imbalance** addressed with SMOTE on the training set

### 3. Models

| Model | Role | Notes |
|---|---|---|
| Random Forest | Baseline | Poor recall on stroke class — missed 113 of 114 real cases |
| XGBoost | Improved | `scale_pos_weight=20`, `n_estimators=200`, `max_depth=4`, `learning_rate=0.05` |

### 4. Evaluation
- Confusion matrix
- Classification report (precision, recall, F1)
- ROC-AUC score and curve

---

## Results

### Baseline – Random Forest

| | Predicted No Stroke | Predicted Stroke |
|---|---|---|
| **Actual No Stroke** | 1927 (TN) | 3 (FP) |
| **Actual Stroke** | 113 (FN) | 1 (TP) |

High accuracy, but clinically dangerous — 113 out of 114 real stroke patients were missed.

### Improved – XGBoost

| | Predicted No Stroke | Predicted Stroke |
|---|---|---|
| **Actual No Stroke** | 1544 (TN) | 386 (FP) |
| **Actual Stroke** | 47 (FN) | 67 (TP) |

**ROC-AUC: ~0.78** — significant improvement in recall for the stroke class.

---

## Key Findings

- **Age** is the single strongest predictor of stroke (mean age of stroke patients: ~68 years)
- **Average glucose level** (mean ~133 for stroke cases) and **hypertension** are the next most influential features
- **XGBoost outperformed Random Forest** on AUC-ROC and recall by capturing non-linear feature interactions
- SMOTE was essential to make the minority class learnable

---

## Limitations

1. **Confidential data provenance** — the originating institution is undisclosed, limiting assessment of sample representativeness
2. **Geographic generalisability** — the dataset's population is unspecified; direct applicability to the UAE requires validation on local clinical data
3. **Small positive class** — only 249 real stroke cases; SMOTE generates synthetic samples but cannot substitute for genuine patient data

---

## Dependencies

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn   # SMOTE
xgboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```
