# 🏦 Credit Score Classification — Random Forest Classifier

A machine learning project that classifies **Credit Mix** (credit score category) using a tuned Random Forest model on a real-world-style credit bureau dataset. Built as part of the ML coursework at Manav Rachna University.

---

## 📌 Project Overview

This notebook implements a full end-to-end ML pipeline on the **Credit Score Classification** dataset. The goal is to predict a customer's `Credit_Mix` label (`Good`, `Standard`, or `Bad`) based on 27 financial and behavioral features — covering income, debt, loan history, payment behavior, and more.

The pipeline follows a structured **bake-off methodology**:

1. **Dataset Loading & Preview** — shape inspection, column types, initial statistics
2. **Preprocessing Pipeline** — noise removal, imputation, feature engineering
3. **Hyperparameter Tuning Log** — systematic experiments across RF configurations
4. **Final Model Selection Leaderboard** — ranked by validation accuracy
5. **Best Model Analysis** — classification report, feature importance, confusion matrix

---

## 📂 Repository Structure

```
📦 credit-score-rf/
├── Random_Forest_Credit.ipynb   # Main notebook (full pipeline)
├── test.csv                     # Dataset (27 features, credit bureau records)
└── README.md
```

---

## 📊 Dataset

| Property | Detail |
|---|---|
| File | `test.csv` |
| Features | 27 columns |
| Target | `Credit_Mix` (Good / Standard / Bad) |
| Key Features | Age, Annual Income, Monthly Inhand Salary, Outstanding Debt, Credit History Age, Type of Loan, Payment Behaviour, Num of Delayed Payment, and more |

**Preprocessing steps applied:**
- Strip noise characters (`_`, `-`) and coerce to numeric
- Convert `Credit_History_Age` from `"X Years Y Months"` string format to total months
- Median imputation for numeric columns; mode imputation for categoricals
- Expand `Type_of_Loan` into binary dummy variables (maximizing features)
- Label encoding for all categorical targets
- `StandardScaler` normalization on key numeric columns
- Drop PII columns: `ID`, `Customer_ID`, `SSN`, `Name`, `Month`

---

## 🤖 Model & Experiments

Three Random Forest configurations were benchmarked:

| Exp ID | n_estimators | max_depth | Validation Accuracy |
|--------|-------------|-----------|---------------------|
| RF_01  | 100         | 10        | —                   |
| RF_02  | 200         | 20        | —                   |
| RF_03  | 300         | None      | —                   |

The best-performing configuration is automatically selected and re-trained for final evaluation.

---

## 📈 Outputs

- **Classification Report** — Precision, Recall, F1-Score per class
- **Top 15 Feature Importance Plot** — Horizontal bar chart of the most impactful features
- **Confusion Matrix** — Visual heatmap of predicted vs. actual labels

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numeric operations |
| `scikit-learn` | ML model, preprocessing, metrics |
| `matplotlib` | Visualization |
| `seaborn` | Styling |

---

## 🚀 Getting Started

**1. Clone the repository**
```bash
git clone git clone https://github.com/vanshbhatia3841-beep/Credit-Score-Classifier.git
cd Credit-Score-Classifier
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

**3. Run the notebook**
```bash
jupyter notebook Random_Forest_Credit.ipynb
```

Make sure `test.csv` is in the same directory as the notebook before running.

---

## 📝 Key Design Decisions

- **Feature Maximization** — `Type_of_Loan` is one-hot expanded rather than label-encoded, so every unique loan type contributes an independent predictive signal.
- **Stratified Split** — `train_test_split` uses `stratify=y` to preserve class balance across train/test sets.
- **Safe Type Handling** — The leaderboard pipeline explicitly recasts `n_estimators` back to `int` and uses `pd.isna()` to safely handle `None` max_depth values after DataFrame serialization.

---

## 👤 Author

**Vansh** — AI/ML, Manav Rachna University  
Course: Machine Learning (CSH217C)

---

## 📄 License

This project is for academic purposes. Dataset sourced from public credit classification benchmarks.
