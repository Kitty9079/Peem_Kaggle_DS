# 🏦 Credit Risk — Loan Default Prediction

> Module 5 Assignment — Machine Learning Pipeline  
> **Author:** Kantinan Sukkert | Risk Officer, BAAC  
> **Date:** June 2026

## 📊 Project Overview

End-to-end ML pipeline สำหรับทำนายความน่าจะเป็นที่ลูกค้าจะเกิด **90-day delinquency** ภายใน 2 ปี โดยใช้ Give Me Some Credit dataset จาก Kaggle

## 🎯 Goals

- ✅ พัฒนาและเปรียบเทียบ ≥3 โมเดล (Logistic+Lasso, Random Forest, LightGBM)
- ✅ ใช้ Stratified 5-fold Cross-Validation
- ✅ ประเมินด้วย industry metrics: ROC-AUC, Gini, KS
- ✅ Explainability ด้วย SHAP
- ✅ ส่งมอบ notebook + รายงาน

## 📂 Project Structure

```
CREDITRISK/
├── data/                          # Raw data จาก Kaggle
│   ├── cs-training.csv            # Training set (with target)
│   ├── cs-test.csv                # Test set (no target)
│   ├── Data Dictionary.xls        # Feature description
│   └── sampleEntry.csv            # Submission format
│
├── notebook/                      # Jupyter notebooks
│   └── 01_loan_default.ipynb      # Main pipeline
│
├── report/                        # Output artifacts
│   ├── figures/                   # Plots
│   ├── models/                    # Saved .pkl files
│   ├── results/                   # CSV results
│   └── final_report.md            # 3-5 page report
│
├── requirements.txt
└── README.md
```

## 🛠️ Setup

```bash
# Create virtual env
python -m venv .venv
.venv\Scripts\activate

# Install
pip install -r requirements.txt

# Launch notebook
jupyter notebook
```

## 📊 Dataset

- **Source:** [Kaggle - Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)
- **Size:** 150,000 customers × 11 features
- **Target:** `SeriousDlqin2yrs` (90-day delinquency within 2 years)
- **Default rate:** ~6.7% (imbalanced)

## 🏆 Models Used

| Model | Why |
|---|---|
| **Logistic + Lasso (L1)** | Interpretable baseline + auto feature selection |
| **Random Forest** | Non-linear, robust to outliers |
| **LightGBM** | State-of-the-art for credit scoring |

## 📈 Industry Metrics

- **ROC-AUC** — Overall discrimination
- **Gini = 2×AUC − 1** — Banking standard
- **KS statistic** — Max separation (regulator metric)
- **PR-AUC, F1, Recall** — For imbalanced data

## 📜 License

For educational purpose only.