# Credit Risk Scoring Model

A machine learning project to predict the probability of a borrower 
experiencing financial distress within two years, based on the 
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset.

## Overview

This project mirrors real-world credit risk workflows used by financial 
institutions to make lending decisions. It covers the full data science 
pipeline from raw data to a validated model.

## Dataset

- 150,000 individuals with 10 financial features
- Target: `SeriousDlqin2yrs` — whether a borrower defaulted within 2 years
- Class imbalance: 93.3% non-default, 6.7% default

## Project Structure

- Data cleaning & outlier handling
- Exploratory data analysis (EDA)
- Feature engineering
- Model training & evaluation (Logistic Regression + XGBoost)

## Key Results

| Model | AUC-ROC |
|-------|---------|
| Logistic Regression (baseline) | 0.8556 |
| XGBoost | 0.8667 |

## Feature Engineering

Created 4 new features from the raw data:
- `TotalLatePayments` — sum of all late payment categories 
  (became the single strongest predictor with importance score of 0.70)
- `IncomePerDependent` — monthly income adjusted for number of dependents
- `DebtToIncome` — absolute debt burden derived from debt ratio and income
- `ZeroIncome` — binary flag for borrowers with no reported income

## Key Findings

- Late payment history is by far the strongest predictor of default
- High revolving credit utilization is the second strongest signal
- The model achieves 78% recall on the default class, prioritising the 
  detection of actual defaulters over minimising false alarms — consistent 
  with real-world credit risk appetite
- Class imbalance was handled using `scale_pos_weight` in XGBoost

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

## Author

Financial Engineering student & Data Scientist with experience in 
insurance risk modeling (Achmea) and data governance (NOC*NSF).

