# Customer Churn Prediction

A machine learning project focused on predicting customer churn using the Telco Customer Churn dataset and a Decision Tree Classifier.

The project covers data preprocessing, categorical encoding, model selection, cross-validation, and performance evaluation.

## Project Overview

Customer churn prediction helps identify customers who are likely to discontinue a service. This project uses customer demographics, account information, subscribed services, and billing details to predict whether a customer will churn.

## Dataset

- Original records: 7,043
- Original columns: 21
- Target variable: `Churn`
- Identifier: `customerID` (removed before training)
- Training samples: 5,634
- Test samples: 1,409

The target variable is encoded as:
- `No` → 0
- `Yes` → 1

## Machine Learning Pipeline

1. Data loading
2. Data preprocessing
   - Convert `TotalCharges` to numeric
   - Handle missing `TotalCharges` values
   - Remove customer identifiers
3. Train/test split using stratification
4. Numerical preprocessing using median imputation
5. Categorical preprocessing using most-frequent imputation and One-Hot Encoding
6. Decision Tree classification
7. Hyperparameter tuning with GridSearchCV
8. Model evaluation

All preprocessing steps are integrated into a Scikit-learn Pipeline to ensure they are fitted independently within each cross-validation fold.

## Model Selection

A Decision Tree Classifier was evaluated using different `max_depth` values:

`[1, 2, 3, 4, 5, 7, 10, 15, 20]`

Model selection was performed using:
- 5-Fold Cross-Validation
- F1-score as the optimization metric
- GridSearchCV

**Best hyperparameter:**

| Parameter | Selected value |
|---|---:|
| Criterion | Gini |
| max_depth | 2 |
| CV folds | 5 |

Best mean Cross-Validation F1-score: **58.75%**

## Model Evaluation

The selected model was evaluated on the held-out test set.

| Metric | Score |
|---|---:|
| Accuracy | 75.09% |
| Precision | 52.73% |
| Recall | 59.36% |
| F1-score | 55.85% |
| ROC-AUC | 77.99% |

### Confusion Matrix

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| Actual No Churn | 836 | 199 |
| Actual Churn | 152 | 222 |

The model correctly identified 222 out of 374 customers who churned in the test set.

## Project Structure

```text
Customer-churn-prediction/
│
├── data/
│   └── raw/
│       ├── Telco.csv
│       └── sample.csv
│
├── notebooks/
│   ├── load_prc.py
│   └── prc_eda.py
│
├── reports/
│   ├── eda_report.md
│   └── figures/
│       ├── churn_by_gender.png
│       └── churn_distribution.png
│
├── src/
│   ├── data/
│   │   ├── Data_cleaning/
│   │   │   └── Cleaner.py
│   │   └── ingestion/
│   │       └── load_data.py
│   │
│   ├── exploration/
│   │   └── eda.py
│   │
│   ├── features/
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── decision_tree.py
│   │   └── model_training.py
│   │
│   └── visualization/
│       └── figures.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/A-heidari/Customer-churn-prediction.git
cd Customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Run the Decision Tree model from the project root:

```bash
python -m src.models.decision_tree
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Author

**Alireza Heidari**

AI / Machine Learning Engineering Student

[GitHub](https://github.com/A-heidari)
