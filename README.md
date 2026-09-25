# Customer Churn Prediction

A machine learning project for predicting customer churn using the **Telco Customer Churn dataset** and a **Decision Tree Classifier**.

## Pipeline

```text
Data Loading
    ↓
Data Cleaning
    ↓
Feature Engineering
    ↓
Train/Test Split
    ↓
Decision Tree
    ↓
Model Evaluation
```

## Features

* Data cleaning and missing-value handling
* Categorical encoding
* Feature engineering
* Stratified train/test split
* Decision Tree classification
* Model evaluation

### Engineered Features

* `num_services`
* `avg_monthly_spend`
* `is_month_to_month`
* `has_family`

## Baseline Results

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 72.03% |
| Precision | 47.36% |
| Recall    | 50.54% |
| F1 Score  | 48.89% |
| ROC-AUC   | 65.26% |

These are the results of the initial Decision Tree baseline before hyperparameter tuning and pruning.

## Project Structure

```text
Customer-churn-prediction/
├── data/
├── reports/
└── src/
    ├── data/
    ├── exploration/
    ├── features/
    ├── models/
    └── visualization/
```

## Run

From the project root:

```bash
python -m src.models.model_training
```

## Tech Stack

* Python
* Pandas
* Scikit-learn
* NumPy
* Matplotlib

## Next Steps

* Hyperparameter tuning
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* Cost-Complexity Pruning (`ccp_alpha`)
* `GridSearchCV`
