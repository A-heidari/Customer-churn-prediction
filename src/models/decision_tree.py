
import pandas as pd

from src.data.ingestion.load_data import load_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# -----------------------------------------
# 1. Load and prepare data
# -----------------------------------------

df = load_data("Telco.csv")

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(0)

df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

df = df.drop(columns=["customerID"])

X = df.drop(columns=["Churn"])
y = df["Churn"]


# -----------------------------------------
# 2. Train / Test Split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# -----------------------------------------
# 3. Numerical Pipeline
# -----------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])


# -----------------------------------------
# 4. Categorical Pipeline
# -----------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# -----------------------------------------
# 5. Feature Lists
# -----------------------------------------

numerical_features = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

categorical_features = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod"
]


# -----------------------------------------
# 6. Column Transformer
# -----------------------------------------

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])


# -----------------------------------------
# 7. Complete ML Pipeline
# -----------------------------------------

model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    ))
])


# -----------------------------------------
# 8. Hyperparameter Search
# -----------------------------------------

param_grid = {
    "tree__max_depth": [1, 2, 3, 4, 5, 7, 10, 15, 20]
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)


# -----------------------------------------
# 9. Cross-Validation Results
# -----------------------------------------

results = pd.DataFrame(grid_search.cv_results_)

results = results[[
    "param_tree__max_depth",
    "mean_train_score",
    "mean_test_score",
    "std_test_score",
    "rank_test_score"
]]

results = results.sort_values("rank_test_score")

print("\n--- CROSS VALIDATION RESULTS ---")
print(results.to_string(index=False))


# -----------------------------------------
# 10. Best Model
# -----------------------------------------

best_model = grid_search.best_estimator_

print("\n--- BEST MODEL ---")
print("Best Parameters:", grid_search.best_params_)
print("Best CV F1:", grid_search.best_score_)


# -----------------------------------------
# 11. Final Test Evaluation
# -----------------------------------------

y_test_pred = best_model.predict(X_test)

y_test_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred, zero_division=0)
recall = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

cm = confusion_matrix(y_test, y_test_pred)


# -----------------------------------------
# 12. Print Evaluation
# -----------------------------------------

print("\n--- FINAL TEST RESULTS ---")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)
print("ROC-AUC  :", roc_auc)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))
