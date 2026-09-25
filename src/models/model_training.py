from src.data.ingestion.load_data import load_data
from src.data.Data_cleaning.Cleaner import clean_data
from src.features.feature_engineering import build_features

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------
# Load
# ---------------------------------------------

df = load_data("Telco.csv")


# ---------------------------------------------
# Cleaning
# ---------------------------------------------

df = clean_data(df)


# ---------------------------------------------
# Feature Engineering
# ---------------------------------------------

df = build_features(df)


# ---------------------------------------------
# Separate X and y
# ---------------------------------------------

X = df.drop(columns=["Churn"])
y = df["Churn"]


# ---------------------------------------------
# Train / Test Split
# ---------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------------
# Decision Tree
# ---------------------------------------------

model = DecisionTreeClassifier(
    random_state=42
)

model.fit(X_train, y_train)


# ---------------------------------------------
# Prediction
# ---------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ---------------------------------------------
# Evaluation
# ---------------------------------------------

print("\n--- Model Performance ---")

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("ROC-AUC  :", roc_auc_score(y_test, y_prob))


print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))


print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))