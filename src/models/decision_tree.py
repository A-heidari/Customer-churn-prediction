import pandas as pd

from src.data.ingestion.load_data import load_data
from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


from sklearn.metrics import (accuracy_score, confusion_matrix,recall_score, precision_score, f1_score)


from sklearn.pipeline import Pipeline

df = load_data("Telco.csv")

# TotalCharges → numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing TotalCharges
df = df.dropna(subset=["TotalCharges"])

# Churn → numerical
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Remove customer identifier
df = df.drop(columns=["customerID"])


print(df.shape)
print(df["Churn"].value_counts())
print(df["Churn"].dtype)



# ----------------------------------------------

X = df.drop(columns=["Churn"])

y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.8,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)

print(y_train.value_counts())
print(y_test.value_counts())



# -----------------------------------------
# Numerical pipeline
# -----------------------------------------

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

# -----------------------------------------
# Categoriacal pipeline
# -----------------------------------------

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# -----------------------------------------
# Feature lists
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
# preprocessor 
# -----------------------------------------

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# -----------------------------------------
# preprocessor 
# -----------------------------------------

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

print("Number of features:", len(feature_names))
print(feature_names)


# -----------------------------------------
# Decision tree
# -----------------------------------------

depths = [1, 2, 3, 4, 5, 7, 10, 15, 20]

for depth in depths: 


    tree =  DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train_processed, y_train)

    print("Tree depth: ", tree.get_depth())
    print("Number of leaves: ", tree.get_n_leaves())


# -----------------------------------------
# train prediction
# -----------------------------------------

    y_train_pred = tree.predict(X_train_processed)

# -----------------------------------------
# TEST PREDICTION
# -----------------------------------------

    y_test_pred = tree.predict(X_test_processed)

# -----------------------------------------
# TRAIN Evaluation
# -----------------------------------------

    # accuracy = accuracy_score(y_train, y_train_pred)
    # recall = recall_score(y_train, y_train_pred)
    # precision = precision_score(y_train, y_train_pred)
    # f1 = f1_score(y_train, y_train_pred)
    # cm = confusion_matrix(y_train, y_train_pred)


    # print("\n--- EVALUATION ---")
    # print("ACCURACY : ", accuracy)
    # print("RECALL :", recall)
    # print("PRECISION: ", precision)
    # print("F1 : ", f1)
    # print(cm)

# -----------------------------------------
# TEST EVALUATION 
# -----------------------------------------

    accuracy = accuracy_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred)


    print("\n--- EVALUATION ---")
    print("ACCURACY : ", accuracy)
    print("RECALL :", recall)
    print("PRECISION: ", precision)
    print("F1 : ", f1)
    print(cm)
  

# -----------------------------------------
# Cross validation
# -----------------------------------------

    model_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    ))
])


    scores = cross_val_score(
    model_pipeline,
    X,
    y,
    cv=5,
    scoring="f1"
)