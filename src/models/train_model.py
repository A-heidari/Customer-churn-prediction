from src.data.ingestion.load_data import load_data


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import pandas as pd

# load data

df = load_data("Telco.csv")
df.columns = df.columns.str.lower()

df["churn"] = df["churn"].astype(str).str.strip().str.lower()
df["churn"] = df["churn"].map({"yes": 1, "no": 0})
df = pd.get_dummies(df, drop_first=True)

# feature and target 
x = df.drop("churn", axis=1)
y = df["churn"]

# Train/test split 
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# model 
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10,n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    results[name] = acc

    print(name, acc)

print("\nBEST MODEL:")
print(max(results, key=results.get))