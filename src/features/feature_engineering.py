import pandas as pd
from src.data.ingestion.load_data import load_data
from src.data.Data_cleaning.Cleaner import clean_data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new features and encode categorical columns."""
    df = df.copy()

    # New features
    df["num_services"] = (df[[
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]] == "Yes").sum(axis=1)

    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)

    # Encode target and binary columns
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

    binary_cols = [
        "Partner", "Dependents", "PhoneService", "MultipleLines",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "PaperlessBilling",
    ]
    df[binary_cols] = df[binary_cols].apply(lambda col: col.map({"Yes": 1, "No": 0}))

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)

    return df


if __name__ == "__main__":
    df = load_data("Telco.csv")
    df = clean_data(df)
    df = build_features(df)

    print("Shape:", df.shape)
    print(df.head())