from src.data.ingestion.load_data import load_data

df = load_data("Telco.csv")
print(df.head())

# ---------------------------------------------
# Basic Infromation
# ---------------------------------------------

print("Shape: ")
print(df.shape)

print("\nColumns: ")
print(df.columns)

print("\nInformation: ")
df.info()

# ---------------------------------------------
# Missing value 
# ---------------------------------------------

print("\nMissing value: ")
print(df.isnull().sum())


# ---------------------------------------------
# Duplicated
# ---------------------------------------------

print("\nDuplicated: ")
print(df.duplicated().sum())

# ---------------------------------------------
# statistic
# ---------------------------------------------

print("\nStatistic: ")
print(df.describe())

# ---------------------------------------------
# Data Type
# ---------------------------------------------

print("\nData type: ")
print(df.dtypes)



print("\n--- gender and Churn mean")
print(df.groupby("gender")["Churn"].value_counts(normalize=True) * 100)
print("\n--- Contract and Churn mean")
print(df.groupby("Contract")["Churn"].value_counts(normalize=True) * 100)

