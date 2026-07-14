from src.data.ingestion import load_data

df = load_data("Telco.csv")

print("\n--- BEFORE FEATURE ENGINEERING ---")
print(df.head())

# feature_engineering 
# 1. TIMING BEHAVIOR OF THE CUSTOMER
df["is_new_customer"] = (df["tenure"] < 12).astype(int) 

# 2. COST BEHAVIOR OF THE CUSTOMER
df["cost_per_tenure"] = df["monthly_charges"] / (df["tenure"] + 1)

# check result

print("\n--- AFTER FEATURE ENGINEERING ---")
print(df.head())

