import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.ingestion.load_data import load_data

# ریشه‌ی پروژه (src/visualization/figures.py -> سه سطح بالاتر)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = load_data("Telco.csv")

# Churn Distribution
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_distribution.png", dpi=150)
plt.close()

# Churn by Gender
pd.crosstab(df["gender"], df["Churn"]).plot(kind="bar")
plt.title("Churn by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "churn_by_gender.png", dpi=150)
plt.close()

print(f"Figures saved to: {FIGURES_DIR}")