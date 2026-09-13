import pandas as pd
from pathlib import Path


# Find project root
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = BASE_DIR / "data" / "raw" / "fake_job_postings.csv"


# Load dataset
df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("SIGNALGUARD - DATASET INSPECTION")
print("=" * 60)

print(f"\nDataset path:")
print(DATA_PATH)

print(f"\nDataset shape:")
print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for column in df.columns:
    print(f"- {column}")


print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum().sort_values(ascending=False))


print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)

print(df["fraudulent"].value_counts())


print("\nTarget percentage:")

print(
    df["fraudulent"]
    .value_counts(normalize=True)
    .mul(100)
)


print("\n" + "=" * 60)
print("DUPLICATES")
print("=" * 60)

print(f"Duplicate rows: {df.duplicated().sum()}")


print("\n" + "=" * 60)
print("FIRST 3 ROWS")
print("=" * 60)

print(df.head(3).to_string())