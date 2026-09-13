import pandas as pd
import re

# =========================
# Load dataset
# =========================

DATA_PATH = "data/raw/fake_job_postings.csv"

df = pd.read_csv(DATA_PATH)

print("\n" + "=" * 60)
print("SIGNALGUARD - DATA QUALITY ANALYSIS")
print("=" * 60)

print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")


# =========================
# 1. Target distribution
# =========================

print("\n" + "-" * 60)
print("1. TARGET DISTRIBUTION")
print("-" * 60)

target_counts = df["fraudulent"].value_counts()
target_percent = df["fraudulent"].value_counts(normalize=True) * 100

print("\nCounts:")
print(target_counts)

print("\nPercentages:")
print(target_percent.round(2))


# =========================
# 2. Missing values
# =========================

print("\n" + "-" * 60)
print("2. MISSING VALUES")
print("-" * 60)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_table = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent.round(2)
})

missing_table = missing_table[
    missing_table["missing_count"] > 0
].sort_values("missing_count", ascending=False)

print(missing_table)


# =========================
# 3. Unique values
# =========================

print("\n" + "-" * 60)
print("3. UNIQUE VALUES")
print("-" * 60)

for column in df.columns:
    print(f"{column:25} {df[column].nunique():6} unique values")


# =========================
# 4. Text length analysis
# =========================

print("\n" + "-" * 60)
print("4. TEXT LENGTH ANALYSIS")
print("-" * 60)

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

for column in text_columns:

    lengths = df[column].fillna("").astype(str).str.len()

    print(f"\n{column}")
    print(f"  Average length : {lengths.mean():.2f}")
    print(f"  Minimum length : {lengths.min()}")
    print(f"  Maximum length : {lengths.max()}")
    print(f"  Empty values   : {(lengths == 0).sum()}")


# =========================
# 5. Empty / whitespace values
# =========================

print("\n" + "-" * 60)
print("5. EMPTY / WHITESPACE VALUES")
print("-" * 60)

for column in df.columns:

    empty_count = (
        df[column]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty_count > 0:
        print(f"{column:25} {empty_count}")


# =========================
# 6. HTML detection
# =========================

print("\n" + "-" * 60)
print("6. HTML CONTENT")
print("-" * 60)

html_pattern = re.compile(r"<[^>]+>")

for column in text_columns:

    html_count = (
        df[column]
        .fillna("")
        .astype(str)
        .apply(lambda x: bool(html_pattern.search(x)))
        .sum()
    )

    print(f"{column:25} {html_count} rows containing HTML")


# =========================
# 7. URL detection
# =========================

print("\n" + "-" * 60)
print("7. URL / LINK DETECTION")
print("-" * 60)

url_pattern = re.compile(
    r"(https?://|www\.|\.com\b|\.org\b|\.net\b)",
    re.IGNORECASE
)

for column in text_columns:

    url_count = (
        df[column]
        .fillna("")
        .astype(str)
        .apply(lambda x: bool(url_pattern.search(x)))
        .sum()
    )

    print(f"{column:25} {url_count} rows containing URLs")


# =========================
# 8. Suspicious URL placeholders
# =========================

print("\n" + "-" * 60)
print("8. URL PLACEHOLDERS")
print("-" * 60)

placeholder_pattern = re.compile(r"#URL_[^#]+#", re.IGNORECASE)

for column in text_columns:

    count = (
        df[column]
        .fillna("")
        .astype(str)
        .apply(lambda x: bool(placeholder_pattern.search(x)))
        .sum()
    )

    print(f"{column:25} {count} rows")


# =========================
# 9. Categorical distributions
# =========================

print("\n" + "-" * 60)
print("9. CATEGORICAL FEATURES")
print("-" * 60)

categorical_columns = [
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

for column in categorical_columns:

    print(f"\n### {column}")

    print(
        df[column]
        .fillna("MISSING")
        .value_counts()
        .head(15)
    )


# =========================
# 10. Fraud rate by categorical features
# =========================

print("\n" + "-" * 60)
print("10. FRAUD RATE BY CATEGORICAL FEATURES")
print("-" * 60)

for column in categorical_columns:

    print(f"\n### {column}")

    fraud_rate = (
        df.groupby(df[column].fillna("MISSING"))["fraudulent"]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
        .head(15)
    )

    fraud_rate["fraud_percent"] = (
        fraud_rate["mean"] * 100
    ).round(2)

    print(fraud_rate[["count", "fraud_percent"]])


# =========================
# 11. Fraud rate by text availability
# =========================

print("\n" + "-" * 60)
print("11. FRAUD RATE AND MISSING TEXT")
print("-" * 60)

for column in text_columns:

    missing_mask = df[column].isna()

    print(f"\n{column}")

    print(
        df.groupby(missing_mask)["fraudulent"]
        .agg(["count", "mean"])
        .rename(index={
            False: "Present",
            True: "Missing"
        })
    )


# =========================
# 12. Potential data leakage
# =========================

print("\n" + "-" * 60)
print("12. POTENTIAL DATA LEAKAGE CHECK")
print("-" * 60)

suspicious_words = [
    "fraud",
    "scam",
    "fake",
    "fraudulent",
    "legitimate"
]

for column in text_columns:

    print(f"\n### {column}")

    for word in suspicious_words:

        count = (
            df[column]
            .fillna("")
            .astype(str)
            .str.contains(
                word,
                case=False,
                regex=False
            )
            .sum()
        )

        if count > 0:
            print(f"{word:15} {count} rows")


# =========================
# 13. Duplicate analysis
# =========================

print("\n" + "-" * 60)
print("13. DUPLICATE ANALYSIS")
print("-" * 60)

print(
    "Duplicate complete rows:",
    df.duplicated().sum()
)

print(
    "Duplicate job IDs:",
    df["job_id"].duplicated().sum()
)


# =========================
# 14. Correlation of simple numeric features
# =========================

print("\n" + "-" * 60)
print("14. NUMERIC FEATURES")
print("-" * 60)

numeric_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "fraudulent"
]

print(df[numeric_columns].corr().round(3))


# =========================
# Finished
# =========================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)