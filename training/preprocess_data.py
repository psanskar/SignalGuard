import pandas as pd
import re
import html
from pathlib import Path


# ==============================
# 1. PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "fake_job_postings.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "cleaned_jobs.csv"


# ==============================
# 2. TEXT CLEANING FUNCTION
# ==============================

def clean_text(text):
    """
    Clean a text field while preserving useful information.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities such as &amp;
    text = html.unescape(text)

    # Replace URL placeholders such as #URL_123#
    text = re.sub(r"#URL_[^#]*#", " URL ", text, flags=re.IGNORECASE)

    # Replace actual URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URL ",
        text,
        flags=re.IGNORECASE
    )

    # Remove HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text


# ==============================
# 3. LOAD DATASET
# ==============================

print("=" * 60)
print("SIGNALGUARD - DATA PREPROCESSING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(RAW_FILE)

print(f"Original shape: {df.shape}")


# ==============================
# 4. REMOVE DUPLICATE ROWS
# ==============================

before = len(df)

df = df.drop_duplicates()

removed = before - len(df)

print(f"Duplicate rows removed: {removed}")


# ==============================
# 5. CLEAN TEXT COLUMNS
# ==============================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

print("\nCleaning text columns...")

for column in text_columns:
    df[column] = df[column].apply(clean_text)


# ==============================
# 6. CLEAN CATEGORICAL COLUMNS
# ==============================

categorical_columns = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

print("Cleaning categorical columns...")

for column in categorical_columns:

    df[column] = (
        df[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    # Convert empty strings to Unknown
    df[column] = df[column].replace("", "Unknown")


# ==============================
# 7. CLEAN BINARY COLUMNS
# ==============================

binary_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

print("Cleaning binary columns...")

for column in binary_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0).astype(int)


# ==============================
# 8. CREATE COMBINED TEXT
# ==============================

print("Creating combined text...")

df["combined_text"] = (
    "Title: " + df["title"] + " "
    + "Company Profile: " + df["company_profile"] + " "
    + "Description: " + df["description"] + " "
    + "Requirements: " + df["requirements"] + " "
    + "Benefits: " + df["benefits"]
)


# ==============================
# 9. BASIC TEXT FEATURES
# ==============================

print("Creating text-based features...")

df["text_length"] = df["combined_text"].str.len()

df["word_count"] = (
    df["combined_text"]
    .str.split()
    .str.len()
)

df["url_count"] = (
    df["combined_text"]
    .str.count(r"\bURL\b")
)


# ==============================
# 10. FRAUD-SIGNAL FEATURES
# ==============================

print("Creating fraud-signal features...")

fraud_keywords = [
    "money",
    "payment",
    "fee",
    "deposit",
    "bank account",
    "credit card",
    "wire transfer",
    "western union",
    "cash",
    "check",
    "cheque",
    "bitcoin",
    "cryptocurrency",
    "gift card",
    "urgent",
    "immediately",
    "guaranteed",
    "easy money",
    "work from home",
    "no experience",
    "personal information",
    "social security",
    "ssn",
    "passport",
    "driver license",
    "telegram",
    "whatsapp"
]


def count_fraud_keywords(text):
    """
    Count how many predefined suspicious keywords/phrases
    appear in the job posting.
    """

    text = text.lower()

    count = 0

    for keyword in fraud_keywords:
        if keyword in text:
            count += 1

    return count


df["fraud_keyword_count"] = (
    df["combined_text"]
    .apply(count_fraud_keywords)
)


# ==============================
# 11. MISSING INFORMATION FEATURES
# ==============================

print("Creating missing-information features...")

df["missing_text_fields"] = (
    (df["company_profile"] == "").astype(int)
    + (df["description"] == "").astype(int)
    + (df["requirements"] == "").astype(int)
    + (df["benefits"] == "").astype(int)
)


# ==============================
# 12. SAVE PROCESSED DATA
# ==============================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==============================
# 13. FINAL REPORT
# ==============================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nFinal shape: {df.shape}")

print(f"\nOutput saved to:")
print(OUTPUT_FILE)

print("\nColumns created:")
print("- combined_text")
print("- text_length")
print("- word_count")
print("- url_count")
print("- fraud_keyword_count")
print("- missing_text_fields")

print("\nRemaining missing values:")
print(df.isnull().sum().sum())

print("\nTarget distribution:")

if "fraudulent" in df.columns:
    print(df["fraudulent"].value_counts())

print("\nFirst 3 processed records:")
print(
    df[
        [
            "title",
            "combined_text",
            "fraud_keyword_count",
            "missing_text_fields",
            "fraudulent"
        ]
    ].head(3)
)