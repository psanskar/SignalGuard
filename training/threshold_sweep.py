import pandas as pd
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================
# 1. LOAD DATA
# ============================================

DATA_PATH = "data/processed/cleaned_jobs.csv"

df = pd.read_csv(DATA_PATH)

X_text = df["combined_text"].fillna("")
y = df["fraudulent"]

structured_columns = [
    "text_length",
    "word_count",
    "url_count",
    "missing_text_fields",
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

X_structured = df[structured_columns].fillna(0)


# ============================================
# 2. SAME TRAIN / TEST SPLIT
# ============================================

X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_text,
    X_structured,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================
# 3. LOAD SAVED VECTORIZER + MODEL
# ============================================

vectorizer = joblib.load(
    "models/tfidf_vectorizer_structured.joblib"
)

model = joblib.load(
    "models/xgb_structured_model.joblib"
)


# ============================================
# 4. TRANSFORM TEST DATA
# ============================================

X_text_test_tfidf = vectorizer.transform(X_text_test)

X_struct_test_sparse = csr_matrix(
    X_struct_test.astype(float).values
)

X_test_combined = hstack([
    X_text_test_tfidf,
    X_struct_test_sparse
]).tocsr()


# ============================================
# 5. GET FRAUD PROBABILITIES
# ============================================

y_probability = model.predict_proba(
    X_test_combined
)[:, 1]


# ============================================
# 6. THRESHOLD SWEEP
# ============================================

thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


results = []


print("\n" + "=" * 75)
print("THRESHOLD SWEEP")
print("=" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'FP':<10}"
    f"{'FN':<10}"
)

print("-" * 75)


for threshold in thresholds:

    # Convert probability into prediction
    y_pred = (y_probability >= threshold).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn
    })

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{fp:<10}"
        f"{fn:<10}"
    )


# ============================================
# 7. FIND BEST THRESHOLD BY F1
# ============================================

results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["f1"].idxmax()
]


print("\n" + "=" * 75)
print("BEST THRESHOLD BY F1")
print("=" * 75)

print(
    f"Threshold        : {best_row['threshold']:.2f}"
)

print(
    f"Precision        : {best_row['precision']:.4f}"
)

print(
    f"Recall           : {best_row['recall']:.4f}"
)

print(
    f"F1 Score         : {best_row['f1']:.4f}"
)

print(
    f"False Positives  : {int(best_row['false_positives'])}"
)

print(
    f"False Negatives  : {int(best_row['false_negatives'])}"
)


# ============================================
# 8. RECOMMENDED THRESHOLD FOR HIGH RECALL
# ============================================

# Find the lowest threshold that achieves
# at least 85% recall.

high_recall = results_df[
    results_df["recall"] >= 0.85
]

if not high_recall.empty:

    recommended_row = high_recall.iloc[0]

    print("\n" + "=" * 75)
    print("HIGH-RECALL THRESHOLD")
    print("=" * 75)

    print(
        f"Threshold        : {recommended_row['threshold']:.2f}"
    )

    print(
        f"Precision        : {recommended_row['precision']:.4f}"
    )

    print(
        f"Recall           : {recommended_row['recall']:.4f}"
    )

    print(
        f"F1 Score         : {recommended_row['f1']:.4f}"
    )

    print(
        f"False Positives  : {int(recommended_row['false_positives'])}"
    )

    print(
        f"False Negatives  : {int(recommended_row['false_negatives'])}"
    )

else:

    print("\nNo tested threshold achieved 85% recall.")


# ============================================
# 9. SAVE RESULTS
# ============================================

OUTPUT_PATH = "models/threshold_results.csv"

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nThreshold results saved to:")
print(OUTPUT_PATH)

print("\nThreshold sweep completed successfully.")