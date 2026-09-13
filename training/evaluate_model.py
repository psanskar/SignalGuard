import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/cleaned_jobs.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("SIGNALGUARD MODEL EVALUATION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

X_text = df["combined_text"].fillna("")
y = df["fraudulent"].astype(int)


structured_features = [
    "text_length",
    "word_count",
    "url_count",
    "missing_text_fields",
    "telecommuting",
    "has_company_logo",
    "has_questions"
]


X_structured = df[structured_features].fillna(0).astype(float)


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 80% temporary training data
# 20% final test data

X_train_val_text, X_test_text, \
y_train_val, y_test, \
X_train_val_structured, X_test_structured = train_test_split(
    X_text,
    y,
    X_structured,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Second split:
# From the 80%, create:
# 60% training
# 20% validation

X_train_text, X_val_text, \
y_train, y_val, \
X_train_structured, X_val_structured = train_test_split(
    X_train_val_text,
    y_train_val,
    X_train_val_structured,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)


print("\nDataset split:")

print(f"Training:   {len(y_train)}")
print(f"Validation: {len(y_val)}")
print(f"Test:       {len(y_test)}")


# ============================================================
# 4. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    stop_words="english",
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(
    X_train_text
)

X_val_tfidf = vectorizer.transform(
    X_val_text
)

X_test_tfidf = vectorizer.transform(
    X_test_text
)


# ============================================================
# 5. COMBINE TEXT + STRUCTURED FEATURES
# ============================================================

X_train = hstack([
    X_train_tfidf,
    X_train_structured.values
]).tocsr()


X_val = hstack([
    X_val_tfidf,
    X_val_structured.values
]).tocsr()


X_test = hstack([
    X_test_tfidf,
    X_test_structured.values
]).tocsr()


print("\nFeature shapes:")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 6. TRAIN XGBOOST
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count


print("\nClass distribution:")
print(f"Legitimate: {negative_count}")
print(f"Fraudulent: {positive_count}")

print(
    f"Scale positive weight: "
    f"{scale_pos_weight:.2f}"
)


model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 7. VALIDATION PROBABILITIES
# ============================================================

val_probabilities = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# 8. FIND BEST THRESHOLD ON VALIDATION SET
# ============================================================

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)


best_threshold = None
best_f1 = -1


print("\n" + "=" * 60)
print("VALIDATION THRESHOLD SEARCH")
print("=" * 60)

print(
    f"\n{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)


for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print("\nBest validation threshold:")
print(f"{best_threshold:.2f}")

print(f"Validation F1:")
print(f"{best_f1:.4f}")


# ============================================================
# 9. FINAL TEST EVALUATION
# ============================================================

test_probabilities = model.predict_proba(
    X_test
)[:, 1]


test_predictions = (
    test_probabilities >= best_threshold
).astype(int)


accuracy = accuracy_score(
    y_test,
    test_predictions
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


# ============================================================
# 10. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"\nThreshold: {best_threshold:.2f}"
)

print(
    f"Accuracy:  {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 Score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)


print("\nConfusion Matrix:")

print(cm)


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)