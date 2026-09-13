import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
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

from scipy.sparse import hstack
from xgboost import XGBClassifier


# =========================
# Paths
# =========================

DATA_PATH = "data/processed/cleaned_jobs.csv"

MODEL_PATH = "models/xgb_structured_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer_structured.joblib"
FEATURES_PATH = "models/structured_features.joblib"
THRESHOLD_PATH = "models/threshold.joblib"


# =========================
# Configuration
# =========================

RANDOM_STATE = 42
FINAL_THRESHOLD = 0.55

STRUCTURED_FEATURES = [
    "text_length",
    "word_count",
    "url_count",
    "missing_text_fields",
    "telecommuting",
    "has_company_logo",
    "has_questions"
]


# =========================
# Load data
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


X_text = df["combined_text"].fillna("")
y = df["fraudulent"].astype(int)


# =========================
# Train / Test split
# =========================
# 80% = training
# 20% = untouched final test

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTrain size:", len(X_train_text))
print("Test size:", len(X_test_text))

print("\nTraining class distribution:")
print(y_train.value_counts())


# =========================
# TF-IDF
# =========================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    stop_words="english",
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print("TF-IDF train shape:", X_train_tfidf.shape)
print("TF-IDF test shape:", X_test_tfidf.shape)


# =========================
# Structured features
# =========================

print("\nAdding structured features...")

X_train_structured = df.loc[
    X_train_text.index,
    STRUCTURED_FEATURES
].astype(float).values

X_test_structured = df.loc[
    X_test_text.index,
    STRUCTURED_FEATURES
].astype(float).values


# Combine TF-IDF + structured features

X_train = hstack([
    X_train_tfidf,
    X_train_structured
]).tocsr()

X_test = hstack([
    X_test_tfidf,
    X_test_structured
]).tocsr()


print("Final train feature shape:", X_train.shape)
print("Final test feature shape:", X_test.shape)


# =========================
# Class weight
# =========================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nScale positive weight:", scale_pos_weight)


# =========================
# Train XGBoost
# =========================

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model training completed.")


# =========================
# Test evaluation
# =========================

print("\nEvaluating final model...")

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= FINAL_THRESHOLD
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


# =========================
# Display results
# =========================

print("\n" + "=" * 50)
print("FINAL MODEL RESULTS")
print("=" * 50)

print(f"Threshold : {FINAL_THRESHOLD}")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"PR-AUC    : {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# =========================
# Save model artifacts
# =========================

print("\nSaving model artifacts...")

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    vectorizer,
    VECTORIZER_PATH
)

joblib.dump(
    STRUCTURED_FEATURES,
    FEATURES_PATH
)

joblib.dump(
    FINAL_THRESHOLD,
    THRESHOLD_PATH
)


print("\nSaved:")
print(MODEL_PATH)
print(VECTORIZER_PATH)
print(FEATURES_PATH)
print(THRESHOLD_PATH)

print("\nFinal model setup completed successfully.")