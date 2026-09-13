import pandas as pd
import numpy as np
import joblib

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ==============================
# 1. LOAD DATA
# ==============================

DATA_PATH = "data/processed/cleaned_jobs.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

X_text = df["combined_text"].fillna("")
y = df["fraudulent"]

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(y.value_counts(normalize=True) * 100)


# ==============================
# 2. STRUCTURED FEATURES
# ==============================

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

print("\nStructured features:")
print(structured_columns)

print("\nStructured feature shape:")
print(X_structured.shape)


# ==============================
# 3. TRAIN / TEST SPLIT
# ==============================

X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_text,
    X_structured,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(y_train))
print("Testing samples:", len(y_test))


# ==============================
# 4. TF-IDF
# ==============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    stop_words="english",
    sublinear_tf=True
)

X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)

print("\nTF-IDF training shape:")
print(X_text_train_tfidf.shape)

print("TF-IDF testing shape:")
print(X_text_test_tfidf.shape)


# ==============================
# 5. CONVERT STRUCTURED FEATURES
# ==============================

X_struct_train_sparse = csr_matrix(
    X_struct_train.astype(float).values
)

X_struct_test_sparse = csr_matrix(
    X_struct_test.astype(float).values
)


# ==============================
# 6. COMBINE FEATURES
# ==============================

X_train_combined = hstack([
    X_text_train_tfidf,
    X_struct_train_sparse
]).tocsr()

X_test_combined = hstack([
    X_text_test_tfidf,
    X_struct_test_sparse
]).tocsr()

print("\nCombined training shape:")
print(X_train_combined.shape)

print("Combined testing shape:")
print(X_test_combined.shape)


# ==============================
# 7. HANDLE CLASS IMBALANCE
# ==============================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nLegitimate training samples:", negative_count)
print("Fraudulent training samples:", positive_count)
print("scale_pos_weight:", round(scale_pos_weight, 2))


# ==============================
# 8. TRAIN XGBOOST
# ==============================

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

print("\nTraining structured model...")

model.fit(
    X_train_combined,
    y_train
)

print("Training completed.")


# ==============================
# 9. PREDICTIONS
# ==============================

y_pred = model.predict(X_test_combined)

y_probability = model.predict_proba(
    X_test_combined
)[:, 1]


# ==============================
# 10. EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

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

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print("\n" + "=" * 50)
print("STRUCTURED MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Legitimate", "Fraudulent"],
        zero_division=0
    )
)


print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==============================
# 11. SAVE MODEL
# ==============================

MODEL_PATH = "models/xgb_structured_model.joblib"

VECTORIZER_PATH = "models/tfidf_vectorizer_structured.joblib"

joblib.dump(model, MODEL_PATH)

joblib.dump(vectorizer, VECTORIZER_PATH)


# Save structured feature names
FEATURES_PATH = "models/structured_features.joblib"

joblib.dump(
    structured_columns,
    FEATURES_PATH
)


print("\nSaved files:")

print(MODEL_PATH)
print(VECTORIZER_PATH)
print(FEATURES_PATH)

print("\nExperiment 2 completed successfully.")