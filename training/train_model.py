import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "cleaned_jobs.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_FILE = MODEL_DIR / "xgb_model.joblib"
VECTORIZER_FILE = MODEL_DIR / "tfidf_vectorizer.joblib"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("SIGNALGUARD - MODEL TRAINING")
print("=" * 70)

print("\nLoading processed dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. SELECT INPUT AND TARGET
# ============================================================

X_text = df["combined_text"].fillna("")
y = df["fraudulent"].astype(int)

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(X_train_text)}")
print(f"Testing samples:  {len(X_test_text)}")


# ============================================================
# 5. TF-IDF
# ============================================================

print("\nCreating TF-IDF representation...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3,
    stop_words="english",
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print(f"TF-IDF training shape: {X_train.shape}")
print(f"TF-IDF testing shape:  {X_test.shape}")


# ============================================================
# 6. HANDLE CLASS IMBALANCE
# ============================================================

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

print("\nClass imbalance handling:")
print(f"Legitimate training samples: {negative_count}")
print(f"Fraudulent training samples: {positive_count}")
print(f"scale_pos_weight: {scale_pos_weight:.2f}")


# ============================================================
# 7. CREATE XGBOOST MODEL
# ============================================================

print("\nCreating XGBoost model...")

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


# ============================================================
# 8. TRAIN
# ============================================================

print("\nTraining model...")
print("This may take some time...\n")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 9. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]


# ============================================================
# 10. EVALUATION
# ============================================================

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


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Legitimate",
            "Fraudulent"
        ],
        zero_division=0
    )
)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

print("\nMatrix format:")
print("[[True Negative, False Positive]")
print(" [False Negative, True Positive]]")


# ============================================================
# 13. SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    vectorizer,
    VECTORIZER_FILE
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(f"\nModel:")
print(MODEL_FILE)

print("\nTF-IDF Vectorizer:")
print(VECTORIZER_FILE)

print("\nTraining pipeline completed successfully.")