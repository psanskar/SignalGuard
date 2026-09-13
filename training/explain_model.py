import joblib
import pandas as pd
import shap
import numpy as np
from scipy.sparse import hstack


# -----------------------------
# 1. Load model and components
# -----------------------------

model = joblib.load("models/xgb_structured_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer_structured.joblib")
structured_features = joblib.load("models/structured_features.joblib")

df = pd.read_csv("data/processed/cleaned_jobs.csv")


# -----------------------------
# 2. Prepare text
# -----------------------------

X_text = vectorizer.transform(df["combined_text"])


# -----------------------------
# 3. Prepare structured features
# -----------------------------

X_structured = df[structured_features].astype(float).values

X = hstack([
    X_text,
    X_structured
]).tocsr()


# -----------------------------
# 4. Feature names
# -----------------------------

text_features = vectorizer.get_feature_names_out()

feature_names = list(text_features) + list(structured_features)

print("Total features:", len(feature_names))
print("Model expects:", model.n_features_in_)


# -----------------------------
# 5. Create SHAP explainer
# -----------------------------

print("\nCreating SHAP TreeExplainer...")

explainer = shap.TreeExplainer(model)

print("SHAP explainer created successfully.")


# -----------------------------
# 6. Select one fraudulent job
#    and one legitimate job
# -----------------------------

fraud_index = df.index[df["fraudulent"] == 1][0]
legit_index = df.index[df["fraudulent"] == 0][0]

sample_indices = [fraud_index, legit_index]

X_sample = X[sample_indices]


# -----------------------------
# 7. Calculate SHAP values
# -----------------------------

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(X_sample)

print("SHAP calculation completed.")

print("SHAP shape:", np.array(shap_values).shape)


# -----------------------------
# 8. Display top features
# -----------------------------

for i, index in enumerate(sample_indices):

    print("\n" + "=" * 60)

    actual_label = df.loc[index, "fraudulent"]

    if actual_label == 1:
        print("Example: FRAUDULENT JOB")
    else:
        print("Example: LEGITIMATE JOB")

    print("Job title:", df.loc[index, "title"])

    print("=" * 60)

    values = np.array(shap_values[i])

    # Sort by absolute SHAP contribution
    top_indices = np.argsort(np.abs(values))[::-1][:15]

    print("\nTop SHAP features:")

    for feature_index in top_indices:

        feature = feature_names[feature_index]
        contribution = values[feature_index]

        direction = "→ FRAUD" if contribution > 0 else "→ LEGITIMATE"

        print(
            f"{feature:30s} "
            f"{contribution:+.6f} "
            f"{direction}"
        )