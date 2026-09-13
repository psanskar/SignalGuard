import joblib
import shap
import numpy as np


class SHAPExplainer:
    def __init__(self, model_path="models/xgb_structured_model.joblib"):
        """
        Load the trained model and create a SHAP TreeExplainer.
        """

        self.model = joblib.load(model_path)

        self.explainer = shap.TreeExplainer(self.model)

        print("SHAP explainer initialized successfully.")

    def explain(self, X, feature_names, top_n=10):
        """
        Generate SHAP explanations for one or more samples.

        Parameters:
            X            : Feature matrix
            feature_names: Names corresponding to model features
            top_n        : Number of important features to return

        Returns:
            List of explanation dictionaries
        """

        shap_values = self.explainer.shap_values(X)

        shap_values = np.asarray(shap_values)

        explanations = []

        for row_values in shap_values:

            # Get indices of features with largest absolute contribution
            top_indices = np.argsort(
                np.abs(row_values)
            )[::-1][:top_n]

            features = []

            for index in top_indices:

                contribution = float(row_values[index])

                if contribution > 0:
                    direction = "fraud"
                else:
                    direction = "legitimate"

                features.append({
                    "feature": feature_names[index],
                    "contribution": round(contribution, 6),
                    "direction": direction
                })

            explanations.append(features)

        return explanations