import joblib


class JobPredictor:

    def __init__(
        self,
        model_path="models/xgb_structured_model.joblib",
        threshold_path="models/threshold.joblib"
    ):

        self.model = joblib.load(model_path)
        self.threshold = joblib.load(threshold_path)

        print("Model loaded successfully.")
        print(f"Prediction threshold: {self.threshold}")

    def predict(self, X):

        fraud_probability = float(
            self.model.predict_proba(X)[0][1]
        )

        return fraud_probability

    def predict_with_threshold(
        self,
        X,
        threshold=None
    ):

        fraud_probability = self.predict(X)

        if threshold is None:
            threshold = self.threshold

        fraudulent = fraud_probability >= threshold

        return {
            "fraud_probability": round(
                fraud_probability,
                4
            ),
            "fraudulent": fraudulent,
            "threshold": threshold
        }