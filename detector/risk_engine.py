from detector.predictor import JobPredictor
from detector.preprocessing import JobPreprocessor
from detector.shap_explainer import SHAPExplainer
from detector.red_flags import RedFlagDetector

from rag.rag_engine import RAGEngine

from llm.explainer import LLMExplainer


class RiskEngine:

    def __init__(self):
        print("Initializing Risk Engine...")

        # Preprocessing
        self.preprocessor = JobPreprocessor()

        # Machine learning prediction
        self.predictor = JobPredictor()

        # Model explanation
        self.shap_explainer = SHAPExplainer()

        # Rule-based warning detection
        self.red_flag_detector = RedFlagDetector()

        # RAG knowledge retrieval
        self.rag_engine = RAGEngine()

        # LLM explanation
        self.llm_explainer = LLMExplainer()

        print("Risk Engine initialized successfully.")

    def assess(self, job_data, threshold=None, top_n=10):

        # ---------------------------------------------------------
        # 1. PREPROCESS JOB
        # ---------------------------------------------------------

        X, combined_text, structured_data = (
            self.preprocessor.prepare_job(job_data)
        )

        # ---------------------------------------------------------
        # 2. ML PREDICTION
        # ---------------------------------------------------------

        prediction = self.predictor.predict_with_threshold(
            X,
            threshold=threshold
        )

        fraud_probability = prediction["fraud_probability"]

        # ---------------------------------------------------------
        # 3. RED FLAG DETECTION
        # ---------------------------------------------------------

        red_flags = self.red_flag_detector.detect(
            combined_text
        )

        # ---------------------------------------------------------
        # 4. SHAP EXPLANATION
        # ---------------------------------------------------------

        feature_names = (
            list(
                self.preprocessor.vectorizer
                .get_feature_names_out()
            )
            + self.preprocessor.structured_features
        )

        shap_signals = self.shap_explainer.explain(
            X,
            feature_names,
            top_n=top_n
        )[0]

        # ---------------------------------------------------------
        # 5. RAG RETRIEVAL
        # ---------------------------------------------------------

        rag_result = self.rag_engine.retrieve_for_job(
            job_text=combined_text,
            red_flags=red_flags,
            shap_signals=shap_signals,
            top_k=3
        )

        # ---------------------------------------------------------
        # 6. DETERMINE OVERALL RISK LEVEL
        # ---------------------------------------------------------

        risk_level = self._determine_risk_level(
            fraud_probability,
            red_flags
        )

        # ---------------------------------------------------------
        # 7. BUILD SAFETY RECOMMENDATIONS
        # ---------------------------------------------------------

        recommendations = self._build_recommendations(
            red_flags
        )

        # ---------------------------------------------------------
        # 8. BUILD BASE RISK RESULT
        # ---------------------------------------------------------

        risk_result = {
            "fraud_probability": round(
                fraud_probability,
                4
            ),

            "risk_level": risk_level,

            # Normalized risk value for frontend styling/logic.
            # This should not depend on display text.
            "risk_code": risk_level,

            "ml_prediction": prediction["fraudulent"],

            "threshold": prediction["threshold"],

            "red_flags": red_flags,

            "shap_signals": shap_signals,

            "rag": rag_result,

            "recommendations": recommendations,

            "structured_data": structured_data
        }

        # ---------------------------------------------------------
        # 9. LLM EXPLANATION
        # ---------------------------------------------------------

        llm_report = self.llm_explainer.explain(
            risk_result
        )

        risk_result["llm_report"] = llm_report

        # ---------------------------------------------------------
        # 10. RETURN COMPLETE RESULT
        # ---------------------------------------------------------

        return risk_result

    # =============================================================
    # RISK LEVEL
    # =============================================================

    def _determine_risk_level(
        self,
        fraud_probability,
        red_flags
    ):

        # Count high and medium severity flags
        high_flags = sum(
            1
            for flag in red_flags
            if flag.get("severity") == "high"
        )

        medium_flags = sum(
            1
            for flag in red_flags
            if flag.get("severity") == "medium"
        )

        # ---------------------------------------------------------
        # HIGH RISK
        # ---------------------------------------------------------

        if fraud_probability >= 0.75:
            return "HIGH"

        if high_flags >= 1:
            return "HIGH"

        if medium_flags >= 2:
            return "HIGH"

        # ---------------------------------------------------------
        # MEDIUM RISK
        # ---------------------------------------------------------

        if fraud_probability >= 0.50:
            return "MEDIUM"

        if medium_flags >= 1:
            return "MEDIUM"

        # ---------------------------------------------------------
        # LOW RISK
        # ---------------------------------------------------------

        return "LOW"

    # =============================================================
    # RECOMMENDATIONS
    # =============================================================

    def _build_recommendations(self, red_flags):

        recommendations = []

        categories = {
            flag.get("category")
            for flag in red_flags
        }

        # ---------------------------------------------------------
        # UPFRONT PAYMENT
        # ---------------------------------------------------------

        if "upfront_payment" in categories:

            recommendations.append(
                "Do not pay registration, processing, "
                "training, application, or other upfront "
                "fees for the job."
            )

        # ---------------------------------------------------------
        # FINANCIAL INFORMATION
        # ---------------------------------------------------------

        if "financial_information" in categories:

            recommendations.append(
                "Do not share sensitive banking or financial "
                "information until the employer has been "
                "independently verified."
            )

        # ---------------------------------------------------------
        # PERSONAL INFORMATION
        # ---------------------------------------------------------

        if "sensitive_personal_information" in categories:

            recommendations.append(
                "Avoid sharing sensitive identity or personal "
                "documents until the employer and recruitment "
                "process have been independently verified."
            )

        # ---------------------------------------------------------
        # CRYPTOCURRENCY
        # ---------------------------------------------------------

        if "cryptocurrency" in categories:

            recommendations.append(
                "Do not send cryptocurrency or transfer funds "
                "to a recruiter or employer as part of the "
                "job application or hiring process."
            )

        # ---------------------------------------------------------
        # FAKE CHECK
        # ---------------------------------------------------------

        if "fake_check" in categories:

            recommendations.append(
                "Do not deposit unexpected checks or send money "
                "back after receiving an employer-provided "
                "payment."
            )

        # ---------------------------------------------------------
        # MESSAGING PLATFORM
        # ---------------------------------------------------------

        if "messaging_platform_recruitment" in categories:

            recommendations.append(
                "Independently verify the recruiter and company "
                "before continuing communication through "
                "Telegram, WhatsApp, or similar messaging platforms."
            )

        # ---------------------------------------------------------
        # URGENCY
        # ---------------------------------------------------------

        if "urgency_pressure" in categories:

            recommendations.append(
                "Do not let urgency or limited-time pressure "
                "prevent you from independently verifying the "
                "job and employer."
            )

        # ---------------------------------------------------------
        # UNREALISTIC INCOME
        # ---------------------------------------------------------

        if "unrealistic_income" in categories:

            recommendations.append(
                "Be cautious of guaranteed or unusually high "
                "income claims and verify the compensation "
                "details independently."
            )

        return recommendations