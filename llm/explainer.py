from llm.llm_client import LLMClient, LLMUnavailableError
from llm.prompt_builder import PromptBuilder


class LLMExplainer:

    def __init__(self):

        self.client = LLMClient()
        self.prompt_builder = PromptBuilder()

    def explain(self, risk_result):

        prompt = self.prompt_builder.build(risk_result)

        try:

            response = self.client.generate_json(prompt)

            required_fields = [
                "summary",
                "risk_explanation",
                "key_concerns",
                "safety_recommendations",
                "confidence_note"
            ]

            for field in required_fields:

                if field not in response:
                    raise ValueError(
                        f"LLM response missing required field: {field}"
                    )

            response["status"] = "success"

            return response

        except LLMUnavailableError as error:

            print(
                f"LLM temporarily unavailable: {error}"
            )

            return self._build_fallback_report(risk_result)

    def _build_fallback_report(self, risk_result):

        fraud_probability = risk_result.get(
            "fraud_probability",
            0
        )

        risk_level = risk_result.get(
            "risk_level",
            "UNKNOWN"
        )

        ml_prediction = risk_result.get(
            "ml_prediction",
            False
        )

        red_flags = risk_result.get(
            "red_flags",
            []
        )

        recommendations = risk_result.get(
            "recommendations",
            []
        )

        shap_signals = risk_result.get(
            "shap_signals",
            []
        )

        # Build concerns from detected red flags.
        key_concerns = []

        for flag in red_flags:

            category = flag.get(
                "category",
                "unknown"
            )

            message = flag.get(
                "message",
                "A warning sign was detected."
            )

            key_concerns.append(
                f"{category}: {message}"
            )

        # Add important SHAP signals.
        for signal in shap_signals[:3]:

            feature = signal.get(
                "feature",
                ""
            )

            direction = signal.get(
                "direction",
                ""
            )

            if feature:

                key_concerns.append(
                    f"Model signal: '{feature}' "
                    f"contributed toward {direction}."
                )

        if not key_concerns:

            key_concerns.append(
                "No specific red flags were detected "
                "by the configured rule-based detector."
            )

        classification = (
            "fraudulent"
            if ml_prediction
            else "not classified as fraudulent"
        )

        return {
            "status": "unavailable",

            "summary": (
                f"SignalGuard assigned an overall {risk_level} risk level. "
                f"The ML model estimated a fraud probability of "
                f"{fraud_probability:.2f} and classified the job as "
                f"{classification}. The natural-language LLM explanation "
                f"is temporarily unavailable."
            ),

            "risk_explanation": (
                f"The assessment is based on SignalGuard's machine "
                f"learning prediction, detected warning signs, SHAP "
                f"model signals, and retrieved safety knowledge. "
                f"The current ML fraud probability is "
                f"{fraud_probability:.2f}, with an overall risk level "
                f"of {risk_level}."
            ),

            "key_concerns": key_concerns,

            "safety_recommendations": recommendations,

            "confidence_note": (
                "The LLM explanation service was temporarily "
                "unavailable. This assessment is based on the "
                "SignalGuard ML model and supporting detection "
                "signals and should not be treated as absolute "
                "proof of fraud."
            )
        }