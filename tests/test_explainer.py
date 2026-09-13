from llm.explainer import LLMExplainer


def main():

    risk_result = {
        "fraud_probability": 0.82,
        "risk_level": "HIGH",
        "ml_prediction": True,
        "threshold": 0.55,

        "red_flags": [
            {
                "category": "upfront_payment",
                "severity": "high",
                "message": "Job asks for an upfront payment.",
                "matches": ["registration fee"]
            },
            {
                "category": "financial_information",
                "severity": "high",
                "message": "Job asks for sensitive financial information.",
                "matches": ["bank account details"]
            }
        ],

        "shap_signals": [
            {
                "feature": "registration fee",
                "contribution": 1.42,
                "direction": "fraud"
            },
            {
                "feature": "bank account",
                "contribution": 0.91,
                "direction": "fraud"
            }
        ],

        "rag": {
            "results": [
                {
                    "source": "upfront_payment.md",
                    "score": 0.72,
                    "content": (
                        "Job seekers should be cautious when asked "
                        "to pay registration or processing fees."
                    )
                },
                {
                    "source": "financial_information.md",
                    "score": 0.68,
                    "content": (
                        "Avoid sharing sensitive banking information "
                        "before independently verifying an employer."
                    )
                }
            ]
        }
    }

    explainer = LLMExplainer()

    result = explainer.explain(risk_result)

    print("\nStructured LLM Response:")
    print(result)

    print("\nSummary:")
    print(result["summary"])

    print("\nKey Concerns:")
    for concern in result["key_concerns"]:
        print("-", concern)

    print("\nSafety Recommendations:")
    for recommendation in result["safety_recommendations"]:
        print("-", recommendation)


if __name__ == "__main__":
    main()