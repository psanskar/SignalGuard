from llm.prompt_builder import PromptBuilder


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
                "message": "The job appears to request an upfront payment.",
                "matches": ["registration fee"]
            },
            {
                "category": "financial_information",
                "severity": "high",
                "message": "The job appears to request sensitive financial information.",
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
                    "content": "Legitimate employers generally should not require applicants to pay registration or processing fees before employment."
                },
                {
                    "source": "financial_information.md",
                    "score": 0.68,
                    "content": "Job seekers should be cautious when asked to provide sensitive financial information before independently verifying an opportunity."
                }
            ]
        }
    }

    builder = PromptBuilder()

    prompt = builder.build(risk_result)

    print("\n========== GENERATED PROMPT ==========\n")
    print(prompt)


if __name__ == "__main__":
    main()