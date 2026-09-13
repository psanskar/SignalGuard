class PromptBuilder:

    def build(self, risk_result):
        fraud_probability = risk_result.get("fraud_probability", 0)
        risk_level = risk_result.get("risk_level", "UNKNOWN")
        ml_prediction = risk_result.get("ml_prediction", False)
        threshold = risk_result.get("threshold", 0.55)

        red_flags = risk_result.get("red_flags", [])
        shap_signals = risk_result.get("shap_signals", [])
        rag = risk_result.get("rag", {})

        prompt = f"""
You are the explanation assistant for SignalGuard, a fraudulent job posting detection system.

Your job is to explain the evidence produced by SignalGuard in a clear and understandable way.

IMPORTANT RULES:

1. Do NOT independently decide whether a job is fraudulent.
2. The machine learning model provides the fraud probability and classification.
3. Red flags are warning signs, not proof of fraud.
4. SHAP signals explain which features influenced the machine learning model.
5. Retrieved RAG information provides supporting safety knowledge.
6. Do not invent facts about the recruiter, company, job, laws, statistics, or external events.
7. Only use the evidence provided below.
8. Clearly distinguish between:
   - ML model results
   - detected warning signs
   - model signals
   - retrieved safety knowledge
9. If the evidence is mixed, say that the evidence is mixed.
10. Give practical safety recommendations based only on the detected concerns.

ML RESULT:
Fraud probability: {fraud_probability}
ML classification: {"Fraudulent" if ml_prediction else "Not classified as fraudulent"}
Decision threshold: {threshold}
Overall risk level: {risk_level}

DETECTED RED FLAGS:
"""

        if red_flags:
            for flag in red_flags:
                prompt += f"""
- Category: {flag.get("category", "")}
  Severity: {flag.get("severity", "")}
  Message: {flag.get("message", "")}
  Matches: {flag.get("matches", [])}
"""
        else:
            prompt += "\nNo red flags were detected.\n"

        prompt += """

SHAP MODEL SIGNALS:
"""

        if shap_signals:
            for signal in shap_signals[:10]:
                prompt += f"""
- Feature: {signal.get("feature", "")}
  Contribution: {signal.get("contribution", "")}
  Direction: {signal.get("direction", "")}
"""
        else:
            prompt += "\nNo SHAP signals available.\n"

        prompt += """

RETRIEVED SAFETY KNOWLEDGE:
"""

        rag_results = rag.get("results", [])

        if rag_results:
            for result in rag_results:
                prompt += f"""
SOURCE: {result.get("source", "")}
RELEVANCE SCORE: {result.get("score", "")}

{result.get("content", "")}

---
"""
        else:
            prompt += "\nNo retrieved knowledge was available.\n"

        prompt += """

Return ONLY valid JSON.

The JSON must contain exactly these fields:

{
  "summary": "A short overall explanation.",
  "risk_explanation": "Explain why SignalGuard assigned this risk level using the supplied evidence.",
  "key_concerns": [
    "Concern 1",
    "Concern 2",
    "Concern 3"
  ],
  "safety_recommendations": [
    "Recommendation 1",
    "Recommendation 2",
    "Recommendation 3"
  ],
  "confidence_note": "Briefly explain that the assessment is based on model signals and detected warning signs and is not absolute proof of fraud."
}

Do not include Markdown.
Do not include ```json.
Do not add any fields outside this structure.
"""

        return prompt