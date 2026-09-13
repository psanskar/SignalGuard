from detector.risk_engine import RiskEngine


job = {
    "title": "Work From Home Data Entry",

    "company_profile":
        "A growing company offering remote employment opportunities.",

    "description": """
        Work from home and earn guaranteed income of $5000 per week.
        No experience required.
        You must pay a registration fee before starting.
        Send your bank account details to our recruiter on Telegram.
    """,

    "requirements": "No experience required.",
    "benefits": "Flexible working hours.",

    "location": "Remote",
    "department": "Data Entry",
    "salary_range": "$5000 per week",
    "employment_type": "Part-time",
    "required_experience": "Not Applicable",
    "required_education": "Not Applicable",
    "industry": "Other",
    "function": "Data Entry",

    "telecommuting": 1,
    "has_company_logo": 0,
    "has_questions": 0
}


print("=" * 60)
print("SIGNALGUARD RISK ENGINE TEST")
print("=" * 60)


engine = RiskEngine()

result = engine.assess(job)


print("\nFraud Probability:")
print(result["fraud_probability"])


print("\nRisk Level:")
print(result["risk_level"])


print("\nML Prediction:")
print(result["ml_prediction"])


print("\nRed Flags:")

for flag in result["red_flags"]:
    print(
        f"- {flag['category']} "
        f"({flag['severity']}): "
        f"{flag['matches']}"
    )


print("\nTop SHAP Signals:")

for signal in result["shap_signals"]:
    print(
        f"- {signal['feature']}: "
        f"{signal['contribution']:+.4f} "
        f"→ {signal['direction'].upper()}"
    )


print("\nRecommendations:")

for recommendation in result["recommendations"]:
    print(
        f"- {recommendation}"
    )