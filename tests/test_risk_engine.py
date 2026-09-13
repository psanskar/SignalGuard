from detector.risk_engine import RiskEngine


engine = RiskEngine()

job_data = {
    "title": "Work From Home Data Entry",
    "company_profile": """
    We are a growing company offering remote opportunities.
    """,

    "description": """
    Work from home and earn guaranteed income.
    Contact our recruiter on Telegram to continue the application.
    Please provide your bank account details.
    """,

    "requirements": """
    Basic computer skills.
    """,

    "benefits": """
    Flexible working hours.
    """
}


result = engine.assess(job_data)


print("\n" + "=" * 60)
print("COMBINED TEXT")
print("=" * 60)

# This won't be in the returned result, so instead we inspect
# the preprocessor directly.
X, combined_text, structured_data = engine.preprocessor.prepare_job(job_data)

print(combined_text)


print("\n" + "=" * 60)
print("RED FLAGS")
print("=" * 60)

for flag in result["red_flags"]:
    print(flag)