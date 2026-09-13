import requests


url = "http://127.0.0.1:5000/predict"

job_data = {
    "title": "Work From Home Data Entry",
    "company_profile": "We are a growing company offering flexible remote opportunities.",
    "description": """
    Work from home and earn guaranteed income.
    No experience required.
    Contact our recruiter on Telegram to continue the application.
    You may need to provide your bank account details.
    """,
    "requirements": "Basic computer skills. Flexible working hours.",
    "benefits": "Remote work and flexible schedule."
}


response = requests.post(
    url,
    json=job_data
)


print("Status Code:", response.status_code)
print("\nResponse:")
print(response.json())