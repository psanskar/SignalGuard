from detector.red_flags import RedFlagDetector


detector = RedFlagDetector()


test_cases = {

    "SAFE JOB": """
    We are looking for a Software Developer to join our growing team.
    You will work with experienced developers on web applications.
    Apply through our official company careers page.
    """,

    "PAYMENT SCAM": """
    Applicants must pay a registration fee before starting.
    A training fee is also required.
    """,

    "NEGATED PAYMENT": """
    We never ask applicants to pay a registration fee.
    There is no application fee.
    """,

    "FINANCIAL SCAM": """
    Send your bank account details to our recruiter on Telegram.
    """,

    "NEGATED FINANCIAL REQUEST": """
    We will never ask for your bank account password.
    We do not require your credit card information.
    """,

    "SUSPICIOUS JOB": """
    Work from home and earn guaranteed income of $5000 per week.
    No experience required.

    To start the job, you must pay a registration fee.
    Send your bank account details to our recruiter on Telegram.
    """
}


for name, text in test_cases.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    results = detector.detect(text)

    if not results:
        print("No red flags detected.")
    else:
        for result in results:
            print(f"\nCategory: {result['category']}")
            print(f"Severity: {result['severity']}")
            print(f"Matches: {result['matches']}")
            print(f"Message: {result['message']}")