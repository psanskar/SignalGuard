from detector.red_flags import RedFlagDetector


detector = RedFlagDetector()


tests = {

    "Telegram recruitment": """
        Contact our recruiter through Telegram to continue the application.
    """,

    "WhatsApp recruitment": """
        Message us on WhatsApp for the next interview.
    """,

    "Financial information": """
        Please provide your bank account details.
    """,

    "Guaranteed income": """
        You will receive guaranteed income of $500 per day.
    """,

    "Upfront payment": """
        You must pay a registration fee before starting.
    """,

    "Negated Telegram": """
        We do not use Telegram for recruitment.
    """,

    "Safe job": """
        We are hiring a software developer.
        Apply through our official company website.
    """
}


for name, text in tests.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    results = detector.detect(text)

    if not results:
        print("No red flags detected.")
    else:
        for result in results:
            print(
                f"{result['category']} | "
                f"{result['severity']} | "
                f"{result['matches']}"
            )