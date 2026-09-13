from rag.semantic_retriever import SemanticRetriever


retriever = SemanticRetriever()


test_cases = [
    {
        "name": "Upfront Payment",
        "text": """
        The employer asks candidates to pay a small onboarding fee
        before joining the company.
        """,
        "expected": "upfront_payment.md"
    },

    {
        "name": "Fake Check Scam",
        "text": """
        The recruiter says I should deposit a cheque and return
        the remaining money after the payment clears.
        """,
        "expected": "fake_check_scams.md"
    },

    {
        "name": "Financial Information + Telegram",
        "text": """
        The recruiter contacted me on Telegram and asked me
        to send my bank details.
        """,
        "expected": "financial_information.md"
    },

    {
        "name": "Cryptocurrency",
        "text": """
        The employer requires applicants to purchase Bitcoin
        before beginning work.
        """,
        "expected": "cryptocurrency_scams.md"
    },

    {
        "name": "Legitimate Job",
        "text": """
        A company is hiring a software developer. Applicants can
        submit their resume through the company's official careers
        website.
        """,
        "expected": "general_job_safety.md"
    }
]


print("\n" + "=" * 70)
print("SIGNALGUARD RAG RETRIEVAL TEST")
print("=" * 70)


for i, test in enumerate(test_cases, start=1):

    print("\n" + "-" * 70)
    print(f"TEST {i}: {test['name']}")
    print("-" * 70)

    print("\nJOB TEXT:")
    print(test["text"].strip())

    results = retriever.retrieve(
        query=test["text"],
        top_k=3
    )

    print("\nTOP 3 RETRIEVED DOCUMENTS:")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['source']} "
            f"(Score: {result['score']})"
        )

    top_result = results[0]["source"]

    print("\nEXPECTED:")
    print(test["expected"])

    print("TOP RESULT:")
    print(top_result)

    if top_result == test["expected"]:
        print("RESULT: PASS")
    else:
        print("RESULT: CHECK")