from rag.semantic_retriever import SemanticRetriever


retriever = SemanticRetriever()


queries = [
    # Financial information
    "The recruiter is asking me for my bank account details",
    "They want my debit card information",
    "The employer wants my banking credentials",

    # Upfront payment
    "They want me to pay a registration fee before getting the job",
    "The company says I must pay a training fee",
    "I was asked to send money before starting work",

    # Fake check
    "The employer sent me a check and wants me to send some money back",
    "They overpaid me and told me to return the difference",
    "The recruiter wants me to deposit a check and transfer money",

    # Cryptocurrency
    "The recruiter wants payment in Bitcoin",
    "Can I trust a company asking me to send BTC?",
    "They want me to transfer crypto to their wallet",
    "The employer says I need to buy Bitcoin before starting",
    "They are asking for USDT as a job payment",

    # Messaging platforms
    "The recruiter contacted me through Telegram",
    "They want me to continue the interview on WhatsApp",
    "The employer only communicates through Telegram",

    # Mixed signals
    "The recruiter contacted me on Telegram and wants Bitcoin",
    "They want my bank details and promise guaranteed income",
    "They ask for a registration fee and say I must act immediately",
]


print("\n" + "=" * 70)
print("SIGNALGUARD SEMANTIC RETRIEVER - EXTENDED TEST")
print("=" * 70)


for query in queries:

    print("\n" + "-" * 70)
    print("QUERY:")
    print(query)

    results = retriever.retrieve(query, top_k=3)

    print("\nTOP 3 RESULTS:")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result['source']} "
            f"(score: {result['score']})"
        )