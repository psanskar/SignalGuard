from rag.retriever import KnowledgeRetriever


retriever = KnowledgeRetriever()

queries = [
    "The recruiter is asking me for my bank account details",
    "They want me to pay a registration fee before getting the job",
    "The employer sent me a check and wants me to send some money back",
    "The recruiter wants payment in Bitcoin",
    "The recruiter contacted me through Telegram",
]


print("\n" + "=" * 60)
print("SIGNALGUARD RAG RETRIEVER TEST")
print("=" * 60)


for query in queries:

    print("\n" + "-" * 60)
    print("QUERY:")
    print(query)

    results = retriever.retrieve(
        query,
        top_k=3
    )

    print("\nRETRIEVED DOCUMENTS:")

    for result in results:
        print(
            f"- {result['source']} "
            f"(score: {result['score']})"
        )