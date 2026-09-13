from rag.rag_engine import RAGEngine


rag_engine = RAGEngine()


job_text = """
We are hiring remote data entry workers.

The selected candidate must pay a registration fee before starting.
You will receive guaranteed income of $500 per day.

The recruiter contacted me through Telegram and asked me
to provide my bank account details.

Please act immediately because only a few positions are available.
"""


red_flags = [
    {
        "category": "upfront_payment",
        "severity": "high",
        "message": "The job requests payment before employment."
    },
    {
        "category": "financial_information",
        "severity": "high",
        "message": "The recruiter requests sensitive financial information."
    },
    {
        "category": "messaging_platform_recruitment",
        "severity": "medium",
        "message": "Recruitment is being conducted through Telegram."
    },
    {
        "category": "urgency_pressure",
        "severity": "medium",
        "message": "The recruiter creates pressure to act immediately."
    },
    {
        "category": "unrealistic_income",
        "severity": "medium",
        "message": "The job promises unusually high guaranteed income."
    }
]


shap_signals = [
    {
        "feature": "registration",
        "contribution": 1.2,
        "direction": "fraud"
    },
    {
        "feature": "income",
        "contribution": 0.9,
        "direction": "fraud"
    },
    {
        "feature": "telegram",
        "contribution": 0.7,
        "direction": "fraud"
    },
    {
        "feature": "bank",
        "contribution": 0.6,
        "direction": "fraud"
    },
    {
        "feature": "data entry",
        "contribution": 0.4,
        "direction": "fraud"
    }
]


print("\n" + "=" * 70)
print("SIGNALGUARD RAG QUERY CONSTRUCTION TEST")
print("=" * 70)


result = rag_engine.retrieve_for_job(
    job_text=job_text,
    red_flags=red_flags,
    shap_signals=shap_signals,
    top_k=3
)


print("\n" + "-" * 70)
print("CONSTRUCTED RAG QUERY")
print("-" * 70)

print(result["query"])


print("\n" + "-" * 70)
print("RETRIEVED KNOWLEDGE")
print("-" * 70)


for rank, document in enumerate(result["results"], start=1):

    print(
        f"\n{rank}. {document['source']}"
    )

    print(
        f"Score: {document['score']}"
    )