from rag.rag_engine import RAGEngine


rag = RAGEngine()


job_text = """
We are hiring remote data entry workers.

The selected candidate must pay a registration fee before starting.
You will receive guaranteed income of $500 per day.

The recruiter contacted me through Telegram and asked me
to provide my bank account details.

Please act immediately because only a few positions are available.
"""


print("\n" + "=" * 70)
print("SIGNALGUARD RAG ENGINE TEST")
print("=" * 70)

results = rag.retrieve_for_job(job_text, top_k=3)

print("\nJOB TEXT:")
print(job_text)

print("\n" + "-" * 70)
print("RETRIEVED KNOWLEDGE")
print("-" * 70)

for i, result in enumerate(results, start=1):

    print(f"\n{i}. {result['source']}")
    print(f"Score: {result['score']}")
    print("Content:")
    print(result["content"])