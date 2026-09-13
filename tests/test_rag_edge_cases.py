from rag.rag_engine import RAGEngine


rag_engine = RAGEngine()


print("\n" + "=" * 70)
print("SIGNALGUARD RAG EDGE-CASE TEST")
print("=" * 70)


# ---------------------------------------------------------
# TEST 1: No red flags, no SHAP signals
# ---------------------------------------------------------

job_text = """
We are hiring a software developer.

The candidate will work with Python and Django.
Applicants can apply through our official company careers website.
No payment or sensitive information is required during the application.
"""

print("\n" + "-" * 70)
print("TEST 1: JOB TEXT ONLY")
print("-" * 70)


result = rag_engine.retrieve_for_job(
    job_text=job_text,
    red_flags=None,
    shap_signals=None,
    top_k=3
)


print("\nConstructed query:")
print(result["query"])

print("\nRetrieved knowledge:")

for rank, document in enumerate(result["results"], start=1):
    print(
        f"{rank}. {document['source']} "
        f"(score={document['score']})"
    )


# ---------------------------------------------------------
# TEST 2: Empty lists
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("TEST 2: EMPTY RED FLAGS + EMPTY SHAP SIGNALS")
print("-" * 70)


result = rag_engine.retrieve_for_job(
    job_text=job_text,
    red_flags=[],
    shap_signals=[],
    top_k=3
)


print("\nConstructed query:")
print(result["query"])

print("\nRetrieved knowledge:")

for rank, document in enumerate(result["results"], start=1):
    print(
        f"{rank}. {document['source']} "
        f"(score={document['score']})"
    )


print("\n" + "=" * 70)
print("EDGE-CASE TEST COMPLETED")
print("=" * 70)