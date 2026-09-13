from rag.knowledge_base import KnowledgeBase


kb = KnowledgeBase()

documents = kb.get_documents()

print("\n" + "=" * 60)
print("SIGNALGUARD KNOWLEDGE BASE TEST")
print("=" * 60)

print(f"\nDocuments loaded: {len(documents)}")

for document in documents:
    print(f"- {document['source']}")