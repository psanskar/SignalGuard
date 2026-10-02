from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.knowledge_base import KnowledgeBase


class KnowledgeRetriever:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.documents = self.knowledge_base.get_documents()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.document_vectors = self.vectorizer.fit_transform(
            [document["content"] for document in self.documents]
        )

    @staticmethod
    def _normalize_source(source):
        source = str(source).lower().strip()
        source = source.replace("\\", "/").split("/")[-1]
        if source.endswith(".md"):
            source = source[:-3]
        return source.replace("-", "_").replace(" ", "_")

    def retrieve(self, query, top_k=3, allowed_sources=None):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        candidate_indices = list(range(len(self.documents)))

        if allowed_sources:
            normalized_allowed = {self._normalize_source(source) for source in allowed_sources}
            candidate_indices = [
                index for index in candidate_indices
                if self._normalize_source(self.documents[index]["source"]) in normalized_allowed
            ]

        if not candidate_indices:
            return []

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: similarities[index],
            reverse=True
        )

        return [
            {
                "source": self.documents[index]["source"],
                "content": self.documents[index]["content"],
                "score": round(float(similarities[index]), 4)
            }
            for index in ranked_indices[:top_k]
        ]
