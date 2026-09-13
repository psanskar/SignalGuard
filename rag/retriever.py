from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.knowledge_base import KnowledgeBase


class KnowledgeRetriever:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()

        self.documents = self.knowledge_base.get_documents()

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.document_vectors = self.vectorizer.fit_transform(
            [
                document["content"]
                for document in self.documents
            ]
        )

    def retrieve(self, query, top_k=3):

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.document_vectors
        )[0]

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "source": self.documents[index]["source"],
                "content": self.documents[index]["content"],
                "score": round(
                    float(similarities[index]),
                    4
                )
            })

        return results