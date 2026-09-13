from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag.knowledge_base import KnowledgeBase


class SemanticRetriever:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):
        print("Loading embedding model...")

        self.model = SentenceTransformer(model_name)

        self.knowledge_base = KnowledgeBase()
        self.documents = self.knowledge_base.get_documents()

        print(f"Loaded {len(self.documents)} knowledge documents.")

        # Create embeddings for all knowledge documents
        document_texts = [
            document["content"]
            for document in self.documents
        ]

        self.document_embeddings = self.model.encode(
            document_texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print("Knowledge document embeddings created.")

    @staticmethod
    def _normalize_source(source):
        """
        Convert different source formats into a comparable key.

        Examples:
        messaging_platform_scams.md
        messaging_platform_scams
        Messaging Platform Scams

        all become:
        messaging_platform_scams
        """

        source = str(source).lower().strip()

        # Remove path if present
        source = source.replace("\\", "/").split("/")[-1]

        # Remove file extension
        if source.endswith(".md"):
            source = source[:-3]

        # Normalize separators
        source = source.replace("-", "_").replace(" ", "_")

        return source

    def retrieve(
        self,
        query,
        top_k=3,
        allowed_sources=None
    ):
        """
        Retrieve semantically relevant knowledge documents.

        When allowed_sources is provided, semantic retrieval is
        restricted to those sources.

        This prevents unrelated knowledge categories from being
        retrieved merely because they are semantically similar.
        """

        # Convert query into an embedding
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Calculate similarity between query and all documents
        similarities = cosine_similarity(
            query_embedding,
            self.document_embeddings
        )[0]

        # Build candidate document indices
        candidate_indices = list(range(len(self.documents)))

        if allowed_sources:
            normalized_allowed = {
                self._normalize_source(source)
                for source in allowed_sources
            }

            candidate_indices = [
                index
                for index in candidate_indices
                if self._normalize_source(
                    self.documents[index]["source"]
                ) in normalized_allowed
            ]

        # If nothing matches the allowed sources,
        # return no results instead of unrelated knowledge.
        if not candidate_indices:
            return []

        # Rank only the eligible documents
        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: similarities[index],
            reverse=True
        )

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