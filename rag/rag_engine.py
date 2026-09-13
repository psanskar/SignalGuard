from rag.semantic_retriever import SemanticRetriever


class RAGEngine:

    # Map detected red-flag categories to their knowledge documents.
    FLAG_SOURCE_MAP = {
        "upfront_payment": [
            "upfront_payment"
        ],

        "financial_information": [
            "financial_information"
        ],

        "sensitive_personal_information": [
            "general_job_safety"
        ],

        "cryptocurrency": [
            "cryptocurrency_scams"
        ],

        "messaging_platform_recruitment": [
            "messaging_platform_scams"
        ],

        "urgency_pressure": [
            "urgency_pressure"
        ],

        "unrealistic_income": [
            "unrealistic_income"
        ],

        "fake_check": [
            "fake_check_scams"
        ],
    }

    def __init__(self):
        print("Initializing RAG Engine...")

        self.retriever = SemanticRetriever()

        print("RAG Engine initialized successfully.")

    def build_query(self, job_text, red_flags=None, shap_signals=None):
        """
        Build a focused RAG query using:
        - Detected red flags
        - Important SHAP signals
        - Original job context
        """

        query_parts = []

        # Add detected red flags
        if red_flags:
            query_parts.append("Detected warning signs:")

            for flag in red_flags:
                category = flag.get("category", "")
                message = flag.get("message", "")

                query_parts.append(
                    f"{category}: {message}"
                )

        # Add important SHAP signals
        if shap_signals:
            query_parts.append("\nImportant model signals:")

            for signal in shap_signals[:5]:
                feature = signal.get("feature", "")
                direction = signal.get("direction", "")

                query_parts.append(
                    f"{feature} ({direction})"
                )

        # Add job context
        query_parts.append("\nJob context:")
        query_parts.append(job_text)

        return "\n".join(query_parts)

    def get_allowed_sources(self, red_flags=None):
        """
        Determine which knowledge sources are relevant to the
        detected red flags.
        """

        allowed_sources = set()

        if red_flags:
            for flag in red_flags:
                category = flag.get("category", "")

                sources = self.FLAG_SOURCE_MAP.get(
                    category,
                    []
                )

                allowed_sources.update(sources)

        # Clean jobs receive general job-safety guidance.
        if not allowed_sources:
            allowed_sources.add("general_job_safety")

        return list(allowed_sources)

    def retrieve_for_job(
        self,
        job_text,
        red_flags=None,
        shap_signals=None,
        top_k=3
    ):
        """
        Retrieve one best knowledge document for each
        detected red-flag category.

        The top_k parameter is retained for compatibility,
        but detected warning signs are prioritized so that
        important evidence is not hidden by a global top-k limit.
        """

        query = self.build_query(
            job_text=job_text,
            red_flags=red_flags,
            shap_signals=shap_signals
        )

        # Determine the knowledge sources that correspond
        # to the detected warning signs.
        allowed_sources = self.get_allowed_sources(
            red_flags=red_flags
        )

        # Retrieve one best result for each allowed source.
        results = []
        seen_sources = set()

        for source in allowed_sources:

            source_results = self.retriever.retrieve(
                query=query,
                top_k=1,
                allowed_sources=[source]
            )

            for result in source_results:

                normalized_source = self.retriever._normalize_source(
                    result["source"]
                )

                # Prevent duplicate documents.
                if normalized_source in seen_sources:
                    continue

                seen_sources.add(normalized_source)
                results.append(result)

        # Sort final results by semantic similarity.
        results.sort(
            key=lambda result: result["score"],
            reverse=True
        )

        return {
            "query": query,
            "results": results
        }