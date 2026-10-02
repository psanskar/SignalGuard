import os

from rag.retriever import KnowledgeRetriever


class RAGEngine:
    FLAG_SOURCE_MAP = {
        "upfront_payment": ["upfront_payment"],
        "financial_information": ["financial_information"],
        "sensitive_personal_information": ["general_job_safety"],
        "cryptocurrency": ["cryptocurrency_scams"],
        "messaging_platform_recruitment": ["messaging_platform_scams"],
        "urgency_pressure": ["urgency_pressure"],
        "unrealistic_income": ["unrealistic_income"],
        "fake_check": ["fake_check_scams"],
    }

    def __init__(self):
        print("Initializing RAG Engine...")
        light_mode = os.getenv("SIGNALGUARD_LIGHT_MODE", "0").lower() in {"1", "true", "yes"}

        if light_mode:
            print("Using lightweight TF-IDF safety retrieval.")
            self.retriever = KnowledgeRetriever()
        else:
            from rag.semantic_retriever import SemanticRetriever
            print("Using semantic safety retrieval.")
            self.retriever = SemanticRetriever()

        print("RAG Engine initialized successfully.")

    def build_query(self, job_text, red_flags=None, shap_signals=None):
        query_parts = []

        if red_flags:
            query_parts.append("Detected warning signs:")
            for flag in red_flags:
                query_parts.append(f"{flag.get('category', '')}: {flag.get('message', '')}")

        if shap_signals:
            query_parts.append("\nImportant model signals:")
            for signal in shap_signals[:5]:
                query_parts.append(f"{signal.get('feature', '')} ({signal.get('direction', '')})")

        query_parts.append("\nJob context:")
        query_parts.append(job_text)
        return "\n".join(query_parts)

    def get_allowed_sources(self, red_flags=None):
        allowed_sources = set()

        if red_flags:
            for flag in red_flags:
                allowed_sources.update(self.FLAG_SOURCE_MAP.get(flag.get("category", ""), []))

        if not allowed_sources:
            allowed_sources.add("general_job_safety")

        return list(allowed_sources)

    def retrieve_for_job(self, job_text, red_flags=None, shap_signals=None, top_k=3):
        query = self.build_query(job_text, red_flags, shap_signals)
        results = []
        seen_sources = set()

        for source in self.get_allowed_sources(red_flags):
            for result in self.retriever.retrieve(query=query, top_k=1, allowed_sources=[source]):
                normalized_source = self.retriever._normalize_source(result["source"])
                if normalized_source in seen_sources:
                    continue
                seen_sources.add(normalized_source)
                results.append(result)

        results.sort(key=lambda result: result["score"], reverse=True)

        return {"query": query, "results": results}
