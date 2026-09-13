from pathlib import Path


class KnowledgeBase:
    def __init__(self, knowledge_dir="rag/knowledge"):
        self.knowledge_dir = Path(knowledge_dir)

    def load_documents(self):
        documents = []

        for file_path in self.knowledge_dir.glob("*.md"):
            content = file_path.read_text(
                encoding="utf-8"
            )

            documents.append({
                "source": file_path.name,
                "content": content
            })

        return documents

    def get_documents(self):
        return self.load_documents()