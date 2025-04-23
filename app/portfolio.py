import pandas as pd
import uuid
from chromadb.config import Settings


class Portfolio:
    def __init__(self, file_path="app/resource/my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        
        # ✅ Use in-memory ChromaDB client for compatibility with Streamlit Cloud
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=None,  # disables filesystem-based persistence
            anonymized_telemetry=False
        ))
        
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        if not self.collection.count():
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas=[{"links": row["Links"]}],
                    ids=[str(uuid.uuid4())]
                )

    def query_links(self, skills):
        result = self.collection.query(query_texts=skills, n_results=2)
        return result.get('metadatas', [])
