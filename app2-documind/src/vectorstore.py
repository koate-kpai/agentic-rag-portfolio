# vectorstore.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
from .domains import FINANCE_DOCS, HEALTHCARE_DOCS, LEGAL_DOCS


class DomainVectorStore:
    """
    A lightweight, in-memory mock vector store using Scikit-Learn.
    In a real enterprise setting, this would be replaced by Pinecone, Qdrant, or pgvector.
    """

    def __init__(self):
        # TF-IDF converts text into numerical vectors based on term frequency and rarity
        self.vectorizer = TfidfVectorizer(stop_words="english")

        # We separate documents by domain to simulate multi-tenant or siloed enterprise data
        self.docs_by_domain = {
            "finance": FINANCE_DOCS,
            "healthcare": HEALTHCARE_DOCS,
            "legal": LEGAL_DOCS,
        }

        # Train the vectorizer on the entire corpus to build the vocabulary
        all_texts = []
        for docs in self.docs_by_domain.values():
            all_texts.extend([doc["text"] for doc in docs])
        self.vectorizer.fit(all_texts)

    def retrieve(
        self, query: str, domain: str, top_k: int = 2
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieves the top_k most relevant documents for a specific domain.
        """
        domain_docs = self.docs_by_domain.get(domain, [])
        if not domain_docs:
            return []

        corpus = [doc["text"] for doc in domain_docs]

        # Transform the user query and the domain documents into vectors
        query_vec = self.vectorizer.transform([query])
        corpus_vec = self.vectorizer.transform(corpus)

        # Calculate how similar the query vector is to each document vector
        similarities = cosine_similarity(query_vec, corpus_vec).flatten()

        # Sort to find the indices of the highest scoring documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        # Return the documents and their scores, filtering out zero-similarity results
        return [
            (domain_docs[i], similarities[i])
            for i in top_indices
            if similarities[i] > 0
        ]
