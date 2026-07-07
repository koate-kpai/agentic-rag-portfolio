# test_vectorstore.py — Tests for the TF-IDF-based domain vector store
#
# TEST STRATEGY: Real TF-IDF, no mocking
# ----------------------------------------
# DomainVectorStore uses scikit-learn's TfidfVectorizer on 9 hardcoded
# documents.  Vectorizer training is deterministic, fast (< 50 ms), and
# requires no network.  We test with the real implementation to verify
# actual document ranking, not just code paths.

import pytest
from src.vectorstore import DomainVectorStore


@pytest.fixture
def vector_store():
    """
    Fixture that builds a real DomainVectorStore with the 9 hardcoded
    documents (3 per domain).  Training happens once per test class.
    """
    return DomainVectorStore()


class TestRetrieve:
    """
    Retrieval tests verify that the TF-IDF cosine-similarity ranker returns
    the expected domain documents for domain-relevant queries.
    """

    def test_retrieve_finance(self, vector_store):
        """
        GIVEN a finance-related query,
        WHEN retrieve is called with domain='finance',
        THEN the results contain only finance docs (fin-* IDs).
        """
        docs = vector_store.retrieve("SEC rules and insider trading", domain="finance")
        assert len(docs) > 0
        # All returned doc IDs should start with "fin-"
        assert all(d[0]["id"].startswith("fin-") for d in docs)

    def test_retrieve_healthcare(self, vector_store):
        """
        GIVEN a healthcare-related query,
        WHEN retrieve is called with domain='healthcare',
        THEN the results contain only healthcare docs (hlth-* IDs).
        """
        docs = vector_store.retrieve("HIPAA patient data privacy", domain="healthcare")
        assert len(docs) > 0
        assert all(d[0]["id"].startswith("hlth-") for d in docs)

    def test_retrieve_legal(self, vector_store):
        """
        GIVEN a legal-related query about contracts,
        WHEN retrieve is called with domain='legal',
        THEN the results contain only legal docs (law-* IDs).
        """
        # "enforceability" uses a different stem than "enforceable" (doc law-001),
        # so we use "offer acceptance consideration" which directly matches the
        # contract law document's content.
        docs = vector_store.retrieve("offer acceptance consideration", domain="legal")
        assert len(docs) > 0
        assert all(d[0]["id"].startswith("law-") for d in docs)

    def test_retrieve_top_k(self, vector_store):
        """
        GIVEN a domain with 3 documents,
        WHEN retrieve is called with top_k=1,
        THEN exactly 1 document is returned.
        """
        docs = vector_store.retrieve("SEC rule", domain="finance", top_k=1)
        assert len(docs) == 1

    def test_retrieve_empty_domain(self, vector_store):
        """
        GIVEN a domain that does not exist in the store,
        WHEN retrieve is called,
        THEN an empty list is returned (not None, not an error).
        """
        docs = vector_store.retrieve("anything", domain="nonexistent_domain")
        assert docs == []

    def test_retrieve_scores_are_floats(self, vector_store):
        """
        GIVEN any valid query and domain,
        WHEN retrieve is called,
        THEN every result tuple includes a float relevance score.
        """
        docs = vector_store.retrieve("clinical trials FDA", domain="healthcare")
        assert len(docs) > 0
        for doc, score in docs:
            assert isinstance(score, float)
            assert score > 0  # cosine similarity should be positive for relevant docs
