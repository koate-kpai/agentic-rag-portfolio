# domains.py

# Mock domain-specific documents
# In production, these would come from a vector database with embeddings
FINANCE_DOCS = [
    {
        "id": "fin-001",
        "text": "SEC Rule 10b-5 prohibits fraud in connection with the purchase or sale of any security.",
    },
    {
        "id": "fin-002",
        "text": "Under the Sarbanes-Oxley Act, CEOs and CFOs must certify the accuracy of financial statements.",
    },
    {
        "id": "fin-003",
        "text": "Insider trading is illegal when based on material non-public information about a company.",
    },
]

HEALTHCARE_DOCS = [
    {
        "id": "hlth-001",
        "text": "HIPAA requires covered entities to ensure the confidentiality and integrity of protected health information (PHI).",
    },
    {
        "id": "hlth-002",
        "text": "The FDA approval process for new drugs involves three phases of clinical trials.",
    },
    {
        "id": "hlth-003",
        "text": "Patient consent is mandatory before any medical procedure except in emergency situations.",
    },
]

LEGAL_DOCS = [
    {
        "id": "law-001",
        "text": "Contracts must have offer, acceptance, and consideration to be enforceable.",
    },
    {
        "id": "law-002",
        "text": "The doctrine of stare decisis obligates courts to follow precedents set by higher courts.",
    },
    {
        "id": "law-003",
        "text": "Force majeure clauses excuse contractual obligations when extraordinary events occur.",
    },
]
