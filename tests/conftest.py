"""
ClaimTrace: Pytest fixtures and configuration.
"""

import os
import sys
import pytest

# Ensure root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval import DocumentSentence


@pytest.fixture
def sample_sentences():
    return [
        DocumentSentence("Nikolaj_Coster-Waldau", 0, "Nikolaj William Coster-Waldau is a Danish actor and producer."),
        DocumentSentence("Nikolaj_Coster-Waldau", 3, "He played Jaime Lannister in the HBO fantasy drama series Game of Thrones."),
        DocumentSentence("Nikolaj_Coster-Waldau", 7, "He appeared in the Fox television movie New Amsterdam in 2008 as Detective John Amsterdam."),
        DocumentSentence("Michael_Bay", 1, "He is best known for directing high-budget action films including Armageddon and Transformers."),
        DocumentSentence("Marie_Curie", 1, "She was the first woman to win a Nobel Prize, the first person to win twice, and the only person to win in two scientific fields."),
    ]
