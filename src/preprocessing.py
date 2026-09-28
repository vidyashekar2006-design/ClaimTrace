"""
ClaimTrace: NLP Preprocessing Module
Preserves negation, named entities, dates, numbers, and original evidence text
for accurate fact-checking and natural language inference.
"""

import re
import unicodedata
from typing import List, Dict, Any, Tuple

# Negation words critical for claim refutation detection
NEGATION_WORDS = {
    "not", "no", "never", "none", "neither", "nor", "nowhere", "hardly",
    "scarcely", "barely", "doesn't", "don't", "didn't", "wasn't", "weren't",
    "isn't", "aren't", "cannot", "can't", "couldn't", "wouldn't", "shouldn't",
    "without", "refuse", "failed", "denied", "impossible", "unlikely"
}

# Common contractions expansion mapping
CONTRACTIONS = {
    r"\bwon\'t\b": "will not",
    r"\bcan\'t\b": "cannot",
    r"\bn\'t\b": " not",
    r"\b\'re\b": " are",
    r"\b\'d\b": " would",
    r"\b\'ll\b": " will",
    r"\b\'ve\b": " have",
    r"\b\'m\b": " am",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (e.g., accented characters, smart quotes) into NFKD/ASCII friendly form."""
    if not text:
        return ""
    # Standardize curly quotes and dashes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    # Replace non-breaking spaces
    text = text.replace("\xa0", " ")
    return unicodedata.normalize("NFKC", text)


def expand_contractions(text: str) -> str:
    """Expand contractions while preserving word boundaries."""
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def clean_claim_text(text: str, preserve_case: bool = False) -> str:
    """
    Clean input text for tokenization:
    - Normalizes whitespace and unicode
    - Expands contractions
    - Keeps numbers, dates, punctuation relevant to entity context
    - Optionally lowers case (default False to preserve Named Entities)
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = expand_contractions(text)
    # Collapse multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    if not preserve_case:
        return text
    return text.lower()


def tokenize_words(text: str, preserve_negation: bool = True) -> List[str]:
    """
    Tokenizes text into word tokens while preserving:
    - Numbers and dates (e.g. '1984', '3.14', 'July 4')
    - Hyphenated named entities (e.g. 'Coster-Waldau')
    - Negations
    Ignores standalone punctuation marks.
    """
    cleaned = clean_claim_text(text)
    # Regex keeps alphanumeric words, numbers, and hyphenated terms
    raw_tokens = re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", cleaned)
    tokens = []
    for tok in raw_tokens:
        tok_clean = tok.strip()
        if tok_clean:
            tokens.append(tok_clean)
    return tokens


def extract_claim_entities_and_dates(text: str) -> Dict[str, List[str]]:
    """
    Extract key salient surface forms (numbers, dates, capitalized named entity sequences)
    to boost evidence retrieval precision.
    """
    cleaned = clean_claim_text(text)
    
    # 4-digit years or date patterns
    years = re.findall(r"\b(1[789]\d{2}|20\d{2})\b", cleaned)
    
    # Numbers (integers and floats)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", cleaned)
    
    # Capitalized entity candidates (including hyphenated like Coster-Waldau)
    entity_candidates = re.findall(r"\b[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*(?:\s+[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)*\b", cleaned)
    
    # Check if claim has explicit negation
    lowered_tokens = [t.lower() for t in tokenize_words(cleaned)]
    has_negation = any(tok in NEGATION_WORDS for tok in lowered_tokens)
    
    return {
        "years": sorted(list(set(years))),
        "numbers": sorted(list(set(numbers))),
        "entities": sorted(list(set(entity_candidates))),
        "has_negation": has_negation
    }


def parse_fever_lines(lines_field: str) -> List[Tuple[int, str]]:
    """
    Parses the FEVER Wikipedia corpus lines format:
    '0\\tSentence 0 text\\n1\\tSentence 1 text\\n...'
    Returns list of (line_id, sentence_text).
    """
    sentences = []
    if not lines_field:
        return sentences
    
    for row in lines_field.strip().split("\n"):
        parts = row.split("\t")
        if len(parts) >= 2:
            try:
                line_idx = int(parts[0])
                sent_text = "\t".join(parts[1:]).strip()
                if sent_text:
                    sentences.append((line_idx, sent_text))
            except ValueError:
                continue
        elif len(parts) == 1 and parts[0].strip():
            sentences.append((len(sentences), parts[0].strip()))
    return sentences
