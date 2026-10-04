"""
ClaimTrace: NLP Preprocessing Module

Preserves negation, named entities, dates, numbers, and original evidence text
for accurate fact-checking and natural language inference.
"""

import re
import unicodedata
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Negation words
# ---------------------------------------------------------------------------

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "nowhere",
    "hardly",
    "scarcely",
    "barely",
    "doesn't",
    "don't",
    "didn't",
    "wasn't",
    "weren't",
    "isn't",
    "aren't",
    "cannot",
    "can't",
    "couldn't",
    "wouldn't",
    "shouldn't",
    "without",
    "refuse",
    "failed",
    "denied",
    "impossible",
    "unlikely",
}


# ---------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------

CONTRACTIONS = {
    r"\bwon't\b": "will not",
    r"\bcan't\b": "cannot",
    r"\bn['’]t\b": " not",
    r"\b['’]re\b": " are",
    r"\b['’]d\b": " would",
    r"\b['’]ll\b": " will",
    r"\b['’]ve\b": " have",
    r"\b['’]m\b": " am",
}


# ---------------------------------------------------------------------------
# Unicode normalization
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters and standardize common punctuation.

    Examples:
        Smart quotes -> regular quotes
        Em/en dashes -> hyphen
        Non-breaking spaces -> normal spaces
    """
    if not text:
        return ""

    text = str(text)

    # Standardize curly quotes.
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
    )

    # Standardize dashes.
    text = text.replace("—", " - ")
    text = text.replace("–", " - ")

    # Replace non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Normalize Unicode compatibility characters.
    text = unicodedata.normalize("NFKC", text)

    return text


# ---------------------------------------------------------------------------
# Contraction expansion
# ---------------------------------------------------------------------------

def expand_contractions(text: str) -> str:
    """
    Expand common English contractions.

    Examples:
        "didn't" -> "did not"
        "can't" -> "cannot"
        "they're" -> "they are"
    """
    if not text:
        return ""

    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return text


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_claim_text(
    text: str,
    preserve_case: bool = True,
) -> str:
    """
    Clean claim/evidence text for downstream NLP processing.

    Operations:
    - Unicode normalization
    - Contraction expansion
    - Whitespace normalization
    - Optional lowercasing

    Args:
        text:
            Input claim or evidence text.

        preserve_case:
            If True, preserve original capitalization.
            If False, convert the text to lowercase.

    Returns:
        Cleaned text.
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    text = expand_contractions(text)

    # Collapse multiple spaces, tabs, and newlines.
    text = re.sub(r"\s+", " ", text).strip()

    if not preserve_case:
        text = text.lower()

    return text


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_words(
    text: str,
    preserve_negation: bool = True,
) -> List[str]:
    """
    Tokenize text into word-like tokens.

    Preserves:
    - Alphabetic words
    - Numbers
    - Decimal numbers
    - Hyphenated words/names
    - Negation words

    Examples:
        "Nikolaj Coster-Waldau" ->
        ["Nikolaj", "Coster-Waldau"]

        "He did not play basketball." ->
        ["He", "did", "not", "play", "basketball"]

    Args:
        text:
            Input text.

        preserve_negation:
            If True, explicitly preserve negation tokens.
            Since contractions are expanded before tokenization,
            forms such as "didn't" become "did", "not".

    Returns:
        List of tokens.
    """
    if not text:
        return []

    cleaned = clean_claim_text(text, preserve_case=True)

    # Keep:
    #   words
    #   numbers
    #   decimals
    #   hyphenated terms
    #
    # Example:
    #   Coster-Waldau
    #   3.14
    #   1984
    token_pattern = r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*"

    raw_tokens = re.findall(token_pattern, cleaned)

    tokens: List[str] = []

    for token in raw_tokens:
        token = token.strip()

        if not token:
            continue

        # Negation is preserved naturally because "didn't"
        # was expanded to "did not" before tokenization.
        if preserve_negation:
            tokens.append(token)
        else:
            if token.lower() not in NEGATION_WORDS:
                tokens.append(token)

    return tokens


# ---------------------------------------------------------------------------
# Claim entities, dates, numbers, and negation
# ---------------------------------------------------------------------------

def extract_claim_entities_and_dates(
    text: str,
) -> Dict[str, List[str]]:
    """
    Extract salient surface features from a claim.

    Extracts:
    - Four-digit years
    - Numeric values
    - Capitalized entity candidates
    - Explicit negation presence

    Returns:
        Dictionary containing:
            years
            numbers
            entities
            has_negation
    """
    if not text:
        return {
            "years": [],
            "numbers": [],
            "entities": [],
            "has_negation": False,
        }

    cleaned = clean_claim_text(text, preserve_case=True)

    # -----------------------------------------------------------------------
    # Years
    # -----------------------------------------------------------------------

    years = re.findall(
        r"\b(?:1[789]\d{2}|20\d{2})\b",
        cleaned,
    )

    # -----------------------------------------------------------------------
    # Numbers
    # -----------------------------------------------------------------------

    numbers = re.findall(
        r"\b\d+(?:\.\d+)?\b",
        cleaned,
    )

    # -----------------------------------------------------------------------
    # Capitalized entity candidates
    #
    # Examples:
    #   Nikolaj
    #   Coster-Waldau
    #   Game of Thrones
    #   Michael Bay
    # -----------------------------------------------------------------------

    entity_pattern = (
        r"\b"
        r"[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*"
        r"(?:\s+[A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*)*"
        r"\b"
    )

    entity_candidates = re.findall(
        entity_pattern,
        cleaned,
    )

    # Remove obvious one-character/irrelevant candidates.
    entity_candidates = [
        entity.strip()
        for entity in entity_candidates
        if entity.strip()
    ]

    # -----------------------------------------------------------------------
    # Negation detection
    # -----------------------------------------------------------------------

    lowered_tokens = [
        token.lower()
        for token in tokenize_words(
            cleaned,
            preserve_negation=True,
        )
    ]

    has_negation = any(
        token in NEGATION_WORDS
        for token in lowered_tokens
    )

    return {
        "years": sorted(set(years)),
        "numbers": sorted(set(numbers)),
        "entities": sorted(set(entity_candidates)),
        "has_negation": has_negation,
    }


# ---------------------------------------------------------------------------
# FEVER line parsing
# ---------------------------------------------------------------------------

def parse_fever_lines(
    lines_field: str,
) -> List[Tuple[int, str]]:
    """
    Parse FEVER Wikipedia corpus line format.

    Expected format:

        0<TAB>Sentence 0 text
        1<TAB>Sentence 1 text
        2<TAB>Sentence 2 text

    Returns:
        List of:
            (line_id, sentence_text)

    The function also supports lines without an explicit
    numeric ID by assigning sequential IDs.
    """
    sentences: List[Tuple[int, str]] = []

    if not lines_field:
        return sentences

    # Normalize line endings.
    lines_field = str(lines_field).replace("\r\n", "\n")
    lines_field = lines_field.replace("\r", "\n")

    for row in lines_field.strip().split("\n"):
        row = row.strip()

        if not row:
            continue

        parts = row.split("\t")

        # Standard FEVER format:
        # ID <TAB> sentence
        if len(parts) >= 2:
            try:
                line_idx = int(parts[0])

                # Join remaining fields in case the sentence itself
                # contains tab characters.
                sent_text = "\t".join(parts[1:]).strip()

                if sent_text:
                    sentences.append(
                        (line_idx, sent_text)
                    )

            except ValueError:
                # If the first field isn't a valid integer,
                # treat the entire row as a sentence.
                sent_text = row.strip()

                if sent_text:
                    sentences.append(
                        (len(sentences), sent_text)
                    )

        # Sentence without an explicit ID.
        else:
            sent_text = parts[0].strip()

            if sent_text:
                sentences.append(
                    (len(sentences), sent_text)
                )

    return sentences