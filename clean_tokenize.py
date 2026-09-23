"""
clean_tokenize.py

Strips Project Gutenberg boilerplate (license header/footer, front matter)
from a raw .txt file, lowercases, removes punctuation with regex, and
tokenizes into a list of terms.

everything is done with built-in re / string operations.
"""

import re
import os


# Gutenberg files wrap the actual book between these markers (format has
# been fairly consistent for years). We use regex so minor variations
# ("*** START OF THIS PROJECT GUTENBERG EBOOK X ***" vs "*** START OF THE
# PROJECT GUTENBERG EBOOK X ***") both match.
START_PATTERN = re.compile(
    r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)
END_PATTERN = re.compile(
    r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.IGNORECASE | re.DOTALL,
)


def strip_gutenberg_boilerplate(raw_text: str) -> str:
    """Return only the text between the START and END Gutenberg markers.

    Falls back to returning the full text (with a warning) if the markers
    aren't found, so you notice if a particular book uses an older format.
    """
    start_match = START_PATTERN.search(raw_text)
    end_match = END_PATTERN.search(raw_text)

    if start_match and end_match:
        body = raw_text[start_match.end():end_match.start()]
    elif start_match:
        print("  [warn] no END marker found; keeping everything after START")
        body = raw_text[start_match.end():]
    else:
        print("  [warn] no Gutenberg START/END markers found; using full text. "
              "Check this file manually for leftover boilerplate.")
        body = raw_text

    return body.strip()


def remove_table_of_contents(text: str, narrative_start: str | None = None) -> str:
    """Cut everything from the start of the file up to (and not including)
    narrative_start, which should be a short, distinctive substring that
    appears at the point where the real narrative begins 

    These anchors were found manually per book because Gutenberg mirrors are
    not consistent enough about TOC formatting for one regex to catch all of
    them reliably (see clean_tokenize testing during development). If no
    anchor is given, the text is returned unchanged.
    """
    if narrative_start is None:
        return text
    idx = text.find(narrative_start)
    if idx == -1:
        print(f"  [warn] narrative_start anchor {narrative_start!r} not found; "
              "table of contents not stripped for this book.")
        return text
    return text[idx:]


def remove_structural_noise(text: str) -> str:
    """Best-effort removal of chapter dividers made of punctuation (e.g. rows
    of underscores/asterisks used as scene breaks) and excessive whitespace.
    
    """
    # Collapse rows of repeated symbols used as dividers (----, ****, ====)
    text = re.sub(r"[-=*_]{4,}", " ", text)
    # Collapse multiple blank lines/whitespace into single spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens.

    Uses \\b\\w+\\b matching rather than str.split() so contractions and
    hyphenated words are handled predictably; digits are excluded from
    tokens (change the pattern if you want to keep numbers as terms).

    """
    text = text.lower().replace("’", "'")
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return tokens


def load_and_tokenize_book(filepath: str, narrative_start: str | None = None) -> list[str]:
    """Full pipeline for one file: read -> strip boilerplate -> strip title
    page/TOC -> clean -> tokenize."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    body = strip_gutenberg_boilerplate(raw)
    body = remove_table_of_contents(body, narrative_start)
    body = remove_structural_noise(body)
    tokens = tokenize(body)
    return tokens


if __name__ == "__main__":
    test_path = "books/frankenstein.txt"
    if os.path.exists(test_path):
        toks = load_and_tokenize_book(test_path)
        print(f"Token count: {len(toks)}")
        print(f"First 30 tokens: {toks[:30]}")
        print(f"Last 30 tokens: {toks[-30:]}")
    else:
        print(f"Put a test file at {test_path} to sanity-check cleaning.")