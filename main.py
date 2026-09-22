"""
main.py

Entry point. Put your six downloaded Gutenberg .txt files in books/ with
the filenames below (or edit BOOKS to match what you actually download),
then run:

    python main.py

This will:
  1. Parse + tokenize each book
  2. Build the term-document count matrix
  3. Compute TF, IDF, TF-IDF
  4. Print top terms per document
  5. Compute a cosine similarity matrix across all 6 books
  6. Save a few plots to output/ for use in your report
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from clean_tokenize import load_and_tokenize_book
from tfidf import (
    remove_stopwords,
    build_term_document_matrix,
    compute_tf,
    compute_idf,
    compute_tfidf,
    top_terms_per_doc,
    cosine_similarity_matrix,
)

# Map a short label -> filename in books/. Edit these filenames to match
# whatever you actually name the downloaded files.
#
# narrative_start is a short, unique substring marking the first sentence of
# the actual book, used to cut off the title page and table of contents that
# these Gutenberg mirrors include (see clean_tokenize.remove_table_of_contents).
# Found manually by inspecting each file - if you swap in a different edition
# of a book, re-check that the anchor still appears (and only once).
BOOKS = {
    "frankenstein":      ("gothic", "books/frankenstein.txt", "St. Petersburgh, Dec. 11th"),
    "dracula":           ("gothic", "books/dracula.txt", "How these papers have been placed in sequence"),
    "treasure_island":   ("adventure", "books/treasure_island.txt", "Squire Trelawney, Dr. Livesey, and the rest"),
    "monte_cristo":      ("adventure", "books/monte_cristo.txt", "On the 24th of February, 1815, the look-out"),
    "republic":          ("philosophy", "books/republic.txt", "The Republic of Plato is the longest"),
    "communist_manifesto": ("philosophy", "books/communist_manifesto.txt", "A spectre is haunting Europe"),
}

OUTPUT_DIR = "output"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1 & 2: parse, tokenize, remove stopwords ---
    doc_tokens = {}
    for label, (genre, path, narrative_start) in BOOKS.items():
        if not os.path.exists(path):
            print(f"[skip] {path} not found - download it and place it there.")
            continue
        print(f"Processing {label}...")
        tokens = load_and_tokenize_book(path, narrative_start)
        tokens = remove_stopwords(tokens)
        doc_tokens[label] = tokens
        print(f"  -> {len(tokens)} tokens after stopword removal")

    if not doc_tokens:
        print("\nNo books found in books/. Download the .txt files and rerun.")
        return

    # --- 3: word-document matrix, TF, IDF, TF-IDF ---
    count_matrix = build_term_document_matrix(doc_tokens)
    print(f"\nVocabulary size: {count_matrix.shape[0]} terms across {count_matrix.shape[1]} documents")

    tf = compute_tf(count_matrix)
    idf = compute_idf(count_matrix, log_base=10)
    tfidf_matrix = compute_tfidf(tf, idf)

    count_matrix.to_csv(f"{OUTPUT_DIR}/count_matrix.csv")
    tf.to_csv(f"{OUTPUT_DIR}/tf.csv")
    idf.to_csv(f"{OUTPUT_DIR}/idf.csv")
    tfidf_matrix.to_csv(f"{OUTPUT_DIR}/tfidf.csv")

    # --- 4: top terms per doc ---
    top_terms = top_terms_per_doc(tfidf_matrix, n=15)
    print("\nTop TF-IDF terms per document:")
    for doc, series in top_terms.items():
        print(f"\n  {doc}:")
        for term, score in series.items():
            print(f"    {term:<15} {score:.5f}")

    # --- 5: similarity matrix across all documents ---
    sim = cosine_similarity_matrix(tfidf_matrix)
    sim.to_csv(f"{OUTPUT_DIR}/similarity_matrix.csv")
    print("\nCosine similarity matrix:")
    print(sim.round(3))

    # --- 6: plots for the report ---
    # (a) similarity heatmap
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim.values, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(sim.columns)))
    ax.set_xticklabels(sim.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(sim.index)))
    ax.set_yticklabels(sim.index)
    ax.set_title("Document Cosine Similarity (TF-IDF vectors)")
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/similarity_heatmap.png", dpi=150)
    plt.close(fig)

    # (b) top-5 term bar chart per document
    for doc, series in top_terms.items():
        top5 = series.head(5)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top5.index[::-1], top5.values[::-1])
        ax.set_xlabel("TF-IDF score")
        ax.set_title(f"Top 5 TF-IDF terms: {doc}")
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/top_terms_{doc}.png", dpi=150)
        plt.close(fig)

    print(f"\nAll CSVs and plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()