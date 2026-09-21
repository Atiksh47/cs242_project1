"""
tfidf.py

Builds the word-document table and computes TF, IDF, and TF-IDF using only
pandas/numpy, per the assignment's requirement that the TF-IDF math itself
not rely on a text-processing package (sklearn's TfidfVectorizer, gensim,
etc. are off limits - this is the from-scratch implementation).
"""

import numpy as np
import pandas as pd
from collections import Counter


# A small manually-curated stopword list. Assignment explicitly allows
# manual stopword removal without requiring a package. Extend this list
# based on what you see in your own vocabulary exploration - don't just
# trust this as complete.
STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's
with won't would wouldn't you you'd you'll you're you've your yours
yourself yourselves
""".split())


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


def build_term_document_matrix(doc_tokens: dict[str, list[str]]) -> pd.DataFrame:
    """Given {doc_name: [tokens...]}, return a raw COUNT matrix:
    rows = terms, columns = documents, values = f(t, d) raw counts.

    This is the "word-document table" the assignment asks for. TF and IDF
    are computed from this in separate steps below so each stays inspectable.
    """
    counters = {doc: Counter(tokens) for doc, tokens in doc_tokens.items()}
    df = pd.DataFrame(counters).fillna(0)
    return df.astype(int)


def compute_tf(count_matrix: pd.DataFrame) -> pd.DataFrame:
    """tf(t, d) = f(t, d) / sum_t'( f(t', d) )   -- column-wise normalization."""
    doc_totals = count_matrix.sum(axis=0)  # total term count per document
    tf = count_matrix.div(doc_totals, axis=1)
    return tf


def compute_idf(count_matrix: pd.DataFrame, log_base: float = 10) -> pd.Series:
    """idf(t, D) = log( N_D / (1 + n_t) )
    where n_t = number of documents containing term t.
    """
    n_docs = count_matrix.shape[1]
    doc_freq = (count_matrix > 0).sum(axis=1)  # n_t per term
    idf = np.log(n_docs / (1 + doc_freq)) / np.log(log_base)
    return idf


def compute_tfidf(tf: pd.DataFrame, idf: pd.Series) -> pd.DataFrame:
    """tfidf(t, d, D) = tf(t, d) * idf(t, D)"""
    return tf.mul(idf, axis=0)


def top_terms_per_doc(tfidf_matrix: pd.DataFrame, n: int = 15) -> dict[str, pd.Series]:
    """Return the top-n highest TF-IDF terms for each document, useful for
    the report's 'which term has the highest TF-IDF per document' question.
    """
    return {
        doc: tfidf_matrix[doc].sort_values(ascending=False).head(n)
        for doc in tfidf_matrix.columns
    }


def cosine_similarity_matrix(tfidf_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise cosine similarity between document TF-IDF vectors, for the
    open-ended exploration section (how similar are documents to each other).
    Computed manually with numpy - no sklearn.
    """
    docs = tfidf_matrix.columns
    vecs = tfidf_matrix.values.T  # shape: (n_docs, n_terms)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid divide-by-zero for an empty doc
    normalized = vecs / norms
    sim = normalized @ normalized.T
    return pd.DataFrame(sim, index=docs, columns=docs)