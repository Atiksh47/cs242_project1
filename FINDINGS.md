# Project Findings Summary (for report drafting)

Reference notes only — not a finished report. Rewrite in your own words and
pull specific numbers from `output/` as needed.

## Corpus

| Book | Genre | Raw size |
|---|---|---|
| Frankenstein (Mary Shelley) | Gothic | 420 KB |
| Dracula (Bram Stoker) | Gothic | 852 KB |
| Treasure Island (R.L. Stevenson) | Adventure | 372 KB |
| The Count of Monte Cristo (Alexandre Dumas) | Adventure | 2.7 MB |
| The Republic (Plato, trans. Jowett) | Philosophy | 1.2 MB |
| The Communist Manifesto (Marx & Engels) | Philosophy | 76 KB |

Total corpus ≈ 5.6 MB, well over the 250 KB minimum. Books were chosen in
three genre pairs (gothic, adventure, philosophy) to support a
"compare books with different subjects" style analysis.

## 1. Parsing

**Table of contents removal.** Every book's raw `.txt` had a title page and
table of contents at the top that was being tokenized as if it were real
content — most noticeably in *The Count of Monte Cristo*, whose 117-chapter
TOC repeats character names (Danglars, Valentine, Villefort) that were
already going to be the book's most distinctive terms. Fixed by adding
`remove_table_of_contents()` to `clean_tokenize.py`, which cuts each file at
a manually verified, unique anchor string marking where the real narrative
begins (one anchor per book — Gutenberg mirrors aren't consistent enough
in TOC formatting for one general rule to catch all six reliably).

**Apostrophe normalization bug.** The tokenizer's regex only matched a
straight apostrophe (`'`), but these files use the curly apostrophe (`'`,
U+2019) in contractions, so `I'll` was being split into two tokens, `i` and
`ll`. Since `"ll"` isn't a stopword, it survived and ranked as the **#2
term in Treasure Island** by TF-IDF. Fixed with a one-line normalization
(`'` → `'`) before tokenizing in `clean_tokenize.py`. After the fix, "ll"
is gone from the rankings, replaced by real terms like `cap'n`.

**No Gutenberg license boilerplate.** All six files came from a mirror
that doesn't include the standard `*** START/END OF THE PROJECT GUTENBERG
EBOOK ***` banner, so that stripping step never fires (prints a fallback
warning instead) — confirmed by inspecting the raw files directly, not a
bug, just nothing to strip.

**Vocabulary/genre observations** (from `output/tfidf.csv` and
`output/top_terms_*.png`): gothic and adventure novels surface almost
entirely character names as top terms (Frankenstein → elizabeth, clerval,
justine; Dracula → helsing, lucy, mina; Treasure Island → jim, squire,
livesey; Monte Cristo → monte, villefort, danglars). Philosophy/political
texts surface conceptual vocabulary instead (The Republic → plato,
socrates, injustice, citizens; Communist Manifesto → bourgeoisie,
proletariat, socialism, exploitation). This is a clean illustration of
TF-IDF at work: character names are rare across the corpus (low document
frequency → high IDF) but frequent within their own book (high TF), so
they dominate fiction; recurring political/philosophical terms play the
same role in the two non-fiction texts.

## 2. Vectorization (TF)

`compute_tf` divides each term's raw count by the document's total token
count (column-wise normalization), matching the assignment formula.
Sorting TF within a document is already meaningful here (proper nouns and
thematic terms, not "the"/"and") because stopwords are removed before the
count matrix is built. TF alone can't tell "important to this document"
from "common everywhere" — a word like "said" would rank high by raw TF in
every narrative-heavy book without being distinctive to any one of them,
which is the gap IDF closes.

## 3. IDF

`compute_idf` uses `log10(N_D / (1 + n_t))`, matching the assignment
formula. With `N_D = 6`, a term unique to one document gets the maximum
IDF (`log10(6/2) ≈ 0.477`), while a term appearing in all 6 documents gets
a slightly **negative** IDF (`log10(6/7) ≈ -0.067`), since `6/7 < 1`.

This has a small but noticeable effect on the cosine similarity matrix: a
handful of very common words (present in every book) end up with negative
TF-IDF scores in every document, and two negatives multiply to a positive
contribution to similarity — which is part of why, e.g., *The Republic*
scores unexpectedly high in similarity against every other book (0.296
with Frankenstein, 0.243 with Dracula, etc.), higher than some pairs that
share a genre label. A standard fix is additive IDF smoothing (`+1`, so
IDF is never negative), but that comes with its own tradeoff: it also
suppresses the sharp per-document top-terms ranking that raw IDF produces
well (character names lose to generic words like "will"/"said" once IDF
stops penalizing them so heavily). Given that tradeoff, the raw IDF
formula was kept as the primary result, with this as a noted limitation
of working with a small (6-document) corpus.

## 4. TF-IDF

`compute_tfidf` multiplies `tf` elementwise by `idf`, matching
`tf(t,d) · idf(t,D)`. Highest TF-IDF term per document (see
`output/tfidf.csv` for exact scores):

- Frankenstein → **elizabeth**
- Dracula → **helsing**
- Treasure Island → **jim**
- Monte Cristo → **monte**
- The Republic → **plato**
- Communist Manifesto → **bourgeoisie**

The Communist Manifesto's top score is noticeably higher in absolute terms
than any other book's — it's by far the shortest document, so a repeated
term makes up a much larger share of its total token count, inflating TF
(and therefore TF-IDF). Worth a caveat when comparing absolute TF-IDF
magnitudes across documents of very different lengths.

## 5. Exploration — Cosine Similarity

Full matrix saved to `output/similarity_matrix.csv`, visualized in
`output/similarity_heatmap.png`:

```
                     frankenstein  dracula  treasure_island  monte_cristo  republic  communist_manifesto
frankenstein                1.000    0.194            0.199         0.136     0.296                0.045
dracula                     0.194    1.000            0.209         0.115     0.243                0.033
treasure_island             0.199    0.209            1.000         0.127     0.247                0.035
monte_cristo                0.136    0.115            0.127         1.000     0.167                0.020
republic                    0.296    0.243            0.247         0.167     1.000                0.069
communist_manifesto         0.045    0.033            0.035         0.020     0.069                1.000
```

The Communist Manifesto is the clear outlier, with the lowest similarity
to every other book (0.02–0.07) — it shares almost no character-name
vocabulary with any novel and is also the shortest, sparsest document. The
Republic scoring highest with Frankenstein (0.296) rather than with the
other gothic novel, Dracula, is the one counterintuitive result worth
calling out explicitly — see the IDF section above for the most likely
explanation (small-corpus negative-IDF effect on shared common
vocabulary).

## Reflection material (Section 5.1 prompts)

Rough pointers to reuse/expand in your own words:
1. *Something you understand differently now*: with a small corpus, IDF
   can go negative for terms that happen to appear in every document,
   which can distort similarity comparisons in ways that aren't obvious
   from the formula alone.
2. *A decision you made*: choosing explicit per-book anchors to strip the
   table of contents, rather than one general regex — the formatting
   wasn't consistent enough across all six books for a single rule to work
   safely.
3. *A moment of uncertainty*: the Republic scoring more similar to
   Frankenstein than to another gothic novel — resolved by tracing it to
   the negative-IDF effect on words common to the whole corpus rather than
   assuming it reflected a real thematic connection.
4. *Looking back*: given more time, n-gram TF-IDF (bigrams/trigrams) would
   likely separate genres more cleanly than single-word terms, since
   common single words end up dominating cross-document comparisons
   regardless of IDF formula choice.
