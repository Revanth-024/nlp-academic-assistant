# Workshop TODO Index

Task 01 established the convention. Exercises 1–12 are IMPLEMENTED / READY in complete working/reference state. Exercises 13 and 14 are ready as instructor-guided/integration experiences with no new algorithm TODOs. Exercise 15 question-paper analysis is ready; final Assistant integration is READY; Task17 reliability auditing is complete. Sequence positions below are not themselves TODO IDs.

**Synthetic Dataset — READY (Task 03).** Academic documents, course descriptions, input examples and four NLP papers are supplied. All learning/integration exercises are ready in reference form; release packaging remains Task18. No student algorithm TODO is introduced by dataset generation.

## Day 1

1. Text Preprocessing — **IMPLEMENTED / READY (reference)**
2. One-Hot Encoding, Bag of Words and N-Grams — **IMPLEMENTED / READY (reference)**
3. TF-IDF From Scratch — **IMPLEMENTED / READY (reference)**
4. Cosine Similarity and Keyword Search — **IMPLEMENTED / READY (reference)**
5. Word Embeddings / Word2Vec — **IMPLEMENTED / READY (reference)**
6. Sentence Embeddings — **IMPLEMENTED / READY (reference)**
7. Semantic Search — **IMPLEMENTED / READY (reference)**

## Day 2

8. Attention and Transformer Concepts — **IMPLEMENTED / READY (reference)**
9. Transformer NLP Experiments — **READY — instructor-guided experimentation; no algorithm TODO**
10. Document Processing and Chunking — **IMPLEMENTED / READY (reference)**
11. Semantic Chunk Retrieval — **IMPLEMENTED / READY (reference)**
12. RAG Context Construction — **IMPLEMENTED / READY (reference)**
13. Local Open-Source LLM Integration — **READY — instructor-guided optional integration; no algorithm TODO**
14. Transparent RAG Integration — **COMPLETE / READY — no new algorithm TODO**

15. Question Paper Analyzer — **IMPLEMENTED / READY (reference)**

Remaining planned topic: Intelligent Academic Assistant Integration (Task16).

## Exercise 1 — Text Preprocessing

The source contains completed `STUDENT TODO BOUNDARY` markers, not unfinished implementations. A later starter checkpoint will remove only the marked small cores, preserving signatures, documentation, validation, library objects, resource checks and UI. No duplicate solution files are used. Stems and lemmas are parallel alternatives on the same tokens.

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 1.1 | Lowercase normalization | src/preprocessing.py | ★ Guided | Day 1 |
| 1.2 | Punctuation handling | src/preprocessing.py | ★ Guided | Day 1 |
| 1.3 | Word tokenization | src/preprocessing.py | ★ Guided | Day 1 |
| 1.4 | Stopword removal | src/preprocessing.py | ★ Guided | Day 1 |
| 1.5 | Stemming | src/preprocessing.py | ★ Guided | Day 1 |
| 1.6 | Lemmatization | src/preprocessing.py | ★ Guided | Day 1 |
| 1.7 | Vocabulary + frequency | src/preprocessing.py | ★★ Core | Day 1 |

Locations: 1.1 `lowercase_text`; 1.2 `remove_punctuation`; 1.3 `tokenize_words`; 1.4 `remove_stopwords`; 1.5 `stem_words`; 1.6 `lemmatize_words`; 1.7 `build_vocabulary` and `calculate_word_frequencies`.

Validate with `python -m pytest tests/test_preprocessing.py`, inspect the Preprocessing Lab and run `notebooks/01_preprocessing.ipynb`. Read the full Exercise 1 sections in both guides. NLTK resources are installed explicitly by the starter setup helper; students do not implement downloading or POS algorithms. Exercise 2 TODOs are registered below.


## Exercise 2 — IMPLEMENTED / READY (reference)

Completed boundaries in `src/classical_nlp.py` preserve working implementations. Future starter conversion removes only student cores; index, validation, matrix assembly, comparison and UI remain provided.

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 2.1 | Multi-document vocabulary | src/classical_nlp.py | ★★ Core | Day 1 |
| 2.2 | One-hot encoding | src/classical_nlp.py | ★★ Core | Day 1 |
| 2.3 | Bag of Words | src/classical_nlp.py | ★★ Core | Day 1 |
| 2.4 | N-Grams | src/classical_nlp.py | ★★ Core | Day 1 |

Functions: 2.1 `build_vocabulary`; 2.2 `one_hot_encode`; 2.3 `bag_of_words`; 2.4 `generate_ngrams`. Validate `python -m pytest tests/test_classical_nlp.py`, lab and notebook. Unknown terms are rejected; one coordinate order is reused throughout. Exercises 3 and 4 are registered below.


## Exercises 3 and 4 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 3.1 | Term Frequency | src/classical_nlp.py | ★★ Core | Day 1 |
| 3.2 | Document Frequency | src/classical_nlp.py | ★★ Core | Day 1 |
| 3.3 | Inverse Document Frequency | src/classical_nlp.py | ★★ Core | Day 1 |
| 3.4 | TF-IDF | src/classical_nlp.py | ★★ Core | Day 1 |
| 3.5 | Query TF-IDF Vector | src/classical_nlp.py | ★★★ Challenge | Day 1 |
| 4.1 | Cosine Similarity | src/classical_nlp.py | ★★ Core | Day 1 |
| 4.2 | Query-Document Similarities | src/retrieval.py | ★★ Core | Day 1 |
| 4.3 | Rank Documents | src/retrieval.py | ★★ Core | Day 1 |

Functions in order: calculate_tf, calculate_df, calculate_idf, calculate_tfidf, build_query_tfidf_vector, cosine_similarity, score_documents, rank_documents. TF=count/token length, DF=document presence, IDF=ln(N/DF), TF-IDF=TF×IDF. Query ignores OOV and reuses corpus IDF; zero vectors score zero; ties retain corpus order. Validate `python -m pytest`, TFIDF Search Lab and shared notebook. Exercise 5 is registered below; later exercises remain planned.


## Exercise 5 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 5.1 | Prepare tokenized sentences | src/embeddings.py | ★ Guided | Day 1 |
| 5.2 | Train Word2Vec | src/embeddings.py | ★★ Core | Day 1 |
| 5.3 | Retrieve word vector | src/embeddings.py | ★ Guided | Day 1 |
| 5.4 | Find similar words | src/embeddings.py | ★★ Core | Day 1 |
| 5.5 | Word similarity | src/embeddings.py | ★★ Core | Day 1 |

Functions: prepare_tokenized_sentences, train_word2vec, get_word_vector, find_similar_words, calculate_word_similarity. Small synthetic corpus, deterministic defaults, no pretrained vector downloads. Caching and PCA are starter code. Verify `python -m pytest tests/test_embeddings.py`, the shared notebook and lab. Task 07 introduced no sentence embeddings; Exercise 6 is registered below. Semantic search is implemented in the next registered exercise.

## Student TODO Convention

Use `STUDENT TODO <module>.<exercise>` identifiers, for example `STUDENT TODO 3.2`. IDs are stable once assigned; later tasks assign IDs when exercises are designed.

- ★ Guided — scaffolded practice with direct hints.
- ★★ Core — essential independent implementation.
- ★★★ Challenge — optional deeper exploration.

Convention example using the now-assigned TODO 3.3 (its complete reference is in src):

```python
# ============================================================
# STUDENT TODO 3.3 — Calculate Inverse Document Frequency
# Difficulty: ★★ Core
# ============================================================
# Goal:
# Calculate how informative each vocabulary term is across
# the document collection.
#
# Formula:
# IDF(t) = ln(N / DF(t))  # natural logarithm
#
# Expected:
# Return IDF values aligned to the shared vocabulary order.
#
# Hint:
# Terms appearing in fewer documents should receive
# larger IDF values.
# ============================================================
```

Include Goal, Formula where relevant, Expected and Hint. Explain return values and meaningful edge cases when the exercise is designed. Task 01 contains no assigned algorithm TODOs.

## Documentation Synchronization Contract

Every future implementation task introducing a student TODO must update all five artifacts together:

1. Source code
2. `workshop/TODO_INDEX.md`
3. `docs/STUDENT_GUIDE.md`
4. `docs/INSTRUCTOR_GUIDE.md`
5. Relevant tests

This is a permanent project rule. Register only TODOs that actually exist; record their IDs, difficulty, source location, objective and validation instructions.

## Exercise 6 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 6.1 | Encode sentences | src/embeddings.py | ★★ Core | Day 1 |
| 6.2 | Sentence similarity | src/embeddings.py | ★★ Core | Day 1 |

Functions: encode_sentences and sentence_similarity. Explicit pretrained MiniLM setup/loading is starter code in src/sentence_resources.py. Keep original complete texts, N×384 shape, no extra normalization, existing manual cosine. Source, both guides and tests are synchronized. Verify preflight, tests/test_sentence_embeddings.py, the shared notebook and functional Sentence Embeddings Lab. Task 08 introduced pairwise similarity only; Exercise 7 is registered below.

## Exercise 7 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 7.1 | Encode corpus documents | src/retrieval.py | ★★ Core | Day 1 |
| 7.2 | Encode search query | src/retrieval.py | ★ Guided | Day 1 |
| 7.3 | Semantic similarities | src/retrieval.py | ★★ Core | Day 1 |
| 7.4 | Top-K semantic retrieval | src/retrieval.py | ★★ Core | Day 1 |

Functions: encode_corpus_documents, encode_search_query, calculate_semantic_similarities, retrieve_semantic_results. Same exact MiniLM encoder/revision, descriptions only, N×384 corpus, manual cosine and existing stable ranking. Model/cache/UI/evaluation are starter infrastructure. Validate tests/test_semantic_retrieval.py, shared notebook and side-by-side lab. Instructor evaluator/report live in workshop/. No Day-2/Task-10 TODO is introduced.


## Exercises 8–9 — IMPLEMENTED / READY

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 8.1 | Attention scores | src/transformers_nlp.py | ★★ Core | Day 2 |
| 8.2 | Softmax attention weights | src/transformers_nlp.py | ★★ Core | Day 2 |
| 8.3 | Weighted context representation | src/transformers_nlp.py | ★★ Core | Day 2 |

Exercise 9: **Instructor-guided pretrained Transformer experimentation — no algorithm TODO**. Resources, pipelines, inference and caching are starter code. Simplified attention is educational, not measured BERT/GPT attention. Tasks through Exercise 9 registered 33 algorithm TODO IDs. Exercise 10 is registered below.

## Exercise 10 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 10.1 | Conservative extracted-text cleaning | src/document_processor.py | ★ Guided | Day 2 |
| 10.2 | Word-window chunking | src/document_processor.py | ★★ Core | Day 2 |
| 10.3 | Overlapping chunks | src/document_processor.py | ★★ Core | Day 2 |
| 10.4 | Chunk provenance metadata | src/document_processor.py | ★★ Core | Day 2 |

Exactly four new boundaries; 37 algorithm TODO IDs total. PDF extraction, uploads, offsets, token analysis and statistics are starter infrastructure. Validate tests/test_document_processor.py and notebooks/08_rag.ipynb. Task 11 left chunk embeddings/retrieval planned; Exercise 11 below now implements them. Later RAG stages remain planned. Source, both guides, tests and this index are synchronized.

## Exercise 11 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 11.1 | Encode document chunks | src/retrieval.py | ★★ Core | Day 2 |
| 11.2 | Encode user question | src/retrieval.py | ★ Guided | Day 2 |
| 11.3 | Question-chunk similarity | src/retrieval.py | ★★ Core | Day 2 |
| 11.4 | Retrieve Top-K evidence | src/retrieval.py | ★★ Core | Day 2 |

Exactly four new boundaries; 41 algorithm TODO IDs total. Model setup, chunk processing, token checks, caching and evaluator are infrastructure. Verify tests/test_chunk_retrieval.py and notebooks/08_rag.ipynb. The same encoder/cosine/ranking power course and passage retrieval. Exercise12/context and optional generation are now implemented below; no Task13 algorithm TODO exists.

## Exercise 12 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 12.1 | Combine retrieved chunks into context | src/rag.py | ★★ Core | Day 2 |
| 12.2 | Preserve source labels/provenance | src/rag.py | ★★ Core | Day 2 |
| 12.3 | Build grounded prompt | src/rag.py | ★★★ Challenge | Day 2 |

Exactly three new boundaries; 44 algorithm TODO IDs total. Budgeting/mapping/model loading/generation are starter infrastructure. Verify tests/test_rag.py and tests/test_local_llm.py plus notebooks/08_rag.ipynb. Exercise 13 is instructor-guided optional local LLM integration with **no low-level algorithm TODO**. Task14 transparent RAG presentation is now implemented below.

## Exercise 14 — COMPLETE / READY (integration)

No new algorithm TODOs. Reuse Exercises 10–13; **44 algorithm TODO IDs remain**. Integration orchestration, deterministic attribution, source-label inspection, diagnostics and UI/session-state are starter infrastructure. Validate tests/test_transparent_rag.py and the complete shared RAG notebook. Question-paper analysis and final Assistant are implemented below.

## Exercise 15 — IMPLEMENTED / READY (reference)

| TODO | Concept | File | Difficulty | Day |
|---|---|---|---|---|
| 14.1 | Normalize question text | src/question_analyzer.py | ★ Guided | Day 2 |
| 14.2 | Match topic aliases | src/question_analyzer.py | ★★ Core | Day 2 |
| 14.3 | Count primary topic frequencies | src/question_analyzer.py | ★★ Core | Day 2 |
| 14.4 | Rank topics deterministically | src/question_analyzer.py | ★ Guided | Day 2 |
| 14.5 | Filter questions with provenance | src/question_analyzer.py | ★★ Core | Day 2 |

Exercise 15 deliberately uses TODO prefix 14; earlier IDs are unchanged. **49 algorithm TODO IDs total.** Extraction/reporting/UI/evaluation are infrastructure. Validate tests/test_question_analyzer.py and the app. No model, new dependency or ninth notebook. Ground truth is separate evaluation only. Final Assistant integration is complete in Task16.

## Final Integration Exercise — COMPLETE / READY

Task16 integrates Academic Q&A / course navigation, Previous Papers and grounded Study Support on page 10. **No new algorithm TODOs; 49 IDs remain frozen.** Reuse transparent RAG and deterministic paper analysis. Routing, resource loading, caching and UI are starter infrastructure. Validate `tests/test_academic_assistant.py` and the 20-case evaluator. Tasks 17–18 remain future reliability/release work.


## Task17 — Reliability Audit COMPLETE

Exactly49 unique algorithm boundary headers and49 index rows. TODO1.7 contains two separately labelled BEGIN/END core regions; all50 regions are mechanically identifiable. Its second function now uses a PART header rather than duplicating the BOUNDARY header. No logic stripped or IDs changed. Older milestone totals describe development history; current total49. Automated checks: `tests/test_reliability.py` verifies index/guides/core pairing, notebook reuse and the Streamlit-free source boundary. No new algorithm TODO.

## Task18 — Final Packaging

49 algorithm IDs remain frozen. Instructor/reference source remains complete. `python -m workshop.build_student_starter` produces student TODO scaffolds deterministically; the generated package has50 stub regions under49 unique IDs, preserving signatures/docstrings/formulas/hints. No new algorithm TODO. Reference-only guide/evaluators/measurements are excluded; START_HERE and per-exercise tests are supplied. Validate with workshop.validate_student_starter and review RELEASE_VALIDATION.md before owner publication.
