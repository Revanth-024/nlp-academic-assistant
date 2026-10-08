# Workshop Overview

**From Classical NLP to Generative AI: Building an Intelligent Academic Assistant** is a 2-Day Hands-On NLP Workshop for Engineering Students. This guide evolves alongside implementation tasks. Tasks 01–10 provide the foundation and Exercises 1–9; later document processing and assistant modules remain planned.

# Learning Objectives

- Explain and inspect successive numerical representations of text.
- Implement core classical NLP concepts before educational library comparisons.
- Compare keyword and semantic retrieval.
- Explore attention and Transformer experiments.
- Trace document retrieval and transparent RAG.
- Integrate, test and present an academic assistant with evidence and sources.

# Workshop Philosophy

1. Students implement NLP concepts, not infrastructure. Starter code handles layouts, navigation, session state, formatting, plots, uploads, validation, environment handling, model downloads/caching/loading and deployment configuration when introduced.
2. Manual implementation comes before library comparison where educationally useful: manual TF-IDF → compare with scikit-learn.
3. One implementation in `src/` powers notebooks, tests and Streamlit. Do not duplicate algorithms in notebooks or pages.
4. Every major lab should expose Input → Intermediate Representation → Processing → Scores/Calculations → Result.
5. The same synthetic university dataset will later support TF-IDF Search → Semantic Search → RAG → Academic Assistant.
6. RAG must remain transparent: Question → Chunks → Embeddings/Retrieval → Similarity Scores → Top-K Chunks → Constructed Context → Generated Answer → Sources.
7. No paid service may be required to complete the workshop.

The fictional institution is **Hindu College of Engineering**. Task 02 defines its canonical facts in `data/university_facts.json`; Task 03 supplies the derived Markdown documents. Every synthetic academic document must explicitly describe the institution as fictional and contain: **Synthetic educational data created for NLP workshop purposes.**

No local LLM is selected yet. Later evaluation must consider CPU usability, memory, download size, Windows and student laptop compatibility, Streamlit deployment feasibility and generation quality. Retrieval must remain independently usable when generation is unavailable.

# Final Project

**NLP-Based Intelligent Academic Assistant Using Semantic Search, Transformers and Retrieval-Augmented Generation**. Current state: shell, placeholders, workshop framework, canonical synthetic fact contract and generated Markdown dataset. Preprocessing, BoW/N-Grams, TF-IDF Search, Word2Vec, Sentence Similarity and Semantic Search are implemented in working/reference form; later NLP implementations remain pending.

# Synthetic Dataset Teaching Strategy

The same fictional university data will later power TF-IDF Search → Semantic Search → RAG → Academic Assistant → Question Paper Analyzer. Keeping the academic world constant lets students focus on how representations and retrieval methods change, compare results against the same facts, and trace generated answers back to evidence.

`data/university_facts.json` is the single source of truth for curricula, policies, syllabi, faculty information, calendars and papers. New in-scope facts must be added and validated there before documents use them. Task 03 documents are derived views, not independent sources of invented policy values. The contract itself is a validation/design artifact, not content to index as a substitute for searching documents.

Some benchmark queries deliberately use different terminology from the future source passages. For example, a query about interpreting human language should retrieve the NLP course's computational-methods description. Compare lexical and semantic rankings empirically; paraphrasing is a teaching opportunity, not a guaranteed semantic-search win. Benchmarks also include multi-source questions requiring both calendar and syllabus evidence.

Question-paper targets prescribe primary-topic counts for the four generated synthetic papers. They are design ground truth, not real historical examination statistics. When the analyzer is implemented, compare its independently computed counts with those targets and discuss related-topic overlap.

No student coding exercise or NLP TODO was required for Tasks 02–03; Task 04 introduces Exercise 1 below. Review the course placements, attendance/pass policy, calendar and primary-topic counting convention in [Synthetic Data Design](SYNTHETIC_DATA_DESIGN.md). Preserve the contract and its consistency tests when adding derived materials.

## Synthetic Academic Dataset

The institution, location, faculty, policies and previous-paper sets are entirely fictional. This avoids presenting workshop material as authoritative real university advice and allows deliberately consistent, inspectable retrieval scenarios. All 14 academic Markdown documents and four papers carry the exact synthetic-data statement and an explicit fictional-institution disclaimer. Markdown is canonical; selective PDF extraction comes later.

Canonical facts are structured in JSON. Documents present those facts in distributed, natural language: attendance detail belongs primarily to the policy, content to syllabi, dates to the calendar and dependencies to the prerequisite register. Overlap reinforces a fact without requiring every document to repeat every policy. A multi-source question should preserve those boundaries.

The 20-row `data/search_corpus/course_descriptions.csv` provides short, contrasting retrieval items that students can inspect together. Use it later to compare TF-IDF and semantic retrieval without the additional complexity of document chunking. Each name, code, semester, credit value, description and keyword list follows the fact contract. Queries may paraphrase the descriptions; meaning-based retrieval is worth comparing empirically, not assumed to outperform lexical retrieval on every query.

The full documents are reserved for later chunking, retrieval and RAG experiments. They contain longer sections and information distributed across sources. `data/document_manifest.json` identifies the 14 academic records and four papers. It does not contain full text. Use actual Markdown content as evidence, excluding contract, benchmark and question-label artifacts from the answer corpus.

Every question paper contains twelve compulsory five-mark questions with no choice branches. The four papers match the per-paper primary-topic targets exactly, while varying wording, question verbs, aliases and implicit descriptions. They are undated synthetic archive-style papers, not records of four actual sittings. Topic overlap remains an instructional opportunity: primary counting and related-topic filtering are distinct tasks.

**Permanent ground-truth rule:** the future Question Paper Analyzer must not load `question_metadata.json` or frequency targets to produce results. It must extract questions, identify topics, calculate frequencies and rank them from paper text. Tests may use metadata for evaluation. Likewise, RAG must retrieve document evidence rather than reading expected benchmark references as answers. Do not demonstrate labels as if an unimplemented analyzer computed them.

`data/retrieval_benchmarks.json` maps all 15 canonical queries to existing document IDs and adds three supplementary queries, for 18 total. Chunk IDs remain undefined. Use the benchmarks to inspect source coverage now, then record actual retrieval behaviour after algorithms exist. Demonstration queries include:

- “What attendance percentage is required?” — inspect the dedicated policy.
- “Which subject teaches computers to interpret human language?” — contrast query wording with the NLP description.
- “Which class learns to predict outcomes from examples?” — inspect the ML description.
- “Where do connected devices learn the rules for exchanging messages?” — discuss the networks profile and query ambiguity.
- “When is the NLP examination and what topics are included?” — inspect calendar and syllabus together.
- “Compare the focus of Machine Learning and Deep Learning.” — inspect both syllabi.
- “Summarize Unit 3 of NLP.” — locate the distributed-word-representation section.
- “Show questions related to transformers.” — inspect paper text and distinguish primary from related topics.

No algorithm execution is implied by this dataset inspection. Task 03 supplies the knowledge environment and introduces no student coding exercise.

# Architecture Overview

```text
Streamlit UI
     ↓
src modules
     ↓
NLP logic
```

Pages may import `src` functions. `src` must never import Streamlit. This keeps NLP code reusable, independently testable and accessible from notebooks.

# Day 1 Teaching Journey

1. Text Preprocessing — IMPLEMENTED / READY (reference)
2. One-Hot Encoding, Bag of Words and N-Grams — IMPLEMENTED / READY (reference)
3. TF-IDF From Scratch — IMPLEMENTED / READY (reference)
4. Cosine Similarity and Keyword Search — IMPLEMENTED / READY (reference)
5. Word Embeddings / Word2Vec — IMPLEMENTED / READY (reference)
6. Sentence Embeddings — IMPLEMENTED / READY
7. Semantic Search — IMPLEMENTED / READY

Move from raw text to interpretable counts and weights, then learned word vectors and later sentence retrieval. Exercises 1–7 have assigned boundaries and complete references; Day-2 exercises remain planned.

# Day 2 Teaching Journey

8. Attention and Transformer Concepts — READY
9. Transformer NLP Experiments — READY (instructor-guided; no algorithm TODO)
10. Document Processing and Chunking — PLANNED
11. Semantic Chunk Retrieval — PLANNED
12. RAG Context Construction — PLANNED
13. Local Open-Source LLM Integration — PLANNED
14. Question Paper Analyzer — PLANNED
15. Intelligent Academic Assistant Integration — PLANNED

Connect model concepts to document evidence and the integrated assistant. Keep retrieval observable and usable independently of generation.

# Exercise 1 — Text Preprocessing

### Topic and Why We Teach This

Raw text cannot immediately become most classical numerical representations. Token boundaries, consistent forms and a vocabulary determine what later counts and vectors mean. This exercise makes those decisions visible before students start mathematical representations. It also establishes a reusable implementation shared by tests, notebooks and Streamlit.

### Connection to Previous Topic

Tasks 01–03 provide the shell and one coherent fictional academic dataset. Use its short input examples now; the same vocabulary-building ideas will later apply to course descriptions. No search or ranking algorithm is needed in this exercise.

### Intuition and Simple Example

Start with `NLP is useful! NLP studies languages.` Lowercasing gives `nlp is useful! nlp studies languages.` Punctuation becomes spaces, then tokenization gives `["nlp", "is", "useful", "nlp", "studies", "languages"]`. English filtering removes `is`. From the same remaining tokens, compare stems with noun lemmas. Do not chain the two morphology branches.

The selected lemma representation is `["nlp", "useful", "nlp", "study", "language"]`. Its sorted vocabulary is `["language", "nlp", "study", "useful"]`; `nlp` occurs twice while each other token occurs once. The unique-word count and occurrence count answer different questions.

### Important Concepts and Mathematics

- Lowercasing changes case only; it may discard useful information about names and acronyms.
- This punctuation policy replaces characters with spaces so neighbours do not merge. Unicode punctuation and ASCII punctuation are recognized by a supplied helper. Whitespace is not silently collapsed.
- NLTK WordPunct tokenization returns ordered units. It needs no Punkt resource, but can split decimal points and hyphens, and can emit preserved punctuation tokens.
- Stopword filtering is task-dependent, case-insensitive for membership and preserves retained spelling/order. English filtering removes negation such as `not`.
- Porter stemming is heuristic and may yield non-dictionary stems. It also lowercases by default.
- WordNet lemmatization uses a grammatical assumption; noun is the default. A verb assumption can change `learning` to `learn`, but assigning one POS to every token is not POS-aware sentence analysis.
- Vocabulary is a unique token set with deterministic sorted order.
- `frequency(t)` is the number of token occurrences equal to `t`. The sum of frequencies equals the selected token count, while vocabulary size counts distinct tokens.

### What Students Implement

Boundaries 1.1–1.6 are ★ Guided; boundary 1.7 is ★★ Core and covers two small functions. Students apply library operations and write filtering/counting logic; they do not implement tokenization, Porter rules, WordNet internals, POS tagging or resource downloading.

### What Starter Code Provides

Signatures, docstrings, examples, input checks, the punctuation predicate, resource verification, the stopword set, stemmer/lemmatizer objects, error messages, optional POS support, all UI, comparison/frequency formatting and notebook import setup are provided. `src` never imports Streamlit. Resource downloads are restricted to the explicit setup command.

### TODO IDs and Reference Implementation

The repository remains complete and tested. `STUDENT TODO BOUNDARY` labels identify eventual removals; `BEGIN/END STUDENT CORE` delimit only the educational statements. Do not distribute this reference checkout as an unsolved starter or replace its only implementation with `pass`.

| TODO | Function | Reference solution logic |
|---|---|---|
| 1.1 | `lowercase_text` | Return the string's lowercase form. |
| 1.2 | `remove_punctuation` | Visit each character; append a space for supplied punctuation matches and retain other characters; join. |
| 1.3 | `tokenize_words` | Return `wordpunct_tokenize(text)`. |
| 1.4 | `remove_stopwords` | Return a new list of tokens whose lowercase form is absent from the supplied set. |
| 1.5 | `stem_words` | Apply the provided stemmer's `stem` method to every token. |
| 1.6 | `lemmatize_words` | Apply the provided lemmatizer's `lemmatize(token, pos=pos)` inside retained error handling. |
| 1.7 | `build_vocabulary` | Return sorted unique tokens. |
| 1.7 | `calculate_word_frequencies` | Initialize a mapping, increment for every occurrence, return sorted keys. |

Instructor reference cores, matching the existing source:

```python
# 1.1
return text.lower()

# 1.2
characters = []
for character in text:
    characters.append(" " if _is_punctuation(character) else character)
return "".join(characters)

# 1.3
return wordpunct_tokenize(text)

# 1.4
return [token for token in tokens if token.lower() not in stopword_set]

# 1.5
return [stemmer.stem(token) for token in tokens]

# 1.6 — inside the retained try/except
return [lemmatizer.lemmatize(token, pos=pos) for token in tokens]

# 1.7 — vocabulary
return sorted(set(tokens))

# 1.7 — frequency
counts = {}
for token in tokens:
    counts[token] = counts.get(token, 0) + 1
return dict(sorted(counts.items()))
```

### Intended Student Starter and Recovery

For every function, preserve the signature, full docstring, boundary explanation, hints and all surrounding infrastructure. Replace only marked core statements with an actual `STUDENT TODO 1.x` and an explicit unfinished sentinel. For lemmatization retain the resource check, lemmatizer and error-handling block; place the student sentinel inside the retained `try`. Boundary 1.7 requires both the vocabulary and frequency cores. The Student Guide lists each preserved signature and scaffold.

The Git starter/checkpoint strategy remains a later task. There are no duplicated `solution_*.py` files. Until checkpoints are established, retain the current working reference and use its functions/tests to diagnose a student's misunderstanding. Preserve student work before restoring a known-working checkpoint; do not overwrite the reference to simulate a starter.

### Teaching Steps and Live Demonstration

1. Verify resources before class and open the Preprocessing Lab.
2. Ask students to predict what lowercasing changes, then display only that stage first.
3. Add `human-language` and repeated punctuation; discuss replacement versus deletion.
4. Tokenize and identify repeated tokens before filtering.
5. Toggle stopwords for `not useful`; ask whether the remaining phrase preserves meaning.
6. Compare `studies` and `processing` in the stem/lemma table; switch noun to verb POS and state the assumption explicitly.
7. Select filtered tokens, stems and lemmas for vocabulary/frequencies. Explain why changing representation can change counts and vocabulary.
8. Repeat `nlp`; demonstrate an unchanged unique vocabulary but increased frequency.
9. Locate the seven source boundaries, run preprocessing tests, then inspect the notebook's shared imports.
10. Optionally open the POS section for the fictional faculty sentence. Treat predictions as model output, not authoritative linguistic annotation.

### Expected Output

| Original | Porter stem | Noun lemma | Verb lemma |
|---|---|---|---|
| studies | studi | study | study |
| processing | process | processing | process |
| learning | learn | learning | learn |

The simple sentence example has five selected lemma tokens, four vocabulary items and total frequency five. Empty input has no tokens, vocabulary or counts. A missing corpus yields clear setup guidance rather than fake outputs. Bypassed operations are labelled in the lab.

### Questions to Ask Students

- Should we always lowercase text?
- Can punctuation contain meaning?
- Should stopwords always be removed?
- Why can stemming produce strange-looking words?
- Why might lemmatization need grammatical context?
- What happens to vocabulary size after preprocessing?
- Why must vocabulary ordering be deterministic for later vectors?
- Why must we count repetitions before discarding duplicates?

### Common Mistakes

- Passing a raw string to a token-list function, accidentally treating characters as words.
- Applying punctuation handling inconsistently before/after tokenization.
- Deleting punctuation and merging neighbouring words.
- Modifying a list while iterating during filtering.
- Deduplicating tokens before frequency calculation.
- Assuming all stems are English dictionary words.
- Assuming lemmatization changes every token or automatically identifies POS.
- Passing stems into the lemmatizer instead of comparing both on the original tokens.
- Using an unsorted set as the final vocabulary.
- Assuming the optional POS model performs NER or recognizing invented names perfectly.

### Troubleshooting

Run `python -m src.nltk_resources --download` once, then `python -m src.nltk_resources` to verify the two core resources. The helper stores local data in ignored `.nltk_data/`, recognizes WordNet ZIP corpora and retains standard NLTK search locations. There is no need for `punkt`, `punkt_tab`, `omw-1.4` or a spaCy model in this English exercise.

For optional POS, add `--include-pos` to explicit setup; the resource is `averaged_perceptron_tagger_eng`, not the older unnamed-language package. If absent, the POS section shows guidance and the core lab continues. For unreliable classroom internet, prepare and copy the same `.nltk_data` folder with resources before class. For corrupt/incomplete resources, replace the affected folder/ZIP with a working copy and verify again. Check proxy access and folder permissions if downloads fail.

Notebook import errors usually mean the kernel is outside the repository root or `notebooks/`; use the supplied bootstrap and the workshop environment's kernel. An unchanged noun lemma for a verb is an assumption issue, not necessarily an installation failure. Avoid changing corpus handling or UI during the student exercise.

### POS/NER Decision and Limitation

POS is a secondary instructor demonstration using NLTK's small English tagger, installed only on request. No automatic POS-aware lemmatization algorithm is required. NER is deferred to the Transformer task, avoiding spaCy and an additional large model now. The simple tokenizer/punctuation policy is not a universal multilingual or domain-specific preprocessing solution. Educational clarity takes precedence over hiding these limitations.

### Syllabus Refinement

NLP Unit 4 now contrasts BERT's encoder-oriented masked modeling/contextual representations with GPT's decoder-oriented autoregressive next-token prediction/text generation. GPT is an alias/subtopic under the existing `language_models` taxonomy ID, not a new primary question-paper topic. The five-unit structure, 16 primary topics and approved paper-frequency targets remain intact. No GPT inference is introduced.

### Key Takeaway and Transition

Preprocessing consists of visible, task-dependent decisions that determine the representation's units and information. One tested source implementation serves every workshop surface.

> We now have clean tokens and a vocabulary. But computers still cannot directly perform mathematical comparison on words. Next we need to represent language numerically.

This motivates **One-Hot Encoding → Bag of Words → N-Grams**; Exercise 2 now implements these representations.

# Exercise Template

Copy this section when a concrete exercise is introduced. Do not assign placeholder TODO IDs.

## Exercise X — Name

### Topic

To be supplied when the exercise is designed.

### Why We Teach It

To be supplied when the exercise is designed.

### Connection to Previous Topic

To be supplied when the exercise is designed.

### Intuition

To be supplied when the exercise is designed.

### Important Mathematics

To be supplied when the exercise is designed.

### Simple Example

To be supplied when the exercise is designed.

### What Students Implement

To be supplied when the exercise is designed.

### What Starter Code Provides

To be supplied when the exercise is designed.

### TODO IDs

To be supplied when the exercise is designed.

### Teaching Steps

To be supplied when the exercise is designed.

### Live Demonstration

To be supplied when the exercise is designed.

### Expected Output

To be supplied when the exercise is designed.

### Questions to Ask Students

To be supplied when the exercise is designed.

### Common Mistakes

To be supplied when the exercise is designed.

### Troubleshooting

To be supplied when the exercise is designed.

### Key Takeaway

To be supplied when the exercise is designed.

### Limitation

To be supplied when the exercise is designed.

### Transition to Next Topic

To be supplied when the exercise is designed.

# Instructor Recovery Strategy

If students fall behind, help them identify the last working checkpoint and retain their current work in a commit or separate branch before recovery. The instructor should be able to provide or restore a known-working checkpoint/reference implementation, run its tests, and resume from the next concept. Explain what was restored so students can revisit unfinished learning.

The exact Git solution/checkpoint strategy will be decided later. Do not create duplicated `solution_*.py` files in this student repository. Prepare and verify the agreed reference checkpoints before teaching once the strategy is selected.

# Documentation Synchronization Contract

Every future implementation task introducing a student TODO must update all five artifacts together:

1. Source code
2. `workshop/TODO_INDEX.md`
3. `docs/STUDENT_GUIDE.md`
4. `docs/INSTRUCTOR_GUIDE.md`
5. Relevant tests

This is a permanent project rule. Register only TODOs that actually exist; record their IDs, difficulty, source location, objective and validation instructions.


# Exercise 2 — One-Hot Encoding, Bag of Words and N-Grams

## Topic, why we teach it, and connection

We cleaned language and produced tokens. Now we need numbers. Manual representations make coordinates, sparsity, counts and lost information visible before libraries automate the process. Input → Tokens → Shared Vocabulary → Numbers → Matrix → Interpretation is the teaching flow.

## Vocabulary: the coordinate system

One sorted vocabulary across all documents defines what each column means. `create_vocabulary_index` exposes this mapping. Rebuilding vocabulary per document destroys alignment. Exercise 1's flat-token utility is reused after flattening the corpus in TODO 2.1; students implement only the flatten-and-delegate core.

## One-hot encoding: dimensions and sparsity

TODO 2.2 creates V zeros and activates the known word's index. Length is V; exactly one coordinate is one. Mathematical form: coordinate j is 1 if j equals the selected index, otherwise 0. Most entries are zero for a large vocabulary. Car and automobile have independent axes; no semantic closeness is encoded.

## Bag of Words and document-term matrix

TODO 2.3 initializes V counts, visits tokens and increments indexed positions. Coordinate j counts occurrences of vocabulary term j. The row sum equals document token count. Starter matrix assembly calls `bag_of_words` per document: rows are documents, columns are shared terms. Python lists avoid introducing array mechanics into the student cores.

## N-Grams: limited local order

TODO 2.4 uses complete sliding windows and returns tuples, preserving repetitions. Number of windows is max(0, L-n+1). n=1 gives one-element tuples; n>L gives none. Windows must not cross documents. Bigrams distinguish dog bites/man bites while unigram counts do not. More possible phrases increase dimensionality; local windows still do not model arbitrary long-range meaning.

## Reference implementation logic

Completed solution cores are bounded in `src/classical_nlp.py`:

```python
# 2.1: after starter document validation
tokens = [token for document in documents for token in document]
return unique_tokens(tokens)
# 2.2: after starter index/unknown validation
vector = [0] * len(vocabulary)
vector[index[word]] = 1
return vector
# 2.3: after starter index/unknown validation
vector = [0] * len(vocabulary)
for token in tokens:
    vector[index[token]] += 1
return vector
# 2.4: after starter token/n validation
return [tuple(tokens[start:start + n]) for start in range(len(tokens) - n + 1)]
```

Students implement only these small cores. Starter code provides validation, index mapping, matrix assembly, library configuration, formatting, examples, notebook structure and Streamlit controls. Preserve completed working/reference state until the dedicated starter checkpoint task. Remove only BEGIN/END cores in the eventual starter; no duplicate solution modules.

## CountVectorizer: manual vs library

Teach manual understanding → library automation → verification. Our callable `analyzer=list` returns already-tokenized input; a fixed vocabulary mapping establishes identical columns. `lowercase=False`, `token_pattern=None` communicate no additional raw-text policy. Callable analyzers bypass preprocessing/tokenizer and ngram_range: this helper compares unigram counts, not generated N-Grams. Default raw-text tokenization drops single-character tokens and lowercases; demonstrate why defaults matter using x and NLP. Tests additionally learn a library vocabulary independently and align its columns before comparison.

## Simple example and expected output

BOW-01 documents: students study language; students study data; language models study patterns.
Sorted vocabulary: data, language, models, patterns, students, study, with six dimensions.
Rows: `[0,1,0,0,1,1]`, `[1,0,0,0,1,1]`, `[0,1,1,1,0,1]`.
Students one-hot: `[0,0,0,0,1,0]`. Manual and library matrices match exactly.

## Live demonstration and teaching steps

1. Open the lab and read three short input documents aloud.
2. Show separate lowercase/punctuation/tokenization policy from Exercise 1.
3. Predict the sorted vocabulary and index before showing them.
4. Select students; connect the active coordinate to the index.
5. Repeat a term in D1, then observe its count and row sum.
6. Show matrix rows/columns, including zeros for absent terms.
7. Select unigram, bigram, trigram; inspect window boundaries.
8. Reveal CountVectorizer only after manual counts; confirm equality.
9. Demonstrate reversed word order, synonyms and repeated common terms.
10. Run the notebook mini challenge and core tests, then commit checkpoint 2.

## Questions to ask students

- Why does vocabulary order matter? What happens for an unknown term?
- Why are one-hot vectors sparse? Can they relate car and automobile?
- What does BoW lose? How do bigrams help?
- What happens to dimensions as N-Gram range increases?
- Why might common words dominate counts?

## Common mistakes and troubleshooting

Avoid sets as ordered return values, separate vocabulary per document, wrong increment index, overwriting rather than incrementing, confusing one-hot with BoW, off-by-one window bounds and comparing unaligned library columns. Raw strings are rejected instead of treated as characters. Both encoding methods reject unknown terms; build the shared vocabulary first. Duplicate coordinates are invalid. Empty inputs remain well-defined; the lab requests 2–5 lines and explains all-punctuation input without fitting an empty library vocabulary.

If sklearn is missing, install requirements in the active environment. This lab requires no NLTK corpora: lowercasing, punctuation handling and WordPunct tokenization are resource-free. Keep preprocessing/resource troubleshooting from Exercise 1 separate. Check that notebook imports resolve from repo or notebooks, and that imports distinguish the two vocabulary APIs.

## Recovery, key takeaway and limitation

Restore the known-working reference checkpoint following the workshop Git strategy once finalized; keep students on the same coordinate mapping. No duplicated solutions are supplied. Checkpoint 2 proves numerical representation, not semantic understanding or relevance weighting. All four TODOs and tests/guides/index must stay synchronized.

> Bag of Words tells us how often words occur, but frequency alone does not tell us how informative a word is. Next we will build TF-IDF from scratch.


# Exercises 3 and 4 — TF-IDF, Cosine Similarity and Classical Search

## Why raw counts are not enough / connection

Exercise 2 made coordinates and counts visible. A repeated common term can dominate counts even when it carries little discrimination across documents. TF normalizes document length; DF measures prevalence; IDF discounts corpus-wide terms. This is a statistical weighting heuristic, not guaranteed truth or importance.

## Important mathematics and intuition

- TF(t,d)=count(t,d)/document token count. Empty documents yield zero coordinates.
- DF(t)=number of documents containing t, not total occurrences. Repetition in one document counts once.
- IDF(t)=ln(N/DF(t)), natural logarithm, unsmoothed, no added constant. Require N>0 and 1≤DF≤N; skip IDF preparation for an empty corpus.
- TF-IDF(t,d)=TF(t,d)×IDF(t). Align corpus vocabulary, DF, IDF and every vector.
- Query: filter OOV terms, use the same vocabulary and corpus IDF, normalize TF by retained known tokens. This deliberately differs from strict Exercise 2 BoW unknown handling.
- Cosine(A,B)=dot(A,B)/(magnitude(A)×magnitude(B)). Return 0.0 for a zero denominator. The optional explain=True display mode exposes the same source calculations without UI duplication. Direction matters more than length: A=[1,1], B=[1,0] gives 1/√2; doubling either nonzero vector leaves cosine unchanged. Scores are not probabilities.

## Hand calculation and expected output

D1=`common nlp nlp`, D2=`common text`, D3=`common text`. Vocabulary `[common,nlp,text]`; DF=[3,1,2]; IDF=[0,1.098612,0.405465]. TF rows=[1/3,2/3,0], [1/2,0,1/2], [1/2,0,1/2]. TF-IDF rows=[0,0.732408,0], [0,0,0.202733], [0,0,0.202733]. Query `nlp robotics` ignores robotics and gives [0,1.098612,0]. Scores=[1,0,0]; D1 first, D2 then D3 under stable ties. All-OOV gives [0,0,0], scores all zero and original order, not meaningful matches.

## What students implement / reference logic

| TODO | Function | Reference core logic | Difficulty |
|---|---|---|---|
| 3.1 | calculate_tf | Divide provided BoW counts by token length; empty gives zeros | ★★ Core |
| 3.2 | calculate_df | Count positive rows for each provided BoW column | ★★ Core |
| 3.3 | calculate_idf | Apply math.log(N/df) to each valid document frequency | ★★ Core |
| 3.4 | calculate_tfidf | Multiply paired TF and corpus-IDF coordinates | ★★ Core |
| 3.5 | build_query_tfidf_vector | Filter known tokens, delegate to calculate_tfidf | ★★★ Challenge |
| 4.1 | cosine_similarity | Dot product, two square-root magnitudes, denominator, guarded division | ★★ Core |
| 4.2 | score_documents | Call manual cosine once for each document row | ★★ Core |
| 4.3 | rank_documents | Stable descending sort of positions, slice top K, copy metadata plus score | ★★ Core |

All are Day 1. Source contains full reference answers in BEGIN/END boundaries, not unfinished TODOs. Starter code handles validation, vocabulary/count reuse, matrix assembly, preprocessing, corpus/query orchestration, comparison settings, metadata, file loading and visualization. Students do not build UI or a retrieval framework. Recovery uses the reference checkpoint strategy when finalized; no duplicate solutions. Keep source, TODO Index, both guides and tests synchronized.

## Search ranking and architectural boundary

Representation math belongs in classical_nlp; retrieval calls it to prepare corpus, vectorize query, score, rank and return simple dictionaries. Streamlit imports these modules; src never imports Streamlit. Corpus metadata includes document_id and text with optional title. Ranking includes zero-score rows transparently, preserves original order on ties and never mutates corpus. A prepared corpus can be reused across query experiments. No benchmark expected answers or question metadata are read by application logic.

## Manual vs sklearn

Classroom: normalized TF × ln(N/DF). Comparison: TfidfVectorizer with raw counts × [ln((1+N)/(1+DF))+1], smooth_idf=True, sublinear_tf=False, norm=None. Fixed vocabulary and analyzer=list preserve tokens/case/columns; callable analyzer bypasses raw-text tokenization and ngram_range. sklearn default norm is L2, disabled here for inspection. smooth_idf=False still gives ln(N/DF)+1, not the classroom convention. Do not assert equal raw matrices. Tests verify both formulas independently against hand calculations.

For ranking use the library's corpus IDF on the query. The shared query function normalizes known-token counts; this is a positive rescaling of raw count weights and does not change cosine. TF normalization and L2 row normalization alone do not change nonzero cosine rankings; different IDF can. Consult [scikit-learn term weighting documentation](https://scikit-learn.org/1.7/modules/feature_extraction.html#tfidf-term-weighting).

## Live demonstration and teaching steps

1. Start with BoW and repeat a common term; question whether count equals information.
2. Work the three-document example on paper: TF → DF → IDF → TF-IDF.
3. Trace one explorer coordinate, then show the labelled matrices.
4. Predict Learning Corpus winner for natural language processing.
5. Inspect query tokens/OOV/shared vocabulary and IDF before scoring.
6. Calculate dot product and magnitudes; compare direction versus length.
7. Reveal every score, stable ranking and top-K with matched terms.
8. Try empty, fully OOV and corpus-wide-only queries; explain all-zero results.
9. Reveal sklearn comparison and explain formulas before looking at rankings.
10. Switch to unchanged course descriptions; show exact and paraphrased query behavior.
11. Inspect recorded full-document baselines below; distinguish retrieval from answering or analyzing.
12. Run tests/notebook, reach Checkpoint 3, and commit the first search engine.

## Questions to ask students

Why does DF count documents? What happens when a word appears everywhere? Why must the query use corpus coordinates and IDF? What happens to unseen terms? Why cosine rather than raw dot product? What if a query is zero? Does high lexical similarity guarantee relevance? Why might paraphrases fail?

## Common mistakes, troubleshooting and limitation

Avoid occurrence-count DF, independent query IDF, new query vocabulary, inconsistent coordinate order, division by zero, ascending ranking and treating similarity as probability. If a zero score seems wrong, inspect overlap and IDF before changing code. A one-document corpus makes every observed term IDF zero under this convention; this is expected and the lab warns about zero queries. Terms absent from the corpus are not assigned arbitrary IDF. Tests use approximate float comparisons.

Course indexing uses descriptions only, not course names or keywords. This makes some title-style queries fail realistically; field selection is an explicit design decision, not a dataset defect. No stemming or stopword filtering is used in this baseline, so morphological mismatch and question/function-word overlap can distract ranking. Full-document length/content mixtures can dilute a relevant section; chunking is still unimplemented. No extra resources/dependencies are required. Python 3.11 remains the official baseline; Task 07 now validates the complete suite on clean Python 3.11.17, alongside Python 3.12.14.

## Task 06 measured lexical baseline

Unchanged Task 03 queries were run through the manual pipeline: lowercase/punctuation/tokenization, retained stopwords and forms, normalized TF, natural-log unsmoothed IDF, cosine and stable ties. Corpus: all 14 academic Markdown documents plus 4 complete question papers, in manifest order. Whole-file text includes headings and disclaimers. No chunking, embeddings, question extraction, metadata answers or title boosting. Expected IDs below are used only after ranking for evaluation, never as retrieval input. Scores rounded to six decimals; reproducible for this corpus/policy snapshot, not semantic-quality guarantees. Paper-analysis queries here retrieve candidate files only; no analyzer or summarizer is implemented.

| Benchmark | Type | Top three document IDs (cosine) | Expected source ranks |
|---|---|---|---|
| RET-01 | direct_fact | HCE-DOC08 (0.142684); HCE-DOC11 (0.094570); HCE-DOC01 (0.093905) | HCE-DOC08: 1 |
| RET-02 | lexical_search | HCE-DOC04 (0.133841); NLP-P02 (0.108407); NLP-P01 (0.102131) | HCE-DOC04: 1 |
| RET-03 | semantic_search | HCE-DOC14 (0.133143); HCE-DOC03 (0.062054); HCE-DOC12 (0.054708) | HCE-DOC04: 5 |
| RET-04 | semantic_search | NLP-P03 (0.053639); HCE-DOC01 (0.048714); HCE-DOC13 (0.042381) | HCE-DOC08: 7; HCE-DOC09: 9 |
| RET-05 | semantic_search | HCE-DOC05 (0.063842); HCE-DOC04 (0.037688); NLP-P01 (0.027972) | HCE-DOC05: 1 |
| RET-06 | semantic_search | HCE-DOC02 (0.076708); HCE-DOC12 (0.042703); HCE-DOC01 (0.026845) | HCE-DOC02: 1 |
| RET-07 | direct_fact | HCE-DOC03 (0.084162); HCE-DOC13 (0.083640); NLP-P01 (0.071320) | HCE-DOC13: 2; HCE-DOC06: 10 |
| RET-08 | comparison | NLP-P01 (0.083203); HCE-DOC12 (0.064125); HCE-DOC10 (0.039268) | HCE-DOC05: 15; HCE-DOC04: 8 |
| RET-09 | cross_document | HCE-DOC04 (0.114865); HCE-DOC05 (0.098986); HCE-DOC06 (0.066766) | HCE-DOC07: 8; HCE-DOC04: 1 |
| RET-10 | direct_fact | HCE-DOC03 (0.023494); HCE-DOC14 (0.022334); HCE-DOC08 (0.019873) | HCE-DOC07: 4 |
| RET-11 | direct_fact | HCE-DOC07 (0.294244); HCE-DOC01 (0.057064); HCE-DOC12 (0.050292) | HCE-DOC07: 1 |
| RET-12 | direct_fact | HCE-DOC09 (0.131794); HCE-DOC01 (0.090486); HCE-DOC11 (0.046950) | HCE-DOC09: 1; HCE-DOC04: 8 |
| RET-13 | summarization | HCE-DOC10 (0.176834); HCE-DOC03 (0.041302); HCE-DOC13 (0.034082) | HCE-DOC10: 1 |
| RET-14 | question_paper_analysis | HCE-DOC07 (0.120849); HCE-DOC05 (0.058889); HCE-DOC04 (0.052116) | NLP-P01: 8; NLP-P02: 9; NLP-P03: 6; NLP-P04: 7 |
| RET-15 | direct_fact | HCE-DOC11 (0.259844); HCE-DOC12 (0.102468); HCE-DOC02 (0.068954) | HCE-DOC11: 1 |
| RET-16 | comparison | HCE-DOC02 (0.058023); HCE-DOC06 (0.039514); HCE-DOC03 (0.032383) | HCE-DOC05: 7; HCE-DOC06: 2 |
| RET-17 | summarization | HCE-DOC05 (0.081544); HCE-DOC04 (0.069077); HCE-DOC08 (0.045627) | HCE-DOC04: 2 |
| RET-18 | question_filtering | NLP-P03 (0.103820); HCE-DOC04 (0.066542); NLP-P02 (0.034516) | NLP-P01: 5; NLP-P02: 3; NLP-P03: 1; NLP-P04: 4 |

### The four semantic-vs-keyword benchmark observations

- RET-03: **Which subject teaches computers to interpret human language?** Expected ranks: {'HCE-DOC04': 5}. OOV: computers.
- RET-04: **How much classroom presence do I need to sit my finals?** Expected ranks: {'HCE-DOC08': 7, 'HCE-DOC09': 9}. OOV: finals, much, my, presence, sit.
- RET-05: **Which class learns to predict outcomes from examples?** Expected ranks: {'HCE-DOC05': 1}. OOV: learns.
- RET-06: **Where do connected devices learn the rules for exchanging messages?** Expected ranks: {'HCE-DOC02': 1}. OOV: exchanging, messages.

RET-03 is a genuine lexical miss: faculty information ranks first while the NLP syllabus is fifth. RET-04 is more severe: a question paper ranks first while Attendance Policy and Examination Regulations are seventh and ninth; finals/presence/sit are OOV. By contrast RET-05 and RET-06 still put expected sources first: a paraphrase does not guarantee lexical failure, nor will embeddings guarantee success. Exact RET-02 retrieves the NLP syllabus first. Do not rewrite documents or add query terms just to force failure.

### Course-description search snapshot

Only descriptions are indexed. Course names/keywords are labels/unused fields.

| Query | Top three titles (cosine) | OOV query terms |
|---|---|---|
| natural language processing | Natural Language Processing (0.416701); Data Engineering (0.125232); Programming Fundamentals (0.000000) | natural |
| machine learning | Deep Learning (0.651105); Programming Fundamentals (0.000000); Object-Oriented Programming (0.000000) | none |
| database systems | Computer Networks (0.198037); Distributed Systems (0.189949); Data Engineering (0.169311) | database |
| computer communication networks | Distributed Systems (0.299948); Programming Fundamentals (0.000000); Object-Oriented Programming (0.000000) | computer, networks |
| Which subject teaches machines to work with human communication? | Distributed Systems (0.313492); Natural Language Processing (0.172652); Information Retrieval (0.085559) | subject, teaches, which, work |

The human-communication paraphrase selects Distributed Systems (0.313492) before NLP (0.172652), driven by shared terms rather than intended meaning. Machine learning selects Deep Learning: the ML description says supervised models rather than repeating its title. Database systems loses database as OOV and favors systems elsewhere. These observations are limitations of description-only lexical matching, not reasons to silently alter the canonical dataset. Preserve this baseline for Task 09 semantic comparison under identical corpus/field/preprocessing choices.

## Transition to word embeddings

Document: Natural Language Processing studies computational methods for understanding human language.

Query: Which course teaches machines to work with human communication?

TF-IDF depends heavily on shared terms. Humans recognize the relationship, but the vector system may not. Our measured paraphrase failure supplies a concrete example.

> Can we represent words by meaning instead of only by vocabulary position and frequency?


# Exercise 5 — Word Embeddings and Word2Vec

## Previous limitation and why we teach this

Task 06 measured genuine lexical misses: the human-communication course query selected Distributed Systems (0.313492) before NLP (0.172652); RET-03 put the NLP syllabus fifth, and classroom-presence RET-04 put attendance/examination documents seventh/ninth. Shared terms drove rankings. Do not claim tiny Word2Vec fixes document retrieval: it changes individual word representations and prepares the sentence-embedding question.

## One-hot versus embedding / distributional hypothesis

One-hot coordinates are independent, sparse and vocabulary-sized. A learned dense vector is configured-size and its coordinates are useful jointly, not named semantic axes. Words occurring in similar contexts tend to acquire related geometry. Natural examples include students studying machine learning or language processing, and models examining patterns or text. Existing academic prose supplies authentic repeated contexts; no keyword-stuffed repetitions or benchmark answer labels are injected.

## Context windows, CBOW and Skip-Gram

A window selects neighbouring tokens; sentence boundaries prevent crossing unrelated spans. Gensim may sample a smaller effective window with shrink_windows=True. CBOW: context → target, natural ___ processing predicts language. Skip-Gram: target → context, language predicts nearby natural/processing. At a high level training creates prediction examples, optimizes word/context weights and retains useful word vectors. Students invoke Gensim; backpropagation internals are not an exercise. Negative sampling approximates the prediction objective; do not present similarities as probabilities.

## Corpus and exact configuration

The derived JSON contains sentence-like spans from the 20 course names/descriptions and prose of all 14 academic documents. Tables/headings and repeated disclaimer/institution boilerplate are removed; bullet markers stripped, spans split at sentence punctuation followed by uppercase text, spans under six whitespace words omitted, exact duplicates removed in source order. Abbreviations or semicolon lists can remain in spans; this pragmatic segmentation is infrastructure, not a linguistic sentence-boundary algorithm. Source paths are recorded. No canonical facts, source documents or benchmark metadata are changed.

Default constructor settings: vector_size=50, window=5, min_count=1, sg=0, epochs=100, seed=42. Infrastructure fixes workers=1, sample=0.001, negative=5, hs=0, alpha=0.025, min_alpha=0.0001, sorted_vocab=1, batch_words=10000, shrink_windows=True, and SHA-256-based initialization hash. Configuration is centralized in src. Single-worker/fixed-seed/stable-hash and unchanged sentence order reduce nondeterminism; version, platform and native numerical libraries can still differ. Never test a particular semantic neighbour as an invariant.

## Student/reference boundaries and starter provisions

| TODO | Function | Reference logic | Difficulty |
|---|---|---|---|
| 5.1 | prepare_tokenized_sentences | Reuse lowercasing, punctuation spaces and tokenizer per sentence; omit empty lists | ★ Guided |
| 5.2 | train_word2vec | Call Word2Vec with sentences and explicit parameters plus retained deterministic infrastructure | ★★ Core |
| 5.3 | get_word_vector | Read model.wv[word] and return a copied numeric list | ★ Guided |
| 5.4 | find_similar_words | Call keyed-vector most_similar with provided capped topn | ★★ Core |
| 5.5 | calculate_word_similarity | Call Task 06 manual cosine on retrieved vectors | ★★ Core |

All are Day 1. Complete answers remain in bounded source cores; later starter generation removes only educational cores, retaining signatures, hints, validation and infrastructure. No duplicate solutions. Starter handles corpus extraction/loading, parameter checks, stable hash, cache, tables, manual-versus-library comparison and PCA. Recovery uses the known-working reference checkpoint when Git strategy is finalized.

## Vector geometry and similarity interpretation

Cosine is unchanged from Task 06: dot product divided by product of magnitudes. Convert Gensim vectors to lists to use the shared manual implementation. It should agree with Gensim similarity within floating-point tolerance. Neighbours can reflect association, shared syntax or broad subject matter, not only synonymy. Rare network/database words have sparse evidence; min_count=1 retains them but does not give strong learned semantics. Self similarity is approximately one; OOV retrieval raises a helpful ValueError.

## Caching and PCA infrastructure

Streamlit cache_resource is in the page, not src, keyed by the full corpus and model settings with six cached model entries. Models train locally once per cached configuration; selecting words does not retrain. They are shared read-only across sessions and not saved to disk. Comparing architectures trains the other model lazily. After students edit the training core, the provided **Clear cached models and retrain** button rebuilds models; students do not implement this cache infrastructure. Evicted configurations train again when selected. PCA uses sklearn's full SVD on 2–20 selected words; display 8–20 if possible. Coordinates are labelled but axes have no inherent semantic interpretation. This is a **2D projection for visualization only**: reducing dimensions and changing the selected subset can move points and distort closeness. Compare original-space cosine as evidence.

## GloVe, FastText and static embeddings

GloVe uses global co-occurrence statistics rather than Word2Vec's local prediction approach. FastText uses character subword information; learn/learning/learner share pieces, helping morphological variation, rare terms and unseen forms. Conceptual only: no pretrained archives or additional training is introduced.

bank approved the loan / students sat on the river bank. Traditional Word2Vec stores one vector for bank, regardless of sense, if it is known. The workshop corpus need not contain bank to illustrate this limitation; the notebook actually retrieves the same known language vector in two source sentences. Its retrieval function takes a word, not context. This motivates contextual representations later, but this task implements none.

## Live demonstration / expected outputs

1. Recall measured lexical failures, then compare one-hot independence with distributional intuition.
2. Inspect supplied source sentences/tokens before training; explain context windows.
3. Predict CBOW/Skip-Gram input-target directions using natural ___ processing.
4. Train default CBOW; show vocabulary, 50 floating-point values, norm and sparse/dense size comparison.
5. Compare language/text manually and through Gensim; expect agreement within 1e-6, not a predetermined similarity.
6. Show nearest words, including noisy associations; do not hide awkward results.
7. Enable CBOW/Skip-Gram comparison using the same word/corpus/settings.
8. Change window 2↔5 and size 20↔50; students record observations without guaranteed conclusions.
9. Render labelled PCA and compare 2D impressions with original cosine.
10. Try an OOV word; explain subwords conceptually without switching models.
11. Show static-vector/polysemy limitation, run tests/notebook and commit the word-representation milestone.

## Questions to ask students

Why cannot one-hot express similarity? Where does Word2Vec's similarity come from? What is a context window? How do CBOW and Skip-Gram differ? Why might related words score poorly here? Did cosine change? Does nearest imply synonym? Why is one vector for bank problematic? How might a whole sentence be represented?

## Common mistakes and troubleshooting

Avoid raw strings instead of sentence-token lists, reversed sg values, OOV/case queries, interpreting dimensions as explicit meanings, guaranteed-synonym assumptions, expecting pretrained-quality semantics, omitted seed/workers settings and comparing models with multiple parameters changed. Empty/punctuation-only raw sentences are omitted; training requires at least two retained words. Too-high min_count fails clearly. Keep models read-only in the shared cache; edits that mutate them invalidate reproducibility.

Use the active environment and install requirements containing Gensim 4.4.0. Older Gensim releases can conflict with newer SciPy/NumPy APIs; do not repair this by adding arbitrary transitive pins. On 64-bit Python 3.11/3.12, prefer the provided Windows wheels; unexpected compiler requests usually mean unsupported Python/platform or an outdated installer. Check package versions and `python -m pip check`, use a clean environment and reinstall requirements. No corpora/model downloads are needed for this lab. PCA failure with fewer than two words is explained by the UI/helper. No matplotlib, Torch or Sentence Transformers is required.

## Measured workshop observations (Task 07)

On Windows with Gensim 4.4.0, NumPy 2.5.3 and SciPy 1.18.1 under Python 3.12.14, the 543 sentence-like spans contain 8,137 processed tokens and 1,528 distinct words. Default CBOW trained in approximately 2.1 seconds on this machine; timings vary by laptop and first import/training. network appears once and database twice, so these vectors have little training evidence. No sentence repeats were injected to force outcomes.

Default language vector: 50 dimensions, norm approximately 10.190372; first five values [-0.723063, 1.170119, -2.016138, 0.886025, 0.678604]. These coordinate values are measurements, not interpretable labels or answer expectations.

| Pair | Default manual cosine |
|---|---:|
| language/text | 0.475315 |
| machine/learning | 0.800399 |
| machine/database | 0.483138 |
| student/course | 0.324548 |
| attendance/examination | 0.556706 |

Manual scores agree with Gensim within 1e-6. Do not generalize this ordering; machine/database closeness is not a curriculum prerequisite or semantic equivalence.

| Experiment | Dimensions | language/text cosine | Top five language neighbours |
|---|---:|---:|---|
| CBOW default | 50 | 0.475315 | human (0.9034); generating (0.8894); modeling (0.8865); brings (0.8862); centered (0.8832) |
| Skip-Gram | 50 | 0.293234 | generating (0.7368); symbolic (0.7271); human (0.7202); natural (0.7144); statistical (0.7132) |
| Window 2 | 50 | 0.476564 | symbolic (0.9021); human (0.8967); natural (0.8949); masked (0.8946); centered (0.8930) |
| Vector 20 | 20 | 0.414119 | generating (0.8865); understanding (0.8864); human (0.8824); natural (0.8704); ideas (0.8668) |
| Vector 100 | 100 | 0.467815 | generating (0.9164); human (0.9163); understanding (0.9013); symbolic (0.8987); processing (0.8959) |

The architecture changed the neighbour set; the window-2 model emphasized symbolic/natural/masked here, and size-20/100 changed neighbours again. A larger dimension or a higher pair cosine does not prove better semantics. CBOW neighbours brings and centered are useful examples of noisy corpus association. These are measured snapshots for instructor discussion, not fragile test assertions.

## Checkpoint numbering and transition

User-defined Exercise 5 **Checkpoint 4 — Words Have Learned Vector Representations** is recorded as a word-representation milestone. At Task 07, the original semantic-search Checkpoint 4 was pending; Task 09 now satisfies it. Keep the two milestone meanings distinct. Exact scores vary with software/platform and the tiny corpus is educational, not a knowledge authority.

> Word2Vec gives us meaningful vectors for individual words, but our retrieval system needs vectors for complete questions, sentences and documents. Next we will represent entire sentences directly.

# Exercise 6 — Sentence Embeddings and Semantic Similarity

## Topic, why we teach it, and connection

Word2Vec gives static vectors for individual words; we need representations for complete questions and passages. Task 06's genuine human-communication query ranked NLP second (0.172652), below Distributed Systems (0.313492). Its lexical coordinates did not capture the intended concept reliably. This exercise changes representation while retaining cosine, and compares only selected text pairs. It does not implement semantic search.

## Intuition and sentence representation

One row represents one complete text. Column count is fixed at 384 regardless of sentence length. The pretrained encoder composes contextual token representations; it is not an arithmetic average of our Word2Vec vectors. Individual dimensions have no simple named meanings. Similar texts tend to have useful vector relationships, but association, negation and reversed argument roles are not reliably resolved by a similarity scalar.

## Pretrained encoder and why pretrained

Use exactly `sentence-transformers/all-MiniLM-L6-v2`: public Apache-2.0 model, no payment/account/token requirement, relatively lightweight and usable on CPU for workshop similarity and later retrieval. Its 384-dimensional output is compact compared with vocabulary vectors. It uses Transformer-based representations trained on a much larger text-pair collection than our 14 documents; our tiny synthetic corpus cannot provide equivalent training evidence. Do not overstate generalization, factual grounding or domain accuracy. Model inputs longer than 256 word pieces are truncated.

Sources: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [encoding API](https://www.sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html).

## Student implements / starter provides

| TODO | Topic | Reference solution logic |
|---|---|---|
| 6.1 | Encode sentences (★★ Core) | Call the provided model's encoder on a list, batch_size=32, NumPy output, progress off, normalize_embeddings=False. |
| 6.2 | Sentence similarity (★★ Core) | Batch the two strings, take rows 0/1 as lists, pass both to the existing manual cosine. |

Model initialization, downloading, CPU selection, validation, caching, UI and TF-IDF baseline are starter code. The reference stays executable. Later starter generation removes only these two small cores. No duplicate solution module. Source/TODO Index/guides/tests must remain synchronized.

## Mathematics, normalization and expected outputs

Reuse cosine = (A·B)/(||A|| ||B||); zero-vector safety comes from Task 06. The API requests no extra normalization. **Important measured model detail:** this pretrained model includes a final Normalize module, so its native outputs already have approximately unit norm. `normalize_embeddings=False` does not undo it. Preserve the model intact and teach explicit cosine rather than silently remove the module or claim outputs are unnormalized pooling vectors. Empty batches have shape (0,384); blank texts fail with useful guidance. A single item stays two-dimensional (1,384).

## TF-IDF vs semantic similarity

The starter pairwise baseline reuses normalized TF and ln(N/DF). Fit on eight preset texts plus distinct custom pair inputs. Fitting only two documents would set every shared term's IDF to zero and make their overlap demonstration degenerate. Clearly distinguish this small reference collection from the frozen 20-course IDF used in the Task 06 failure-case section. No sklearn default-smoothing formula is substituted. Sentence embeddings need no fit on these examples.

The reversed-role pair is deliberate: "The student teaches the instructor" versus "The instructor teaches the student". BoW/TF-IDF ignore order and therefore score 1; MiniLM can also score high. This prevents the misleading claim that embeddings always outperform lexical weighting or verify proposition equivalence. Measured values and setup observations are recorded below after validation.

## Live demonstration and teaching steps

1. Recall the actual Task 06 failure and ask students which of NLP/Distributed Systems ought to be closer.
2. Predict lexical scores for a paraphrase; inspect the fixed-corpus score rather than refitting IDF secretly.
3. Encode one sentence and inspect (1,384), first ten values and norm.
4. Encode a batch; ask what each axis means.
5. Encode two texts, retrieve rows, expose dot product and both magnitudes, then calculate the existing manual cosine.
6. Compare related/unrelated examples and the four preset pairs.
7. Change wording without changing intended meaning; reverse subject/object roles and discuss misleading scores.
8. Compare only the two chosen course descriptions. Save full corpus ranking for the next lab.

## Questions to ask students

- What does one row represent? Why does vector dimension not change with sentence length?
- Did cosine change from Task 06? What changed instead?
- Why can paraphrases be closer? Should paraphrases score exactly 1?
- Does high similarity imply factual correctness, entailment or relevance?
- Can unrelated texts receive surprising scores?
- Why pretrained rather than train an encoder on 14 documents?
- What would happen if we encoded every course description?

## Common mistakes and limitations

Confusing words with sentences; encoding characters by passing a raw string; assuming one sentence returns a flat vector; cleaning text with the classical pipeline unnecessarily; loading repeatedly; treating cosine as probability; interpreting arbitrary dimensions; claiming perfect understanding; treating high similarity as factual relevance; overlooking truncated long inputs; mixing pairwise reference IDF with course-corpus IDF. Language/domain mismatch and paraphrase ambiguity remain possible.

## Troubleshooting and instructor recovery

Preload while online with `python -m src.sentence_resources --download`, then run local-only verification without the flag. Public download needs no HF account/token. A missing or incomplete model raises SentenceModelUnavailable with setup/cache-permission guidance; the page displays that message and other labs remain available. No fake embeddings or TF-IDF fallback. Copy the complete cache directory for offline machines; preserve blobs/snapshots/refs and verify each copy. Do not add automatic downloads on page reruns.

Successful loader objects are cached once per process. Streamlit caches the resource and up to 64 text batches; its reload button clears both plus the loader cache after student edits or resource repair. Inference uses eval mode on CPU, with at most four PyTorch threads to avoid laptop oversubscription. First import/load is slower than warm encoding. Gensim/NLTK lab resources are unchanged. PyTorch brings substantial installation/disk overhead; verify a clean 64-bit Python 3.11 environment beforehand. Corporate proxies, TLS and write permissions may block public downloads. Use a prepared offline copy rather than disable TLS verification. Update a broken environment from requirements; do not add unrelated model libraries.

Recovery uses the working/reference checkpoint strategy, not duplicated solution files. If model setup cannot be repaired during class, use measured instructor observations and keep earlier labs available; do not pretend an unavailable encoder produced results. Integration tests skip when a model is unprepared, so classroom readiness requires an explicit no-skip preflight run.


## Measured Exercise 6 reference observations — Task 08

Measured locally on Windows CPU with Python 3.11.17; Python 3.12.14 reproduced these values within 1e-6. Model revision pinned in the starter loader: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. These are observations, not fixed quality thresholds or universal semantic guarantees.

| Preset pair | Classroom TF-IDF cosine | MiniLM cosine |
|---|---:|---:|
| Similar wording, similar meaning | 0.527645 | 0.966884 |
| Different wording, similar meaning | 0.146845 | 0.372418 |
| Shared words, reversed meaning | 1.000000 | 0.993103 |
| Unrelated subjects | 0.000000 | 0.117644 |

The core A/B paraphrase scored **0.601510**; A/C database text scored **0.290848**. The reversed-role example's 0.993103 embedding score is especially useful: contextual sentence representations can still conflate who teaches whom. Numerical scale increases are not, by themselves, evidence of better accuracy.

| Task 06 manually selected description | Original corpus TF-IDF cosine | MiniLM cosine |
|---|---:|---:|
| Natural Language Processing | 0.172652 | 0.392357 |
| Distributed Systems | 0.313492 | 0.382454 |

NLP now has the larger measured pairwise MiniLM similarity, but the gap is only about 0.0099. This is weak comparative evidence, not proof that full semantic search will always retrieve it first. Do not infer a full corpus rank from two selected descriptions. The dataset was not modified to force improvement.

One sentence returns (1,384). First five values for the core NLP sentence are 0.027487, 0.011657, 0.068197, -0.071739, 0.020071. Norm is approximately 1 because the exact model includes Normalize; no additional encode normalization is requested.

Download/setup took 43.179 seconds, including about 17 seconds fetching 11 selected model files; one-sentence first inference in that preflight took 2.044 seconds. Model snapshot content is **91,607,178 bytes (91.6 MB / 87.4 MiB)**. Local Windows cache file total is **183,214,396 bytes (183.2 MB / 174.7 MiB)** because blobs and snapshot files are duplicated without symlinks. Installed Python packages require additional disk space. No ONNX/OpenVINO/pretrained-word-vector archives were downloaded.

Fresh-process cached loading, including imports, took 22.637 seconds on Python 3.11 and 25.482 seconds on Python 3.12. Same-process loader reuse took less than 1 ms. A three-sentence batch took 0.235 seconds initially, then approximately 0.025–0.031 seconds when warm. Separate first-inference timing varies with imports, initialization and concurrent CPU activity; these values are planning observations, not performance guarantees.

Tested package resolutions: sentence-transformers 5.1.2, Transformers 4.57.6, PyTorch 2.14.1, huggingface-hub 0.36.2; NumPy 2.4.6 (Python 3.11) / 2.5.3 (Python 3.12). Only sentence-transformers is a new direct requirement. Other packages resolve transitively; no extra compatibility pins were necessary. Model loading is local-only during normal inference; inference never calls a paid or remote API.

## Key takeaway and transition

Representation changed; similarity mathematics did not. This exercise's semantic-sentence Checkpoint 5 is distinct from the earlier roadmap's Transformer Checkpoint 5 (pending at Task 08; completed in Task 10).

> We can now represent a question and a course description in the same semantic vector space. Instead of comparing only two sentences, what if we compare one query against every document and rank them?

That introduces Semantic Search in Task 09, which is not implemented here.

# Exercise 7 — Semantic Search

## Topic, why we teach it, and previous step

Generalize Exercise 6 pairwise sentence similarity into one query versus every document. Students already know the comparison/ranking mathematics. Dense representations make another retrieval strategy possible, without an LLM, vector database or unrelated framework.

## Intuition and shared coordinate space

N descriptions → N×384 matrix; one query → 384-dimensional vector; manual cosine with each row → N scores in original order. Descending sort → indices → Top-K copied metadata. Query and documents must use the exact shared MiniLM model/revision. Preserve its native Normalize module and no-extra-normalization encoding convention. Scores are not confidence, probability, answer correctness or proof that a passage contains all requested facts.

## Student implementations and reference logic

| TODO | Topic | Reference logic |
|---|---|---|
| 7.1 (★★ Core) | Corpus encoding | Collect text fields in original order, call shared encode_sentences. |
| 7.2 (★ Guided) | Query encoding | Wrap query as a one-item batch, take row 0 as a list. |
| 7.3 (★★ Core) | Semantic scores | Reuse score_documents/manual cosine on every matrix row. |
| 7.4 (★★ Core) | Top-K | Reuse rank_documents; descending and original-order ties. |

Signatures/shape checks/model loading/caching/UI/evaluation are starter code. Empty corpus is (0,384); blank query/document texts fail clearly. K=0 is valid, oversized K caps naturally, invalid K raises ValueError. No source mutation. Students may reuse the previous exercise's ranking rather than rewrite it. Keep exactly these four reference boundaries synchronized with the guides, TODO Index and tests.

## Fair comparison and searchable content

Use all twenty rows of the existing CSV. Both methods receive **description only**, in identical order; titles/codes are labels. No keyword boosting, title enrichment or dataset changes. TF-IDF preprocessing remains lowercase/punctuation spaces/WordPunct, with retained forms and stopwords, normalized TF × ln(N/DF). MiniLM receives original complete description strings. Compare relevance and rank, not raw cosine magnitudes as if calibrated identically.

## Caching and intermediate inspection

The page caches the shared encoder and corpus embeddings separately. Corpus-cache keys include document metadata/text/order and exact model ID/revision; source-content changes rebuild it. Query/Top-K/selected-course changes do not re-encode the corpus. Queries are encoded when searching; the corpus remains reused. The cache-clear button resets page resources, corpus/lexical matrices and demonstration caches after source edits. Source modules never import Streamlit.

Inspect N×384 matrix dimensions, query preview, one score per course, original row indices, sorted indices and Top-K. TF-IDF can show shared lexical terms. For embeddings only claim vector proximity, not token attribution. Default Top-K is 5, and both methods share its control.

## Benchmark scope and metrics

The app's twenty descriptions cannot supply policy/calendar/paper answers. Separately, `python -m workshop.evaluate_retrieval` runs every unchanged Task 03 query over the original 14 full Markdown documents plus 4 papers in manifest order, exactly matching the Task 06 lexical baseline. Whole-file text includes headings and disclaimers. MiniLM receives those same raw files but natively truncates at 256 word pieces. All eighteen inputs exceed that limit (548–2571 word pieces including special tokens); the curriculum's encoded prefix does not include Computer Networks. This is a disclosed exploratory whole-document baseline, **not** a claim of equal token exposure or a RAG-ready evaluation. No extraction, chunking, aggregation or alternate encoder is introduced. Token-length diagnostics and source SHA-256 hashes are in the measured report.

`workshop/retrieval_comparison_results.json` contains measured ranks/scores/performance/model revision/environment. It is not a canonical fact contract. Regenerate using the evaluator; do not hand-edit measured claims. Actual application search uses only query/content/representations and never reads expected answers or this report. Expected IDs are used only after retrieval by evaluation tooling. No question metadata or topic targets enter retrieval.

Single-source aggregate eligibility: RET-01,02,03,05,06,10,11,15 (8 cases), each with one listed source and a direct/lexical/semantic retrieval request. Top-1 accuracy is correct first sources divided by 8. Top-3 hit rate is queries with that source in first 3 divided by 8. MRR is mean reciprocal first-relevant rank. Multi-source RET-04,07,09,12, comparison RET-08,16, summary RET-13,17 and paper-analysis/filter RET-14,18 are excluded from these aggregates. They still report every expected rank, any-hit@3 and source coverage@3. Multiple expected IDs here are required evidence, not interchangeable alternatives. Paper analysis requires four papers, so Top-3 cannot retrieve all. Source metrics never measure summary quality, comparisons, filtering, counts or generated answers.

## Live demonstration

1. Start exact natural language processing; inspect both representations and rankings.
2. Try machine learning/database systems/network wording; compare ranks without title boosts.
3. Repeat the exact human-communication paraphrase from Tasks 06/08; do not assume NLP must win until ranking all twenty.
4. Track NLP's rank, inspect all scores and note its margin against Distributed Systems.
5. Show the four unchanged semantic-vs-lexical document benchmarks using the measured evaluation table.
6. Show a regression, not only improvements; explain truncated long documents.
7. Try learning (ambiguous) and baking (unrelated); neither nearest-vector output nor all-zero lexical ties establish relevance.
8. Connect retrieval to evidence selection, leaving chunking/generation for later tasks.

## Questions to ask students

What changed/stayed the same? Why the same encoder? How can a result share no exact query terms? Is cosine confidence? Can related courses lack the answer? Why not a vector database for twenty documents? What happens with hundreds of chunks? How might retrieved evidence later help a language model? Can one retrieved source answer a question requiring two documents?

## Common mistakes, limitations and troubleshooting

Different query/document encoders; stale/recomputed corpus vectors; wrong row order; ranking ascending; enriching only semantic input; comparing score scales directly; calling similarity confidence; assuming embeddings always win; losing deterministic ties; comparing one pair rather than all rows; reading expected IDs to manufacture answers. Original text versus classical preprocessing is intentional. Native truncation discards later sections in long files. Related topics, ambiguous queries and irrelevant nearest neighbours remain possible. No threshold or abstention logic is taught here.

Reuse Task 08 explicit preload/offline verification. Missing model gives setup help, with other labs available. Never download automatically on reruns or substitute fake vectors. Rebuild caches after student edits or CSV changes. Verify tests with no model-related skips and execute the shared notebook in the workshop kernel. Recovery uses the known-working reference checkpoint strategy, not duplicated solutions.

## Measured Exercise 7 results — Task 09

Model: sentence-transformers/all-MiniLM-L6-v2, pinned revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Python 3.11.17 CPU observations. All twenty course descriptions indexed without titles/keywords. Full ordered twenty-result lists are in the measured JSON artifact.

| Query | TF-IDF Top-1 | Semantic Top-1 |
|---|---|---|
| natural language processing | Natural Language Processing (0.416701) | Natural Language Processing (0.505026) |
| machine learning | Deep Learning (0.651105) | Machine Learning (0.444176) |
| database systems | Computer Networks (0.198037) | Database Management Systems (0.560724) |
| computer communication networks | Distributed Systems (0.299948) | Computer Networks (0.449251) |
| Which subject teaches machines to work with human communication? | Distributed Systems (0.313492) | Natural Language Processing (0.392357) |
| learning | Deep Learning (0.613868) | Machine Learning (0.364836) |
| How do I bake a chocolate cake? | Programming Fundamentals (0.000000) | Deep Learning (0.099215) |

Exact human-communication query: **Which subject teaches machines to work with human communication?**

| Rank | TF-IDF course | TF-IDF cosine | Semantic course | Semantic cosine |
|---|---|---:|---|---:|
| 1 | Distributed Systems | 0.313492 | Natural Language Processing | 0.392357 |
| 2 | Natural Language Processing | 0.172652 | Distributed Systems | 0.382454 |
| 3 | Information Retrieval | 0.085559 | Computer Networks | 0.301101 |
| 4 | Programming Fundamentals | 0.081452 | Artificial Intelligence | 0.294080 |
| 5 | Machine Learning | 0.012466 | Object-Oriented Programming | 0.234594 |

NLP moves from lexical rank 2 (0.172652) to semantic rank 1 (0.392357). Distributed Systems moves from lexical rank 1 (0.313492) to semantic rank 2 (0.382454). The semantic gap is only about 0.0099; avoid overstating robustness. The full ranking now confirms what Task 08's two-course comparison could not establish.

### All eighteen original source benchmarks

| ID | Evaluation scope | TF-IDF expected ranks | Semantic expected ranks |
|---|---|---|---|
| RET-01 | single_source_retrieval | HCE-DOC08: 1 | HCE-DOC08: 1 |
| RET-02 | single_source_retrieval | HCE-DOC04: 1 | HCE-DOC04: 6 |
| RET-03 | single_source_retrieval | HCE-DOC04: 5 | HCE-DOC04: 1 |
| RET-04 | multi_source_retrieval | HCE-DOC08: 7; HCE-DOC09: 9 | HCE-DOC08: 1; HCE-DOC09: 2 |
| RET-05 | single_source_retrieval | HCE-DOC05: 1 | HCE-DOC05: 1 |
| RET-06 | single_source_retrieval | HCE-DOC02: 1 | HCE-DOC02: 15 |
| RET-07 | multi_source_retrieval | HCE-DOC13: 2; HCE-DOC06: 10 | HCE-DOC13: 11; HCE-DOC06: 1 |
| RET-08 | comparison_evidence_only | HCE-DOC05: 15; HCE-DOC04: 8 | HCE-DOC05: 4; HCE-DOC04: 1 |
| RET-09 | multi_source_retrieval | HCE-DOC07: 8; HCE-DOC04: 1 | HCE-DOC07: 6; HCE-DOC04: 7 |
| RET-10 | single_source_retrieval | HCE-DOC07: 4 | HCE-DOC07: 1 |
| RET-11 | single_source_retrieval | HCE-DOC07: 1 | HCE-DOC07: 4 |
| RET-12 | multi_source_retrieval | HCE-DOC09: 1; HCE-DOC04: 8 | HCE-DOC09: 5; HCE-DOC04: 6 |
| RET-13 | summary_evidence_only | HCE-DOC10: 1 | HCE-DOC10: 1 |
| RET-14 | analysis_papers_evidence_only | NLP-P01: 8; NLP-P02: 9; NLP-P03: 6; NLP-P04: 7 | NLP-P01: 4; NLP-P02: 3; NLP-P03: 2; NLP-P04: 1 |
| RET-15 | single_source_retrieval | HCE-DOC11: 1 | HCE-DOC11: 1 |
| RET-16 | comparison_evidence_only | HCE-DOC05: 7; HCE-DOC06: 2 | HCE-DOC05: 4; HCE-DOC06: 1 |
| RET-17 | summary_evidence_only | HCE-DOC04: 2 | HCE-DOC04: 8 |
| RET-18 | analysis_papers_evidence_only | NLP-P01: 5; NLP-P02: 3; NLP-P03: 1; NLP-P04: 4 | NLP-P01: 1; NLP-P02: 3; NLP-P03: 2; NLP-P04: 4 |

Four deliberate semantic-vs-lexical cases: RET-03 NLP syllabus **5→1**; RET-04 attendance **7→1**, examination rules **9→2**; RET-05 ML syllabus **1→1**; RET-06 curriculum **1→15**. The attendance/eligibility case is a strong evidence improvement: both needed sources now appear in Top-3. The largest individual rank increase is RET-08 ML syllabus 15→4 (comparison evidence only; still outside Top-3). A key regression is RET-06: the broad curriculum file ranks poorly, and its relevant later course section is outside the encoder's short input window. Truncation is a plausible contributor, not a proven sole cause; we have not performed a controlled comparison with equal token exposure. Other regressions include RET-02 syllabus 1→6, RET-11 calendar 1→4, RET-17 NLP summary source 2→8. These are not corrected by modifying data or benchmarks.

| Metric (8 single-source cases only) | TF-IDF | Semantic |
|---|---:|---:|
| Top-1 accuracy | 6/8 = 75% | 5/8 = 62.5% |
| Top-3 hit rate | 6/8 = 75% | 5/8 = 62.5% |
| MRR | 0.806250 | 0.685417 |

This small, heterogeneous, unchunked/truncated document test does **not** demonstrate semantic retrieval universally outperforms lexical retrieval. Course descriptions and full-document source benchmarks are separate corpora; do not pool their scores or retrofit course IDs into policy benchmarks. Every query was evaluated, but ten multi-source/analysis/summary/comparison cases were excluded from single-source accuracy. Their evidence coverage is recorded, not silently treated as success or answer correctness.

Unrelated baking query: all lexical scores are zero (original-order Top-1 is Programming Fundamentals); semantic Top-1 is Deep Learning at 0.099215 despite no baking course. Learning is ambiguous: semantic ML 0.364836, DL 0.359559, Computer Vision 0.351146. Nearest results exist even without an answer. We do not introduce calibrated cutoffs or false confidence labels.

### Performance and reproducibility

Measured course matrix shape: (20,384). Fresh local-model loading: 15.521 seconds. Corpus encoding: 0.484 seconds. Query encoding across seven examples: 14.4–18.6 ms. Twenty manual cosine comparisons: 33.1–41.6 ms. Ranking only: 0.092–0.119 ms. Corpus reuse is verified by UI instrumentation: query/Top-K/track-course interactions do not call corpus encoding again; cache-clear does. Timings vary by CPU/load/environment; do not promise these numbers for every laptop.

No runtime dependency, source academic document, canonical fact, benchmark query or expected source was changed. The measured report is newly generated instructor output with source hashes, not ground truth used by search. No chunking, RAG, answer construction or generation exists.

## Key takeaway and transition

Representation changed; cosine and ranking remained. Exercise 7's semantic-search Checkpoint 6 is distinct from the original roadmap's document-chunk Checkpoint 6; original semantic-search Checkpoint 4 is now available.

> We now have two retrieval systems. TF-IDF retrieves by lexical evidence, while sentence embeddings retrieve by semantic proximity. On Day 2 we will move from short course descriptions to full academic documents. Full documents are too large to retrieve as one unit, so we will need to extract, clean and divide them into meaningful chunks.
# Exercise 8 — Understanding Attention (READY)

## Topic, why and connection to Day 1

Representation determines retrieval behavior. Task 09's human-communication query moved NLP from lexical rank 2 to semantic rank 1, narrowly ahead of Distributed Systems. Full curriculum RET-06 fell from rank 1 to 15: embeddings do not guarantee relevance, and full documents exceed MiniLM's 256-wordpiece window. Explain contextual machinery before processing longer documents.

Word2Vec gives `bank` one static vector in `bank approved the loan` and `students sat on the river bank`. Contextual representations can change with surrounding tokens. Attention selectively combines information; it is not a standalone solution to meaning.

## Intuition and important mathematics

**Simplified educational attention example**, not actual BERT/GPT weights. Q expresses an information need; K advertises matching features; V carries information to combine. These vector roles are not database keys or dictionary values.

Required unscaled convention: `score_i = Q · K_i`; `w_i = exp(score_i) / sum_j exp(score_j)`; `context = sum_i w_i × V_i`.

Use Q `[1,0]`, K `[[1,0],[0,1],[0.5,0.5]]`, V `[[2,0],[0,2],[1,1]]`. Scores `[1,0,0.5]`; exponential sum `5.367003`; weights `[0.5064803911,0.1863237232,0.3071958857]`; weighted values `[[1.0129607821,0],[0,0.3726474465],[0.3071958857,0.3071958857]]`; context `[1.3201566679,0.6798433321]`.

Subtract the maximum before exponentiation: `exp(s_i-max(s)) / sum exp(s_j-max(s))` is identical algebraically and prevents overflow. Numerical underflow may round tiny weights to zero for extreme gaps; do not promise strict computed positivity for arbitrary inputs. A zero query gives equal scores and uniform weights.

## What students implement / reference logic

Exactly three small cores in `src/transformers_nlp.py`; retain completed reference implementations. Later starter conversion removes only `BEGIN/END STUDENT CORE` lines, without duplicated solutions.

| TODO | Function | Reference logic |
|---|---|---|
| 8.1 | `calculate_attention_scores` | Dot Q with every key row; collect scores |
| 8.2 | `softmax_attention_weights` | Subtract max; exponentiate; divide by sum |
| 8.3 | `calculate_weighted_context` | Zero context; add each weight times its value row |

All are ★★ Core. Starter code provides finite/shape checks, unit-sum validation, composition, scaling comparison, editable UI, plotting, pretrained resources and inference. Empty keys/scores return empty vectors; context rejects an empty collection because dimension is undefined. Value dimensions may differ from Q/K. Source never imports Streamlit.

## Self-attention, scaling and multi-head attention

After the hand calculation, introduce learned W_Q/W_K/W_V projections derived from the same sequence in self-attention. Each matrix row is the token seeking information; columns identify tokens attended to. Contrast one query with all tokens producing queries.

Real formula: `Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k))V`. Larger dimensions can increase dot products and concentrate softmax; scaling helps control this. The checkbox is instructor/starter code, not TODO 8.4.

Multi-head attention uses different learned projections, combines outputs and can learn multiple relationships. Do not assign guaranteed grammar/coreference labels to heads. Attention weights are not full explanations of reasoning.

## Teaching steps and live demonstration

1. Recap retrieval representations and the static bank example.
2. Predict which K matches Q; calculate a dot product on paper.
3. Show scores, softmax weights and weighted value rows separately.
4. Change Q, then V; explain which intermediate values change.
5. Add a constant to every score in the notebook; observe unchanged weights.
6. Compare scaled/unscaled scores, then introduce the self-attention matrix and projections.
7. Use `The student submitted the assignment because it was due` as a language analogy. Do not claim toy weights measure coreference.
8. Transition to pretrained task models.

Expected output is the numeric example above, weights summing to one and a two-dimensional context. Changing values demonstrates selecting information as well as positions.

## Questions to ask students

- Why does contextual information matter if we already have embeddings?
- What does Q represent? How do keys differ from values?
- Why normalize the entire score vector? Why do weights sum to one here?
- What happens when Q is zero? What does context represent?
- What changes in self-attention? Why divide by sqrt(d_k)?
- Does every head have a named human interpretation?

## Common mistakes and troubleshooting

Confusing Q/K/V with literal text; using cosine instead of required dot product; softmax per scalar; missing denominator; unshifted exponentiation overflow; mismatched key/value row order; incorrect weighted sum; treating attention as verified explanation. For shape errors inspect rows/dimensions before changing the core. Restore cleared editor cells to finite numbers. Recover the three small cores using the tested reference checkpoint; exact Git checkpoint distribution is a later task, without solution modules.

## Key takeaway, limitation and transition

Attention selects and combines representations through vector compatibility. Toy vectors provide intuition, not evidence about a trained model. Transformer layers learn richer contextual operations; next inspect task applications.

# Exercise 9 — Pretrained Transformer NLP (READY)

## Topic, why and architecture overview

Instructor-guided experimentation — **no algorithm TODO**. Students inspect real outputs; starter code handles loading, tokenization, inference and caching. Do not add artificial TODO 9.x markers.

Tokenizer → token embeddings + positional information → stacked blocks of self-attention, feed-forward transformations, residual connections and normalization → task-specific output. Attention mixes sequence information; feed-forward layers transform positions; residual/normalization components support deeper stacks. No training or full block implementation is required.

## BERT, GPT and Sentence Transformers

BERT = Bidirectional Encoder Representations from Transformers. Encoder-oriented bidirectional context and masked language modeling produce contextual representations, commonly fine-tuned for classification, NER and extractive QA. Show `The student submitted the [MASK] before the deadline` conceptually, without an MLM checkpoint.

GPT = Generative Pre-trained Transformer. Typically decoder-oriented, causal/autoregressive next-token modeling generates a continuation token by token. Show `Natural language processing helps computers ...` conceptually. No GPT or generative LLM is downloaded or executed.

| Feature | BERT | GPT |
|---|---|---|
| Typical architecture | Encoder | Decoder |
| Context | Bidirectional | Causal/autoregressive |
| Pretraining intuition | Masked tokens | Next token |
| Common strength | Understanding/representation | Generation |
| Typical uses | Classification, NER, extractive QA | Generation, assistants |

Modern architectures vary; avoid absolutes. Live models are fine-tuned DistilBERT variants, not vanilla BERT or GPT. Distillation produces compact encoders. Yesterday's all-MiniLM-L6-v2 already used a Transformer encoder: sentence-similarity training and pooling produce sentence vectors rather than token task labels.

## Pretrained pipelines and model selection

Three public Apache-2.0 checkpoints; CPU baseline, standard Transformers classes, safetensors, no remote code/account/payment. Exact IDs/revisions live in `src/transformer_resources.py`.

- Sentiment: [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), SST-2 fine-tuning. Positive/negative output, with domain and ambiguity limitations.
- NER: [dslim/distilbert-NER](https://huggingface.co/dslim/distilbert-NER), CoNLL-2003 PER/ORG/LOC/MISC categories. Fast-tokenizer `first` aggregation selects a word's first-subtoken label and merges entity groups. Display exact original text slices. Technical terms may be missed or classified as MISC.
- Extractive QA: [distilbert/distilbert-base-uncased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-uncased-distilled-squad), SQuAD v1.1. Selects a span from supplied context; not calibrated to reliably abstain on unsupported questions. This is not RAG or generation.

Only the already installed Transformers/PyTorch stack is used. Measured sizes, CPU timings, outputs and process-cache reuse are in `workshop/transformer_preflight_results.json`. About 796 MB additional download/storage, below 1 GB. Fetch one safetensors checkpoint directly into each revision directory to avoid Windows duplicate blob/snapshot copies. Timings vary: sentiment's first load includes shared-library initialization, so these are not fair model speed benchmarks.

## Expected outputs

Positive feedback: POSITIVE ~0.999687; negative feedback: NEGATIVE ~0.999514. Scores are model task estimates, not universal truth.

For `Riya studies Natural Language Processing at Hindu College of Engineering.`, measured entities include Riya/PER (~0.987183, chars 0–4), Natural Language Processing/MISC (~0.503492, chars 13–40), Hindu College of Engineering/ORG (~0.985173, chars 44–72). Recognition is pattern classification, not proof the model knows fictional university facts.

The context begins `Natural Language Processing is offered in Semester 6.` The semester question returns `Semester 6`, score ~0.700936, chars 42–52. Verify `context[start:end]`. Unsupported questions may still return a span; record behavior rather than promising abstention.

## Live demonstration and experiments

1. Complete manual attention before task models.
2. Contrast BERT masked context with GPT next-token continuation; connect MiniLM.
3. Run positive/negative sentences, then mixed feedback or sarcasm.
4. Run fictional-institution NER; inspect categories, misses and spans; try location/course terms.
5. Run supported QA; change supplied context locally and rerun.
6. Ask an unsupported question; inspect its answer slice and score critically.
7. Ask where context came from: the user supplied it; retrieval and QA are not integrated.

## Questions to ask students

- How do BERT and GPT differ? Which predicts successive next tokens?
- Was MiniLM Transformer-based? Why is a sentence vector different from NER labels?
- Does an entity label establish a real institution? Is a task score factual confidence?
- Can a pretrained model know private synthetic facts without receiving context?
- What if context is too long or the answer absent?

## Common mistakes

Calling BERT generative in the same sense as GPT; treating scores as truth; confusing supplied-context QA with RAG; assuming pretrained knowledge includes synthetic facts; using NER categories as course taxonomy; expecting exact cross-platform scores; assuming unsupported QA abstains; repeatedly loading pipelines.

## Troubleshooting and instructor recovery

Preload with `python -m src.transformer_resources --download`, verify without the flag. Downloads are explicit/public; failures yield setup guidance, not fake fallback. Local paths and `local_files_only=True` prevent ordinary Hub calls. Copy the complete `.cache/transformer_tasks` tree for offline machines. Check disk, permissions, requirements and pinned revision directories. Missing models do not block Exercise 8. Successful pipelines are process-cached and Streamlit resource-cached, loaded only when a task is run. First import can take tens of seconds; wait for the spinner.

Use short input: at most 400 tokens (combined QA question/context), QA question at most 64. Reject excessive input rather than silently discarding content. No document chunking/overlap is introduced; short supplied QA context fits one sequence. Known-working reference cores and notebook support recovery; exact checkpoint strategy remains future work. Limited integration tests skip only when explicitly preloaded resources are absent; release validation prepares every model and runs both Python versions with no skips.

## Checkpoint and transition

Exercise-specific Checkpoint 7 is Attention and Transformer NLP; original roadmap Checkpoint 7 (RAG) remains pending. Original Transformer-experiment Checkpoint 5 is now available. Retrieval + context + model remains a future connection.

> We now understand how Transformer models process contextual language and how pretrained models perform NLP tasks. But our academic knowledge base contains full documents, and Task 09 already showed that embedding entire long documents causes truncation. Before retrieval can support an assistant, we need to extract documents and divide them into smaller meaningful units.


# Exercise 10 — Document Extraction, Cleaning and Chunking

## Topic, connection and why we teach it

Semantic search used compact course descriptions. Task 09 also measured whole Markdown documents, and every long benchmark input exceeded MiniLM's 256-WordPiece window. The expected curriculum source in RET06 ranked 1st lexically and 15th semantically; relevant later curriculum content lies beyond the prefix. Truncation plausibly contributes, but this is not a controlled causal experiment. Chunking is our next representation preparation step, not proof of improved retrieval yet.

Students implement four small cores (10.1–10.4). Starter code supplies Markdown/PDF readers, upload bytes, resource handling, offset helpers, validation, statistics and UI. No embedding, retrieval, context construction or generation belongs in this exercise.

## Intuition and mathematics

A long source must be divided into smaller evidence units. Word windows advance by **size - overlap**. Overlap adds repeated positions so a boundary has some shared context. It cannot guarantee complete sentences or preserve every idea. Words are whitespace-delimited spans, not WordPieces; Markdown table markers count as spans. Model-token counts include special tokens and are calculated without truncation or encoder inference.

## Reference logic and TODO IDs

- **10.1 (★ Guided), clean_document_text:** normalize CRLF/CR to LF; collapse non-newline whitespace per line; trim each line; reduce three-or-more newlines to two; trim document ends. Preserve case, punctuation, numbers and headings. Do not guess/remove headers, dehyphenate, lowercase, stem or delete stopwords.
- **10.2 (★★ Core), split_text_into_chunks:** prepare regex word spans; iterate starts with step=size; reuse `_word_window` to create exact substrings and end-exclusive offsets. Keep the last partial window and return [] for empty input.
- **10.3 (★★ Core), create_overlapping_chunks:** validate size/overlap in starter code; zero overlap reuses 10.2. Repeatedly append a window, stop after its end reaches the word count, otherwise advance by size-overlap. The stop prevents redundant overlap-only tails.
- **10.4 (★★ Core), attach_chunk_metadata:** enumerate windows from a document-wide start index; copy each dict and add `chunk_id`, document identity/title, index, source path/type, real page number. IDs use `document_id::chunk_NNN`, starting at 000. Do not invent Markdown pages or mutate windows.

Simple example: eight words `a b c d e f g h`, size 4, overlap 1 → ranges [0,4), [3,7), [6,8). Without overlap: [0,4), [4,8). A Markdown example chunk has `{chunk_id: HCE-DOC04::chunk_000, chunk_index: 0, source_type: markdown, page_number: null, word_start: 0, word_end: 100}` plus title/path/exact character offsets/text. PDF chunk indexes continue across pages but page-local offsets reset. IDs are positional, not immutable content hashes.

## Extraction, corpus and provenance

The normalized document dict has document_id/title/source_path/source_type and ordered pages of `{page_number,text}`. Markdown has one logical unit with page_number=None; PDF uses physical 1-based page numbers and preserves empty pages in mixed documents. Fully image-only/empty PDFs receive clear text-based-PDF/OCR guidance. PDF chunks are processed independently per page; no multi-page provenance is invented. This can divide an idea across pages, which is an explicit tradeoff.

Three PDFs derive from existing NLP syllabus, attendance policy and examination regulations. Markdown remains canonical; `derived_sample_pdfs` is a separate manifest inventory with canonical IDs/filenames/hashes. Do not index both copies as independent benchmark sources. Every PDF retains the exact synthetic disclaimer and fictional institution description. The historical Task 09 provenance test compares the unchanged original manifest projection; all original source hashes remain enforced. Historical scores/benchmarks are not rewritten.

`pypdf==6.10.0` is the sole new runtime dependency. Instructor PDF regeneration uses optional ReportLab: `python -m workshop.generate_sample_pdfs` in an authoring environment; students do not install it. Text layout/order/table extraction can differ from Markdown. No OCR, automatic header deletion or aggressive classical preprocessing is performed. Upload bytes stay in memory; 10 MiB / 50-page guardrails are classroom limits, not a security sandbox.

## Measured defaults and size/overlap experiments

Run `python -m workshop.analyze_document_chunks` with the existing cached MiniLM tokenizer. Results and source hashes are in `workshop/chunk_analysis_results.json`. This does not load encoder weights or create embeddings. The pinned model/tokenizer identity is unchanged.

Full 14 Markdown documents: 723 / 1188.43 / 2571 model tokens (min/average/max); all 14 exceed 256. Default **100 words, overlap 20**: 152 chunks, 21 / 95.53 / 100 words and 25 / 136.61 / 247 model tokens; **zero** chunks exceed 256. The initial 120/20 proposal produced six oversized chunks, so the measured default was reduced. Zero outliers on supplied documents does not guarantee fit for arbitrary uploads; inspect counts whenever content changes.

| Size / overlap | Chunks | Max model tokens | Above 256 | Duplicated word positions |
|---|---:|---:|---:|---:|
| 60 / 20 | 293 | 162 | 0 | 5580 |
| 100 / 0 | 126 | 247 | 0 | 0 |
| 100 / 20 | 152 | 247 | 0 | 2760 |
| 100 / 40 | 193 | 245 | 0 | 7160 |
| 120 / 20 | 121 | 288 | 6 | 2140 |
| 240 / 20 | 59 | 566 | 50 | 900 |

Smaller windows reduce token risk but lose context and create more units; larger windows retain more context but can truncate later. More overlap increases repeated evidence/cost without adding new source facts. These are preparation measurements, not retrieval-quality results. Future retrieval must handle redundancy and evaluate quality independently.

## Teaching steps and live demonstration

1. Open a canonical NLP Markdown source and inspect its full token length.
2. Open the derived PDF; show page text and compare retained facts, headings and table layout.
3. Clean whitespace; explicitly point out unchanged 75%, course code, marks and case.
4. Predict eight-word windows on paper, then show size/overlap changes.
5. Inspect the final partial chunk and exact source offsets.
6. Open neighboring chunks and highlight shared boundary words.
7. Compare 100/0, 100/20 and 100/40; distinguish duplicate positions from repeated terms.
8. Compare 60/20 and 240/20 token outliers; connect back to Task 09 truncation.
9. Show truthful PDF page metadata and null Markdown page values.
10. Explain that embeddings/retrieval/context/answers remain visibly planned.

## Expected output and questions

Notebook imports the shared module; RAG page displays actual extracted/cleaned text, statistics, chunk metadata and optional cached-tokenizer counts. Ask: Why preserve punctuation/numbers? Why not remove stopwords here? Are 100 words always fewer than 256 WordPieces? What does overlap cost? Can a window cut a sentence? Why is Markdown page 1 misleading? What happens to chunk IDs after changing settings? Does preparation alone solve the RET06 retrieval regression?

## Common mistakes, troubleshooting and recovery

Common mistakes: treating words as model tokens; applying Task 04 aggressive preprocessing to evidence; dropping the final chunk; overlap >= size causing no progress; emitting redundant tails; crossing PDF pages while claiming one page; confusing raw offsets with cleaned offsets; treating positional IDs as permanent; silently truncating during token analysis.

Install requirements for pypdf. For unreadable/encrypted/image-only files, choose a valid unencrypted text-based sample. Missing tokenizer cache shows setup guidance; processing remains usable. Preload MiniLM explicitly using the established setup command before class; there are no new model downloads. An instructor may restore the known-working reference checkpoint; do not add duplicated solution modules. Validate `tests/test_document_processor.py`, both Python versions, the notebook and app. Checkpoint 8 here is exercise-specific; original Academic Assistant Checkpoint 8 remains pending, and original Checkpoint 6 still needs chunk retrieval.

## Key takeaway, limitation and transition

Evidence must remain readable and traceable before retrieval. Word windows are a simple transparent baseline, not sentence/heading-aware or OCR processing. They preserve source evidence without yet selecting it.

> We now have smaller, traceable chunks. Next we will embed those chunks and retrieve the most relevant evidence for a question.

Task 11 validation: 376 tests pass without skips on Python 3.11.17 and 3.12.14. The RAG preparation notebook executes offline with tokenizer-only analysis on both. Home/all ten pages, all three PDFs, size/overlap controls and missing-tokenizer guidance are checked; live browser upload of the attendance PDF succeeds. One existing upstream QA/NumPy deprecation warning remains. PDF extraction order may put the footer before body text; preserve this transparently rather than guessing header/footer removal.

Exercise 10 boundary demonstration: the canonical sentence “Students must maintain at least 75% attendance in each enrolled course.” split at seven words gives “Students must maintain at least 75% attendance” / “in each enrolled course.” With overlap three, the second window is “least 75% attendance in each enrolled course.” Compare the threshold and per-course qualification; overlap improves continuity without making every window a complete rule. The lab derives this sentence from the canonical attendance document.

# Exercise 11 — Semantic Chunk Retrieval

## Previous problem and why we teach it

Task 09 whole-document MiniLM inputs exceeded the 256-token window. Task 11 produced 152 readable, traceable passages with 100-word size and 20-word overlap; their maximum is 247 model tokens, including special tokens. This task changes retrieval granularity while retaining the encoder and cosine mathematics. It implements evidence selection only, not context, prompting, generation or RAG answers.

## Reference solution, TODOs and starter boundary

- **11.1 (★★ Core), encode_document_chunks:** validate chunk provenance in starter code; collect only each `text` in input order; call shared `encode_sentences`. Result N×384; blank text fails; empty corpus (0,384). No metadata or benchmark fields are embedded.
- **11.2 (★ Guided), encode_user_question:** reuse `encode_search_query`, so question and passages share the pinned encoder and one 384-element coordinate system. Blank input fails clearly.
- **11.3 (★★ Core), calculate_chunk_similarities:** reuse `calculate_semantic_similarities`, which reaches the previously taught manual cosine for each row. 152 rows → 152 finite scores, in original order; zero-vector handling remains unchanged.
- **11.4 (★★ Core), retrieve_top_k_chunks:** reuse `rank_documents` on chunk records; preserve descending stable ties and copied metadata; enumerate results from rank 1. Chunk IDs are unique, while source document IDs repeat. Do not deduplicate or rerank.

Starter composition validates matrix dimensions/count, preserves sorted original indices, and never reads expected facts. Same-length matrices supplied by an external caller cannot prove row identity; creation/order tests and cache identity enforce the normal pipeline. NLP code remains independent of Streamlit.

## Input, intermediate representation, calculation and result

14 canonical Markdown sources → existing processor → ordered 152 chunk records → batch text-only matrix (152,384). A question becomes a (384,) vector. Calculate dot product/magnitudes through existing manual cosine, then stable descending rank. Display shape/norm/first values, all scores, sorted indices, evidence text and source metadata. Cosine is direction similarity, not confidence or answer accuracy.

Default K=5 is a teaching compromise: on the measured 14 applicable single-source cases, Hit@3 is 78.57% and Hit@5 is 85.71%. More K can add sources but also noise and repeated context. The UI limits K to 1/3/5/8. There is no automatic relevance threshold or abstention.

## Corpus, model and caching

Only the 14 canonical academic Markdown files are in the benchmark: no question papers or PDF duplicates. A selected/uploaded PDF may be queried separately and keeps physical page provenance. The exact MiniLM ID/revision, CPU baseline and local-only resource loader remain unchanged. Token lengths are checked before corpus encoding; oversized chunks raise a useful message instead of silently relying on encoder truncation. The source/default configuration is unchanged.

UI starter caches processed canonical sources and chunk matrices. Keys include ordered source content/identity, chunk size, overlap, model ID/revision. Question and K are excluded from corpus identity. A successful cached result retains the model and matrix for reuse. Missing-model guidance leaves the extraction/chunking sections visible. Source/setting changes rebuild the corresponding corpus; no automatic downloads or fallback embeddings are allowed.

## Evaluation strategy and mathematical honesty

`python -m workshop.evaluate_chunk_retrieval` writes `workshop/chunk_retrieval_results.json`. Expected source IDs/patterns exist only in the evaluator/tests, never inside search. The UI has an explicitly labelled display-only recorded evaluation panel, separate from live retrieval. It never uses recorded expected evidence to select or alter live results.

The derived evaluation contains 16 unchanged applicable Task03 queries plus 11 Task12 examples. RET-14/RET-18 are explicitly excluded because they concern question papers outside this corpus. Eight original single-source cases and six supplementary single-source cases contribute to document Hit@1/3/5 and MRR. Multi-source, comparison, summary and unsupported queries are reported separately.

Expected source IDs are strict. A correct fact repeated in another document may still fail that source metric. Curated evidence checks require all specified patterns in one retrieved expected-source chunk, not a disconnected union of Top-K text. These checks are lightweight support checks, not proof of complete relevance or answer correctness. Never train/select outputs using expected references.

Across **14** applicable single-source cases: document Hit@1 **57.14%**, Hit@3 **78.57%**, Hit@5 **85.71%**, MRR **0.694777**. On the **eight original** single-source cases only: **87.5% / 100% / 100%**, MRR **0.916667**. Do not compare the expanded 14-case aggregate directly with Task09's eight-case aggregate.

## Whole documents versus chunks — measured examples

| Query | Historical whole-source semantic rank | First expected-source chunk rank |
|---|---:|---:|
| RET-02 TF-IDF unit | 6 | 1 |
| RET-06 device communication | 15 | 1 |
| RET-11 Semester 6 laboratory examinations | 4 | 1 |
| RET-17 NLP Unit 3 | 8 | 1 |

RET-06: whole-document TF-IDF rank 1, semantic rank 15. Chunk retrieval returns **HCE-DOC02::chunk_013**, rank **1**, cosine about **0.295140**. Its text contains **Computer Networks (HCE-CSE402)** and the profile describing protocols/layered architectures and data exchange over connected systems. This passes the curated evidence check. Earlier full-document encoding could miss this later content. Historical comparison used 18 full sources (including four papers); current retrieval uses 14 documents / 152 chunks. It is a granularity/corpus comparison, not a controlled proof that truncation alone caused the improvement.

For RET-09 the calendar improves from rank 6 to 2, but the syllabus regresses from 7 to 15; Top-5 covers only one of two expected sources. RET-12 examination evidence regresses from source rank 5 to 11, while the syllabus improves from 6 to 5; listed-source coverage is only 50% at K=5. Chunking does not improve every source rank or ensure all evidence is available.

## Direct facts, paraphrases, failures and coverage

Attendance question retrieves examination-rule chunk DOC09::002 first (0.6042), attendance-policy DOC08::000 second (0.5853); the latter contains the 75% rule. Minimum end-semester question retrieves a DL assessment chunk first (0.6941), then examination DOC09::002 (0.6622), containing the 24-mark rule. NLP semester query retrieves syllabus DOC04::000 first (0.6416), with semester 6/course identity. NLP prerequisites query reaches syllabus DOC04::001 at rank 4 (0.5418), containing AI and ML prerequisites; prerequisites-register chunks rank higher.

The low-attendance/final-examination paraphrase retrieves both attendance/examination sources within Top-5. The 'subjects before NLP' paraphrase retrieves the prerequisite register, with AI/ML evidence in DOC13::013; syllabus source remains outside Top-5, so listed-source coverage is 50%.

Failures remain visible: 'How many credits does the NLP course carry?' first reaches the expected syllabus at rank 19; that first syllabus chunk is assessment prose, not the credit table. The human-communication paraphrase first reaches the expected syllabus at rank 11; the highest-ranked curriculum contrast discusses both NLP and networks and is related, so a strict expected-source miss is not complete semantic failure. The broad NLP-success question retrieves none of its three listed sources in Top-5 (nor Top-8); do not use these results to manufacture an answer. The ML/DL curriculum relationship example covers two of four listed sources at K=5; the simpler RET-16 focus comparison covers both syllabi.

Unsupported cake recipe: nearest result is DOC06::000, about **0.163470**, with no cooking evidence. No abstention is implemented, and this score is not a universal cutoff.

## Overlap redundancy

RET-01 ranks attendance DOC08::001 and DOC08::000 at 1 and 2; their source windows share 20 word positions. RET-06 ranks curriculum DOC02::013 and DOC02::014 at 1 and 4, also sharing 20 positions. The lab shows such pairs dynamically. Overlap protects boundary context but duplicates evidence and can crowd out other sources. Do not introduce MMR, cross-encoders, deduplication or diversity here; context construction will consider redundancy later.

## Teaching steps and live demonstration

1. Recap 14 documents → 152 chunks and the token maximum.
2. Ask the attendance question; inspect question vector and complete score vector.
3. Trace the first two chunks to their source/offsets and read the actual rule.
4. Paraphrase the question and compare measured ranks rather than predicting exact scores.
5. Ask the broad NLP-completion question and expose missing sources.
6. Inspect neighboring overlap and K=1/3/5/8 tradeoffs.
7. Run RET-06 and compare the recorded whole-document rank with actual passage text.
8. Show credit-query and human-communication limitations without editing the corpus.
9. Try the cake query; distinguish nearest neighbor from useful evidence.
10. Ask whether evidence alone answers a question; transition to controlled context.

## Questions, common mistakes and recovery

Ask: What does one matrix row represent? Why use the same encoder? Did cosine change? Why can neighbors both rank highly? Is Top-1 enough? Does a high score prove relevance? Why preserve IDs/pages/offsets? What happens for unsupported queries? How should five passages be combined? Should later context carry source labels?

Common mistakes: encoding IDs/title instead of text; losing row alignment; changing query encoder; recomputing corpus vectors per question; sorting ascending; stripping provenance; calling scores confidence; treating all Top-K as relevant; confusing retrieval with generation; reading expected answers inside search. For missing cache, run the established explicit MiniLM preflight. Oversized alternative settings require inspection/smaller chunks, not silent truncation. Use the existing reference checkpoint for instructor recovery; no duplicate solutions are added.

## Checkpoint and transition

Exercise-specific Checkpoint 9: Relevant Evidence Can Be Retrieved. Original global portfolio/deployment Checkpoint 9 remains pending; global Checkpoint 6 is now implemented, but complete RAG Checkpoint 7 is not.

> Retrieval gives us evidence, not an answer. The next step is to combine the retrieved chunks into a controlled context, preserve their source labels, and construct a grounded prompt for a language model.


Task 12 CPU measurements (machine-specific, Python 3.11.17 evaluator): source processing **22.91 ms**, fresh-process cached model load **10.57 s**, 152-chunk encoding **5.72 s**. Across 27 queries, mean question encoding **10.77 ms**, manual cosine scoring **101.65 ms**, full 152-chunk ranking **0.67 ms**. Warm Streamlit matrix-cache lookup measured **20.05 ms** on Python 3.11 / **19.63 ms** on Python 3.12, returning the same model/matrix objects. These are local planning measurements, not guaranteed laptop timings.

Offline AppTest verified exactly one canonical chunk-build, one chunk-encoding call and one model-loader call across one edited question and K=1/3/5/8 changes. Selected-source, size and overlap changes each re-encoded the corresponding experimental corpus. In-memory source-content changes produced a new matrix; changed model revision did not reuse the old cache; 120/20 oversized chunks were rejected before encoding. All changes remained separate from canonical source files. The measured report records these cache checks.

Both Python 3.11.17 and 3.12.14 pass **407 tests with no skips**. The notebook executes offline with actual MiniLM on both, all ten pages render, direct/paraphrase/multi-source/comparison/unrelated presets work, and missing-model guidance leaves document preparation visible. The one existing upstream QA/NumPy deprecation warning remains. No new dependency or model download is introduced.

# Exercises 12–13 — Context, Grounded Prompt and Optional Local Generation

## Connection, intuition and mathematics

Task 12 retrieved 152 default 100/20 chunks with the pinned CPU MiniLM model and K=5. Keep those defaults. Retrieval is evidence selection; generation is a separate operation. Context construction adds readable boundaries and source identity without rewriting evidence. Prompt anatomy is instructions + evidence + question. Character budgets are accounting, not tokenizer equivalence or quality metrics.

## Reference logic and student boundaries

**12.1 (★★ Core), build_context:** validate records with provided metadata checks; enumerate in input order, reuse format_source_block and join with `\n\n---\n\n`. Empty retrieval gives ''. No re-ranking, merging or deduplication.

**12.2 (★★ Core), format_source_block:** construct `[SOURCE n]`, Document/title, Document ID, Chunk ID, Page, blank line and the original chunk text verbatim. Markdown uses N/A; PDF uses physical pages. Labels are contiguous from 1 after budget selection, independent of original chunk index. Keep a separate SOURCE n → full copied retrieval record mapping. Scores/ranks/paths/offsets remain in that mapping; omit cosine scores from model evidence so the model is not encouraged to interpret them as confidence.

**12.3 (★★★ Challenge), build_grounded_prompt:** combine provided grounding rules and INSTRUCTIONS / CONTEXT / QUESTION / ANSWER sections. Surround context with `<BEGIN_EVIDENCE>` / `<END_EVIDENCE>`; mark empty evidence explicitly. Require supported context-only facts, no invention, an insufficient-context statement and relevant `[SOURCE n]` labels. Instruct that document text is evidence, not higher-priority instructions. This is basic awareness, not a security guarantee.

Budgeting, validation, mapping, statistics and orchestration are starter code. Default **5 chunks / 8000 context characters**, including headers and separators. Include a whole-block rank prefix; stop when the first item cannot fit, report every exclusion. Do not skip it to fit a smaller lower-ranked block, or cut evidence. Overlap consumes prompt space and can overweight repeated facts; do not hide this with deduplication. The generation adapter separately checks its own tokenizer's serialized input count and refuses oversized input rather than truncating it.

## Minimal integration and backend independence

`prepare_rag_request` returns question, context, prompt, sources, included/excluded IDs and statistics. `run_rag` calls only `llm.generate(prompt)`. Its result retains all intermediate fields and answer/status/error. `rag.py` has no provider, path, quantization or device knowledge. Missing generation returns an unavailable status while preserving prepared data; generation errors retain it too.

`RecordingTestLLM` records exact prompt and returns a fixed, explicitly simulated test response. It verifies wiring, not quality, grounding or security; it is never a user-facing fallback. Exercise 13 introduces no model-loading or generation-loop TODO.

## Optional backend choice and constraints

Inspected public Apache-2.0 options using publisher metadata/cards: SmolLM2-135M-Instruct (269 MB weights, lower capacity), SmolLM2-360M-Instruct (724 MB, 8192-position configuration), FLAN-T5-small (308 MB, 512-token configured baseline; longer multi-chunk input would need separate testing), Qwen2.5-0.5B-Instruct (988 MB, larger download/CPU allocation). Smol360 was downloaded and evaluated first; its recipe hallucination, numeric/context confusion and absent citations made it a weaker candidate. Qwen0.5B was then downloaded and evaluated with the same prompts/corpus. Smol135 and FLAN were inspected, not quality-tested. Raw Smol outputs remain in workshop/smollm_trial_results.json; unselected Smol weights were removed.

Use `Qwen/Qwen2.5-0.5B-Instruct`, revision `7ae557604adf67be50417f59c2c2f167def9a775`, as an **optional educational default** using existing Transformers/PyTorch. Publisher: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct . No new runtime package, hosted inference, credentials, GPU or new quantization runtime. Float32 CPU weights cost more RAM than their stored checkpoint; retrieval must remain independently usable.

Explicit setup: `python -m src.llm_resources --download`; offline verification without --download. Cache is `.cache/local_llm/<revision>/`, avoiding Windows blob/snapshot duplication. Successful loads are process-cached; missing files or load errors show useful guidance. Page startup checks presence only; Generate loads the backend. File presence is not proof of loadability. Chat templating is backend infrastructure; UI can inspect both exact grounded text and model-specific serialized text. Decode only new tokens. Defaults: greedy `do_sample=False`, `num_beams=1`, `max_new_tokens=96`, `max_input_tokens=3072`, CPU/eval/inference mode, up to four CPU threads. No silent truncation or creative sampling.

## Grounding experiments and failure types

`python -m workshop.evaluate_local_llm` records unchanged real outputs, timings, source mappings and label presence in `workshop/local_llm_evaluation_results.json`. Compare weak 'Answer using this context' prompts with explicit grounded rules for attendance, numeric passing requirement, unsupported cake, missing NLP-credit evidence and a measured multi-source attendance/examination query. Compare attendance K=1/3/5; keep K=5/default retrieval unchanged. Do not secretly increase K for the credit question.

Retrieval failure: relevant evidence never entered the context. The NLP-credit expected syllabus first appears at rank 19, but Top-5 lab guidelines link the NLP lab to its corresponding four-credit theory course. This is a source miss, not pure fact absence. At attendance K=1 the percentage is absent and the model invents 80%. Generation failure: adequate evidence is present but the model misreads, invents facts or ignores labels. The cake prompt asks for insufficient context but does not guarantee abstention. Label syntax/existence checks are diagnostics, not factual citation validation. Judge actual claims against the supplied blocks manually.

## Live demonstration, questions and common mistakes

Demo: attendance → Top-5 → original chunks → labelled context → size/exclusions → exact prompt → optional real generation → compare answer and source labels. Then cake, credit failure, K=1/3/5 and weak/grounded output comparison. If no local model is installed, show retrieval/context/prompt normally; use a clearly labelled test backend in notebook or an instructor's prepared machine.

Ask: Is retrieval already generation? Context versus prompt? Why preserve labels? Should cosine be sent as authority? What does overlap cost? Can a prompt recover absent evidence? Does insufficient-context wording guarantee abstention? Why isolate loading? Does SOURCE 2 prove support? How do retrieval and generation failures differ?

Common mistakes: concatenating without boundaries; altering numbers; losing metadata; calling context an answer; sending similarity as confidence; unlimited context; hiding the prompt; assuming rules guarantee truth; treating generated labels as verified; coupling RAG to one backend; making retrieval depend on optional generation. For recovery restore the known-working reference checkpoint, not duplicate solution modules. Before class, test requirements/cache/offline CPU inference and memory; the optional model is not necessary for context exercises.

## Checkpoint and transition

Exercise-specific Checkpoint 10: Retrieved Evidence Becomes a Grounded Prompt. This does not complete the overall RAG checkpoint or portfolio/deployment readiness.

> We now have all of the core components: retrieval, evidence, provenance, a grounded prompt and a generation interface. The next task will assemble these pieces into a transparent RAG system where students can inspect every stage from question to sources to answer.


## Actual optional-model observations — Task 13

Final default: pinned Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0). Weight checkpoint 988,097,824 bytes; retained cache **999,603,564 bytes** (about 1.00 GB). Float32 parameter tensors **1,976,131,072 bytes**. Whole-process working set after experiments **2,123,493,376 bytes** includes MiniLM, generator, PyTorch and allocator; it is not a generator-only or peak guarantee. Other Transformer models loaded simultaneously need additional memory. Generator load after the retrieval stack initialized **5.49 s**; observed generations **4.70–26.41 s**. All raw outputs remain in the measured artifact.

| Experiment | Weak prompt | Grounded prompt |
|---|---|---|
| Attendance K=5 | At least 75% | 75% |
| End-semester marks | 24 marks with a weak supporting quotation | 50 total and 24 end-semester marks |
| Unsupported cake | Says recipe details are absent; output reaches cap | Says no recipe can be determined from supplied information |
| NLP credits K=5 | 4 credits | 4 credits |
| Attendance/final-exam multi-source | Invents shortage of “24 out of 20 marks” | Incorrectly ties eligibility to failing the pass condition |

None emitted `[SOURCE n]`. Label presence alone would not verify support. The weak numeric quotation does not establish the threshold. Grounded multi-source wording is cleaner but still wrong: attendance eligibility differs from examination passing. This is generation failure despite relevant policy evidence. These models are optional educational demonstrations, not validated academic advice.

Credit-case correction: SOURCE 2, HCE-DOC10::chunk_001, links NLP/ML/DL labs to corresponding four-credit theory courses. The 4-credit answer has indirect context support; the model's internal reasoning is unknown. Expected syllabus rank 19 remains a strict source metric. Do not present that source miss as proof all credit evidence is absent. No data or hidden K override was used.

Attendance K=1 generated **“The minimum attendance required is 80%.”** The selected context lacks the threshold, so this is unsupported/wrong. K=3 generated **“The minimum attendance required is 75%.”** K=5 also gives 75%. This genuine missing-evidence experiment keeps default K=5 elsewhere. A tight context budget can similarly exclude useful evidence.

Smol360 was rejected as default after recipe hallucination, marks/credits confusion and multi-source errors. Its raw trial remains in workshop/smollm_trial_results.json; its 726 MB cache was removed. Retained extra classroom storage is approximately 1.00 GB for Qwen. Replacing the adapter does not require changing rag.py.

Optional-model testing policy: only the real generator integration check is skipped on unprepared clones. Core RAG/context, missing-backend, error-handling and deterministic-test-backend tests do not skip. Both prepared development environments ran all 434 tests with no skips.

Task 13 validation: 27 new tests; all **434 tests pass with no skips** on prepared Python 3.11.17 and 3.12.14. The expanded notebook executes offline with actual Qwen on both. AppTest verifies home/all ten pages, context budgets/exclusions, full prompt, optional CPU generation, actual chat serialization and missing-generator isolation; page startup invokes no generator loader. One existing upstream QA/NumPy warning remains.

# Exercise 14 — Transparent RAG Integration

## What is RAG and what is now complete?

External source content is retrieved, formatted into evidence context and supplied with instructions to an optional generative model. The complete educational pipeline is inspectable, including no-generation mode. It is not production-ready. No new algorithm TODO is introduced; retain the 44 existing IDs. Model inventory, prompt wording, 100/20 chunks, canonical fourteen Markdown files, K=5 and Qwen's CPU float32/greedy 96-output/3072-input settings are frozen.

## Four separate concepts

**Retrieval:** what Top-K found. **Context:** which blocks survived budgeting and entered the exact prompt. **Generation:** model output. **Attribution:** provenance of retrieved/included blocks, independent of model text. Expected/preferred source hits are not the same as evidence support; Laboratory Guidelines can support NLP's four-credit theory relationship even when the preferred syllabus misses Top-5.

`run_transparent_rag(question, retriever, llm=None, ...)` invokes the existing retrieval callback, existing context/prompt preparation and shared optional-generation boundary. It returns question; retrieval embedding/scores/sorted indices/results/K; included/excluded context/statistics; prompt; generation availability/status/answer/error/model; deterministic attribution; structural diagnostics; stage timings. Initialization/loading are caller responsibilities. There is no provider coupling, expected-answer lookup, reranking or LLM judge.

## Source attribution and label inspection

Sources Supplied to the Model are ordered copies of the exact included SOURCE mapping. Display them even when the answer contains no source markers or no answer was generated. In prompt-only mode they are prepared model inputs, not proof inference occurred. Show document title/ID, chunk/index, physical page or N/A, original retrieval rank, text and full offsets/path. Do not label the list verified citations.

The source-label helper recognizes `[SOURCE n]` case-insensitively, normalizes numeric labels, deduplicates mentions in appearance order and records occurrence count. Classify no labels, known labels and unknown labels. Known-label validity is structural only; a false claim can cite a real source. Production never performs automatic entailment/factuality scoring. Diagnostics flag overlap, context exclusions, one-document context, unavailable/error generation and absent/unknown labels, without declaring correctness.

## UI and classroom strategy

Question, generated answer and deterministic sources appear first. Expand details for extraction/chunking, evidence, embeddings/scores, context, exact prompt, model and diagnostics. One pipeline supports Retrieval + Prompt Only and Full Local Generation. Opening the page does not load/download Qwen. Relevant input fingerprints clear stale answers when query/K/budget/corpus/settings change; details expansion does not generate again. Canonical corpus embeddings remain cached independently of question/K.

Every student can inspect retrieval, context, prompts and diagnostics. Capable machines/instructor can generate locally. Optional Qwen uses approximately 1 GB cache and Task13 observed about 2.12 GB whole-process RAM; constrained machines can omit it. Timings are observations, not guarantees. Missing Qwen disables generation only. Core tests do not need it; the existing real-model integration check remains conditional on explicit preload.

## Evaluation and scope

`python -m workshop.evaluate_rag` records pre-generation experiments; `--generate` uses the preloaded actual Qwen. Use `--output` for separate reports. Twelve compact cases cover attendance K=1/3/5, numeric passing rules, NLP semester/prerequisites/credits, attendance-exam multi-source, cake, RET-06, ML/DL comparison and an explicit one-chunk context-budget ablation. Read the measured `workshop/rag_evaluation_results.json`; do not populate application answers from it.

Expected facts are derived only in the evaluator from the canonical JSON. Curated evidence classifications are direct support, indirect support or insufficient; conflicting/ambiguous support can be recorded after human review. Deterministic output checks are case-specific numbers/phrases, not general semantic correctness. Multi-source checks record source coverage, the supplied shortage-to-ineligibility relationship, missing numeric facts and known wrong pass-condition wording separately. Label existence is not evidence entailment. Metadata records UTC time, Python/package versions, model/revisions/settings, source hashes, prompt-template/request hashes and stage latency. Raw model outputs are retained unchanged.

## Failure taxonomy and teaching flow

A: retrieval omits useful facts. B: retrieved facts are excluded by budget. C: supplied facts are misinterpreted in generation. D: source markers are absent/unknown. E: no answer exists in the knowledge base. These can coexist; use human inspection rather than an automatic judge.

Live demo: attendance K=5 → question vector → all scores/Top-K → policy evidence → context → unchanged prompt → optional answer → independent source list → label diagnostics. Compare K=1/3/5 and then K=5 with context budget one. Try cake, multi-source attendance/examination, credits, RET-06 and comparison. Do not tune prompts/model/retrieval to conceal failures. Explain that changing retrieved or included evidence can change generation while the model/rules remain fixed.

Ask: Which stage found evidence? Which stage excluded it? Is a preferred-source miss always a fact miss? Does nearest neighbor imply relevance? Does a valid label prove a claim? Can a model fabricate despite grounding instructions? Why show sources without model citations? How can a budget create a new failure? Does generation improve when K increases universally?

Common mistakes: calling every Top-K chunk relevant, calling cosine confidence, confusing prepared context with a generated answer, treating retrieved sources as validated citations, showing an old answer beside changed evidence, interpreting substring checks as semantic accuracy, merging/deleting evidence silently, or requiring Qwen for every learner. Restore a known-working reference checkpoint for recovery. Extraction/PDF page rules and resource preflights remain unchanged.

## Checkpoint and next topic

Exercise checkpoint 11: Complete Transparent RAG Pipeline; original educational RAG Checkpoint 7 is now ready with optional generation. Final Assistant/deployment readiness remains pending. Next tasks address question-paper analysis and final integration; no Task15 work is included here.


## Task14 fresh measured outcomes

Twelve full-generation cases were rerun with unchanged prompt/model/retrieval settings. Attendance: K=1 lacks 75% → **80%** (wrong); K=3 contains policy evidence → **75%**; K=5 → **75%**. Explicit K=5/context-one ablation retrieves useful evidence but excludes it and again gives **80%**, demonstrating context failure separately from retrieval depth.

Passing-rule output gives **50 total / 24 end-semester marks**. Semester output identifies **Semester 6**, but reaches the output cap mid-explanation. Prerequisite output says the student must take NLP before NLP, despite supplied AI/ML evidence: generation failure. Cake retrieves nonzero-scored academic chunks but Qwen says the recipe cannot be determined. This observed abstention is not a guarantee.

Multi-source evidence supplies the shortage-to-ineligibility relationship from attendance and examination sources. The exact 75% threshold is absent from these particular Top-5 blocks; the rule relationship is nevertheless directly available. Qwen incorrectly says failure of a pass condition prevents sitting the examination. Evaluate this relationship separately from a missing-number check; do not classify all policy evidence as absent just because 75 is missing.

Credits: expected syllabus misses Top-5, but Lab Guidelines indirectly link NLP's lab to its four-credit theory course. Qwen answers **4 credits**; source-level metrics and actual support differ. RET-06: the supplied curriculum chunk explicitly describes Computer Networks, yet Qwen answers **machine learning courses** — a clear generation failure after successful retrieval. ML/DL comparison is fluent, but its assignment of clustering/dimensionality reduction to DL needs human review. No automatic comparison correctness is claimed.

Curated useful-evidence availability: **10/12 in retrieved results**, **9/12 in supplied contexts**. The unsupported cake and shallow K=1 attendance lack relevant evidence; the explicit budget ablation loses evidence after retrieval. Four chunks are excluded across the budget experiment. There are **5/10 passing expected-fact number/phrase checks** and **4 known-error phrase flags**; neither is a general answer-accuracy metric. Comparison/cake have different evaluation dimensions. All **12 generated answers emit no SOURCE labels** (0% emission, zero unknown labels). Deterministic supplied-source lists remain complete regardless.

Per-request observed latency (excluding corpus/model setup): average retrieval **0.284 s**, context/prompt preparation **0.0004 s**, generation **14.292 s**, total **14.576 s**; total range **5.14–28.17 s**. Generation dominates. Exact timings, source coverage, context blocks, prompt/request hashes, package versions and raw outputs are in the measured artifact. Evidence annotations were refined after reading actual policy chunks; no generated answer, prompt, dataset or model setting was edited to improve outcomes.

Task14 verification: the completed RAG notebook executes offline on Python 3.11.17 and 3.12.14 with actual optional Qwen. Home/all ten pages render. Prompt-only/full-generation modes, deterministic source display, stale-answer invalidation, corpus cache reuse, budget exclusions, PDF provenance and missing-Qwen behavior were verified. Prepared full suites pass 449 tests on both versions with zero skips; core functionality has no skips, and only the existing optional generator integration check is conditional on preload. No prompt, model, retrieval defaults or canonical data changed.

# Exercise 15 — Deterministic Question Paper Analyzer

## Why this feature is different

Structured headings and an explicit taxonomy justify a small standard-library pipeline. No model, embedding, hosted inference or new dependency is necessary. It is faster to inspect and reproduce than RAG, but lexical coverage is limited. Exercise 15 has exactly five new cores **14.1–14.5**; algorithm ID numbering is intentionally separate from exercise number.

## Extraction and architecture

`load_question_papers` discovers only sorted `nlp_question_paper_*.md` files. Each Paper ID is read from its actual header. `extract_questions` parses `## Qn — m marks` boundaries, retaining the full question body/continuation/subparts as one slot; IDs must be sequential and unique, marks/text positive/non-empty. Header prose is excluded. Four NLP-P01–P04 papers contain 12 slots and 60 marks each, 48 total. Parsing, reporting and UI are starter infrastructure; no new notebook expands the established eight-notebook architecture.

Production needs only paper text and projected taxonomy fields (ID/name/aliases). It does not read `question_metadata.json`, target distributions or expected answers. `load_topic_taxonomy` projects only `nlp_topic_taxonomy` from the contract; related IDs and subtopics are not synonym rules. The separate evaluator loads assignments only after production predictions exist. Tests copy papers into a folder with invalid metadata and a taxonomy-only contract and confirm identical output; a guarded read test rejects any metadata access.

## Reference logic and TODOs

- **14.1 (★ Guided), normalize_question_text:** Unicode NFKC/casefold, punctuation/underscore/hyphen boundaries, joined TFIDF → tf idf, whitespace collapse. Keep alphanumerics; no stemming/stopwords. Word2Vec remains one term. Offsets address normalized strings, not original paper bytes.
- **14.2 (★★ Core), match_question_topics:** collect boundary-aware occurrences of canonical names/aliases prepared by the helper. Permit only a simple final-word singular/plural s variation, not morphological/semantic inference. Duplicate normalized forms prefer canonical names. Rank by specificity tier, phrase word count, taxonomy position, occurrence and canonical-before-alias. Return a selected primary plus every explicit match/candidate/reason. No match is UNMATCHED; no IDs participate in classification.
- **14.3 (★★ Core), count_topic_frequencies:** initialize all 16 zero-count topics and UNMATCHED; increment exactly one primary per slot. Secondary/related matches do not inflate counts.
- **14.4 (★ Guided), rank_topics:** sort observed frequency descending, taxonomy order ties, preserve canonical zero rows. UNMATCHED is displayed separately rather than ranked as an NLP topic.
- **14.5 (★★ Core), filter_questions_by_topic:** primary-only matching by default; optional secondary toggle includes explicit lexical matches. Return full copied provenance/explanations. Unknown selection raises a clear error.

Specificity tiers are transparent matching policy, not taxonomy redesign: CBOW/Skip-Gram/BERT 2; Word2Vec/Attention 1; others 0. These prevent narrow architectures from being swallowed by broad mentions. Equally specific/long topic matches are flagged and resolved by taxonomy order. For example, a CBOW/BERT comparison is ambiguous and selects CBOW by canonical order. This is deterministic, not a claim of perfect semantic classification.

## Canonical distinctions and measured examples

There are exactly 16 canonical categories. GPT and Generative Pre-trained Transformer are aliases of Language Models; no separate GPT/RAG category is invented. Word Embeddings, Word2Vec, CBOW and Skip-Gram remain distinct. A CBOW-only question does not acquire Word2Vec via related IDs.

NLP-P03/Q9: `Compare a single attention head with multi-head attention in terms of the relationships represented.` → normalized `compare a single attention head with multi head attention in terms of the relationships represented` → canonical Attention and alias `multi head attention` → primary attention, with the longer explicit alias winning within its specificity tier. NLP-P01/Q12 Continuous Bag of Words matches both broad Bag of Words and narrower CBOW; CBOW wins. NLP-P03/Q10 the full BERT expansion outranks the broad Transformers mention and masked-language-modeling alias.

## Evaluation results and limits

Run `python -m workshop.evaluate_question_analyzer`. It compares production predictions with ground-truth primary assignments, never passing labels into classification. The measured artifact records source hashes, extracted IDs/text/marks, predictions/matched phrases, ground truth, confusion pairs, frequency differences and per-topic precision/recall/F1.

Baseline: **48 questions, 39 correct, 9 incorrect (all unmatched), 0 matched-but-wrong, 81.25% accuracy**, no ambiguous canonical-paper questions. All emitted topic labels have precision 1.0; the zero-prediction Semantic Similarity row uses precision 0 by documented zero-denominator convention. Recall exposes missing alias coverage rather than hiding it.

Observed frequencies (ground truth): Transformers **8 (8)**; TF-IDF **4 (7)**; Word Embeddings **4 (5)**; Word2Vec **4 (4)**; Attention **3 (5)**; BERT **3 (3)**; Text Preprocessing **2 (2)**; Bag of Words **2 (2)**; N-Grams **2 (2)**; Language Models **2 (2)**; Tokenization **1 (2)**; Cosine Similarity **1 (2)**; CBOW **1 (1)**; Skip-Gram **1 (1)**; Semantic Search **1 (1)**; Semantic Similarity **0 (1)**; unmatched **9**.

Unmatched confusions: TF-IDF 3, Attention 2, Tokenization 1, Cosine Similarity 1, Word Embeddings 1, Semantic Similarity 1. IDF/component-only phrasing and implicit mechanisms lack an explicit canonical alias; do not add question-ID hacks or tune counts to targets. No post-evaluation question-specific rule or taxonomy change was made. Limited separator/plural rules were designed generally before evaluation.

Primary Transformer filter: **8 questions**, Q1 and Q7 in every NLP-P01–P04 paper. Primary Attention filter: **NLP-P01/Q4, NLP-P02/Q9, NLP-P03/Q9**. Word2Vec **4**, CBOW **1**, Skip-Gram **1**, Word Embeddings **4** remain distinct. Including explicit secondary matches may broaden filters; label the mode clearly. The per-paper matrix includes all canonical topics and an unmatched status row whose total preserves 48 slots.

Cold measured timing: extraction about **5.69 ms**, classification **62.37 ms**, aggregation/ranking **0.10 ms** for all papers. These are machine-specific measurements; compare with measured multi-second generation, not a universal speed guarantee. Historical counts in four undated synthetic papers do not predict future exams.

## Live demo and questions

Open one paper → identify Q1–Q12 → select an attention question → original/normalized text → every alias match → specificity/primary reason → analyze 48 → rank canonical topics → filter Transformers → inspect unmatched examples → enable explicitly labelled ground-truth evaluation → compare engineering tradeoffs with RAG.

Ask: Why no LLM here? What is a taxonomy/alias? Why word boundaries? Why preserve CBOW separately? What if two topics tie? Why one primary but visible secondaries? Can frequency predict the next exam? Why isolate ground truth? When would semantic matching help? Which pipeline is easier to debug?

Common mistakes: metadata-driven answers; hard-coded counts; substring matches such as BERT inside unrelated words; deleting hyphens to join terms; propagating related topics; double counting; forcing unmatched questions; nondeterministic ties; losing provenance when filtering; presenting counts as prediction; introducing models unnecessarily. Use the known-working reference checkpoint for recovery; no duplicate solution modules.

## Transition

> We now have two useful academic intelligence systems: a grounded document assistant pipeline and a deterministic question-paper analyzer. Task 16 will integrate these capabilities into the final Academic Assistant so users can ask questions, retrieve academic information, analyze previous papers and navigate the workshop's NLP features through one application.

Task15 validation: all 489 tests pass on both prepared Python versions without skips; 40 new analyzer tests cover extraction, boundaries, specificity, ambiguity, unmatched/frequencies, ties, provenance, filtering, evaluator integrity and ground-truth isolation. Offline AppTest verifies analyzer controls on both. All ten pages and RAG pre-generation/source display render correctly. Macro topic F1 is 0.856218; this four-paper evaluation is not a generalization guarantee. No later Assistant integration was implemented.

# Final Integration Exercise — Intelligent Academic Assistant

## Goal and teaching flow

Connect the existing systems through a deterministic router, not an autonomous agent. No new algorithm TODOs; retain 49 IDs. Demonstrate Academic Knowledge Q&A, Course/Curriculum Navigation, Previous Papers and Grounded Study Support. Course navigation shares the RAG route rather than creating a competing structured-fact engine.

Start with attendance evidence without generation. Show the route reason, supplied source text and prompt-ready context. Ask for NLP semester/prerequisites and its specific examination date. Then enable optional generation and distinguish supplied evidence from verified factual claims. Task14's weak Qwen reasoning and absence of source labels remain limitations; relevant chunks do not guarantee correct answers.

Switch to previous papers: eight primary Transformer questions, Word2Vec count four, Transformers top frequency eight, per-paper comparison. Topic resolution delegates to the existing taxonomy matcher, including self-attention, CBOW, Skip-Gram and GPT→Language Models. List/count/rank/comparison remain separate operations. No production metadata/target-count reads. Preserve nine unmatched questions and the historical-frequency disclaimer. Do not improve apparent accuracy through silent model relabeling.

Finally summarize NLP Unit 4 and request revision questions. Both retrieve syllabus evidence. Optional generation uses the unchanged grounded RAG prompt and backend. New questions carry generated provenance and an **AI-Generated Revision Questions** heading; paper filters carry historical/synthetic-paper provenance. Ask students why high similarity, valid labels and fluent output each fail to prove correctness.

## Routing and boundaries

Auto priority: strong paper requests → strong summary/revision requests → academic/navigation terms → weak study → explicit academic fallback. Papers override revision when both occur. “Which courses should I study before Deep Learning?” uses the academic/navigation rule rather than weak study. Modes override automatic selection. Matched rules, reason and fallback status are visible. No Qwen routing, tool planning, loops or new taxonomy.

`src/academic_assistant.py` receives independent resource callbacks. Paper requests call only the production analyzer. Academic/study requests call `run_transparent_rag`; optional backend failures retain evidence/context. MiniLM failure reports semantic retrieval unavailable while paper analysis remains usable. Paper-data failure does not disable RAG. UI caches content-dependent corpus resources and actual paper analysis. Input/mode/generation/top-K changes clear prior results. No downloads occur from ordinary page startup.

## Evaluation and recovery

Run `python -m workshop.evaluate_academic_assistant` for the 20-case deterministic/retrieval report. Optional `--with-generation --cases ATT SUMMARY CREATE --output <separate-path>` records model experiments without overwriting the standard baseline. Evaluation expectations never reach application code. Curated routing, paper intent/topic/count integrity and evidence-phrase checks are distinct from general accuracy. Inspect the measured artifact and the Task16 report for timing and evidence misses.

If generation fails, leave it disabled and teach from actual supplied evidence. If MiniLM is unavailable, teach deterministic papers independently. Missing paper data should be restored from the canonical Markdown sources, not metadata. Keep the known-working reference checkout; student starter checkpoint strategy remains a later release decision. No duplicated solution modules.



## Task16 measured integration observations

The 20-case retrieval-only run routes 20/20 correctly on this curated set. Paper intent 8/8, topic resolution 6/6 and observed count checks 5/5 pass. These are explicit-rule integration checks, not general accuracy. Supplied-evidence phrase checks pass 8/11: attendance 75%, pass values 50/24, Semester 6 and the specific NLP date 2027-05-07 are present. The short “What are the prerequisites for NLP?” request omits Artificial Intelligence from supplied context, though Machine Learning appears (including a misleading Deep Learning prerequisite passage). “Summarize NLP Unit 4” retrieves other syllabus units and misses BERT/GPT evidence. Do not infer a correct answer from the document title alone.

Optional actual Qwen experiments: attendance correctly outputs 75%. Unit 4 summary incorrectly describes count representations/TF-IDF; five generated revision questions also concern other units. Their generated labels/provenance work, but requested-unit relevance fails. No source labels were emitted in these three outputs. Generation took about 12.75 / 25.34 / 22.20 seconds, excluding loading. Preserve these failures; they motivate Task17 reliability inspection rather than silently optimizing this integration task. The frozen prompt, data, model and retrieval algorithm remain unchanged.

Recovery demonstration: paraphrase a prerequisite query and inspect evidence again; do not promise improvement. With no generation, students can read sources. Missing Qwen and missing MiniLM were separately fault-injected in the assistant page; papers remain independent. All 522 tests pass in both Python3.11.17 and Python3.12.14, no skips. The existing Transformer/NumPy warning is unchanged.

The NLP system is now functionally complete. Task 17 will focus on integration testing, usability, reliability, classroom resilience and consistent user experience across the entire application. Task 18 will then finalize documentation, architecture, GitHub presentation and deployment instructions.


# Task17 — Reliability, Integration and Classroom Resilience

## Current policy and measured failures

The frozen models, canonical data/taxonomy/papers,100-word chunks/20 overlap, K5 and grounded prompt are unchanged. No expansion, boost, filter, reranker or new algorithm TODO was added. Prerequisite evidence containing AI + ML ranks8 and10 for “NLP prerequisites”/“What are the prerequisites for NLP?”, but2 for the full course-name question. Wording/underspecification changes MiniLM geometry; a longer query is not universally guaranteed to work.

NLP unit headings rank: Unit1=3, Unit2=7, Unit3=1, Unit4=12, Unit5=7. Unit4 summary/topics/revision wording moves its heading-bearing chunk to9/36/29. Relevant structural text exists in the corpus; generic/neighboring NLP chunks rank higher.100-word windows can split heading and body. Exact Unit identifiers are weak semantic signals; Top-K ordering excludes the required evidence. This is not evidence that the syllabus is missing.

The assistant applies a minimal lexical **generation guard**, not altered retrieval. Course names/acronyms are derived from supplied canonical syllabus titles. An explicit unit request needs a matching course scope and the exact requested heading in supplied context. Missing/ambiguous scope or heading withholds Qwen loading/generation and exposes the context/diagnostic. This applies to all units, including1/4/5 and ML, not a Unit4 special case. Semester/course-code requests without a unit remain unchanged. A heading is necessary, not proof of complete coverage; mixed-course/multi-unit requests can be withheld conservatively. No expected answer/benchmark fact is read by production.

Task16's wrong-unit generated summary/revision examples are historical failures. Under Task17, the same requests now yield retrieval-only evidence with a visible withheld-generation warning. Pure RAG Lab behavior, Task12 rankings and Task14 contexts/prompts remain unchanged; attendanceK1 still lacks75%, multi-source reasoning/RET06/source-label weaknesses are not tuned away. Use the recorded generation outputs as failures, not solutions.

## Preflight, recovery and usability

`python -m workshop.preflight` inventories packages/data/resources locally; `--full-classroom` requires the learning caches; `--smoke` verifies actual CPU inference without Qwen. No automatic downloads. Missing Qwen is OPTIONAL; missing papers FAIL; missing model caches WARNING in partial mode/FAIL in strict mode. Model load errors give explicit setup guidance. Use document processing/manual attention/paper analysis when their optional learning models are unavailable.

All pages have descriptive titles/labels and text status messages. Retrieval cosine is not confidence; supplied sources are not verified citations. Generated and historical questions remain distinct. Tables use scrollable dataframes where broad coordinates were unwieldy. Embeddings show previews, not384 values. Prompts/evidence remain expandable. Preprocessing/BoW text and teaching-corpus sizes have classroom display limits; long encoder input shows token/truncation warnings. Qwen notes describe optional status, CPU delay/storage/memory and factual limitations.

TODO headers now have49 unique IDs. TODO1.7 retains two labelled core subregions, giving50 paired BEGIN/END regions; no starter stripping occurred. Guides/index match source. Overall checkpoints remain0–9; exercise milestones no longer compete with roadmap numbering. Measured earlier Task09/13/16 sections are historical baselines; current audit results are `workshop/reliability_audit_results.json`.

## Timing reality check (planning estimates, not measured student completion)

| Day | Hands-on block | Minutes |
|---|---|---:|
|1|Preprocessing|45|
|1|One-hot/BoW/N-Grams|45|
|1|TF-IDF + cosine/search|80|
|1|Word2Vec|40|
|1|Sentence embeddings|35|
|1|Semantic search|40|
|2|Attention|40|
|2|Transformer NLP tasks|35|
|2|Document extraction/cleaning/chunking|45|
|2|Chunk retrieval|40|
|2|Context/prompt|30|
|2|Transparent RAG|35|
|2|Question analyzer|45|
|2|Assistant integration|30|

These are285/300 minutes of blocks. Add introductions/review, lunch, two breaks and contingency to plan roughly two7-hour days. Model/package setup must happen before the workshop; downloads and cold notebook kernels can consume minutes and vary substantially by laptop/network. The80-minute TF-IDF/search block and49 cores need instructor pacing/recovery checkpoints. Qwen demos must not become the critical path: every student retrieves/prepares prompts; capable machines optionally generate; instructor demonstrates full generation. No curriculum redesign or packaging is performed here.

Task18 will finalize starter/reference packaging, documentation presentation and deployment instructions. It has not begun.

# Final Workshop Release — Task18

The functional architecture is frozen after Task17. Task18 only packages, documents, validates and prepares deployment. Follow workshop/INSTRUCTOR_START.md for preparation and WORKSHOP_SCHEDULE.md for a realistic two-day path. The canonical reference stays complete; the deterministic builder strips50 labelled regions for49 exercise IDs. Signature/docstring checks, leakage checks, reproducibility and runtime validation guard the release. The generated starter omits this Instructor Guide and measured reports.

Use the bounded recovery strategy: first preserve a student's work/commit, compare only the current exercise, then restore the smallest affected section or file. A full file can contain earlier completed TODOs, so review before replacing it. Actual published checkpoint names/commits will be owner decisions; no remote actions or invented tags occur here. Provide a bounded reference core after discussing the concept, rather than distributing a second hand-maintained answers module.

The existing exercise chapters cover why, intuition/math, reference logic, demonstrations, outputs, questions, mistakes, troubleshooting and transitions. Task17 measurements are the latest retrieval/UX baseline; older Task09/13/16 numbers are explicitly historical. Current release totals and starter/reference matrix are in workshop/RELEASE_VALIDATION.md. The repository license remains owner-selected; no legal license was silently assigned.

## Final release validation

Task18 reference suites pass574 tests on Python3.11/3.12 with zero skips, including23 packaging checks. The63-check offline audit and all11 reference page renders pass. All8 reference notebooks execute; all8 starter notebooks pause with TODO guidance, and11 starter pages render in both versions/fresh3.11. See workshop/RELEASE_VALIDATION.md for the exact fresh-install scope, matrix, resource costs and remaining owner decisions. No remote publication/deployment has occurred; repository license selection remains pending.
