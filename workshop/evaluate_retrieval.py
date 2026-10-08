"""Instructor evaluation only; application search never reads expected answers.

Run ``python -m workshop.evaluate_retrieval`` after model preflight. Produces a
measured artifact, not canonical facts. Normal operation is entirely local.
"""
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
from src.retrieval import (prepare_search_corpus, search_tfidf, encode_corpus_documents,
                           encode_search_query, calculate_semantic_similarities,
                           retrieve_semantic_results, semantic_search)
from src.sentence_resources import (load_sentence_embedding_model, SENTENCE_MODEL_ID,
                                    SENTENCE_MODEL_REVISION)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ["natural language processing", "machine learning", "database systems",
                   "computer communication networks",
                   "Which subject teaches machines to work with human communication?"]


def load_course_documents():
    with (ROOT / "data/search_corpus/course_descriptions.csv").open(encoding="utf-8", newline="") as file:
        return [{"document_id": row["course_code"], "title": row["course_name"],
                 "text": row["description"], "semester": int(row["semester"])}
                for row in csv.DictReader(file)]


def load_benchmark_documents():
    manifest = json.loads((ROOT / "data/document_manifest.json").read_text(encoding="utf-8"))
    # Match Task 06 order/text: fourteen full Markdown documents, then four papers.
    records = manifest["academic_documents"] + manifest["question_papers"]
    return [{"document_id": record["document_id"], "title": record["title"],
             "text": (ROOT / "data" / record["filename"]).read_text(encoding="utf-8")}
            for record in records]


def evaluate_expected_sources(ranked_results, expected_ids):
    """Evidence ranks/coverage only, not answer quality or topic-analysis output."""
    positions = {result["document_id"]: rank for rank, result in enumerate(ranked_results, 1)}
    ranks = {document_id: positions[document_id] for document_id in expected_ids}
    top3 = ranked_results[:3]
    count = sum(document_id in {result["document_id"] for result in top3} for document_id in expected_ids)
    return {"ranks": ranks, "top_3": [{"document_id": r["document_id"], "score": r["score"]} for r in top3],
            "any_expected_in_top_3": count > 0, "expected_source_coverage_at_3": count / len(expected_ids),
            "all_expected_in_top_3": count == len(expected_ids),
            "first_relevant_rank": min(ranks.values()), "reciprocal_rank": 1 / min(ranks.values())}


def evaluation_group(benchmark):
    if benchmark["type"] in {"question_paper_analysis", "question_filtering"}:
        return "analysis_papers_evidence_only"
    if benchmark["type"] == "summarization":
        return "summary_evidence_only"
    if benchmark["type"] == "comparison":
        return "comparison_evidence_only"
    if len(benchmark["expected_document_ids"]) > 1:
        return "multi_source_retrieval"
    return "single_source_retrieval"


def summarize_single_source_metrics(rows, method):
    eligible = [row for row in rows if row["evaluation_group"] == "single_source_retrieval"]
    return {"count": len(eligible), "benchmark_ids": [r["benchmark_id"] for r in eligible],
            "top_1_correct": sum(row[method]["first_relevant_rank"] == 1 for row in eligible),
            "top_1_accuracy": sum(row[method]["first_relevant_rank"] == 1 for row in eligible) / len(eligible) if eligible else None,
            "top_3_hits": sum(row[method]["any_expected_in_top_3"] for row in eligible),
            "top_3_hit_rate": sum(row[method]["any_expected_in_top_3"] for row in eligible) / len(eligible) if eligible else None,
            "mrr": sum(row[method]["reciprocal_rank"] for row in eligible) / len(eligible) if eligible else None}


def main():
    started = time.perf_counter()
    model = load_sentence_embedding_model()
    model_load = time.perf_counter() - started
    courses = load_course_documents()
    lexical = prepare_search_corpus(courses)
    started = time.perf_counter()
    course_vectors = encode_corpus_documents(model, courses)
    corpus_seconds = time.perf_counter() - started
    comparisons = []
    for query in DEFAULT_QUERIES + ["learning", "How do I bake a chocolate cake?"]:
        started = time.perf_counter()
        query_vector = encode_search_query(model, query)
        encoded_seconds = time.perf_counter() - started
        started = time.perf_counter()
        semantic_scores = calculate_semantic_similarities(query_vector, course_vectors)
        scored_seconds = time.perf_counter() - started
        started = time.perf_counter()
        semantic_results = retrieve_semantic_results(courses, semantic_scores, len(courses))
        ranked_seconds = time.perf_counter() - started
        tfidf_results = search_tfidf(query, lexical, len(courses))["results"]
        comparisons.append({"query": query, "query_encoding_seconds": encoded_seconds,
                            "cosine_scoring_seconds": scored_seconds, "ranking_seconds": ranked_seconds,
                            "tfidf_results": tfidf_results, "semantic_results": semantic_results})

    documents = load_benchmark_documents()
    benchmarks = json.loads((ROOT / "data/retrieval_benchmarks.json").read_text(encoding="utf-8"))["benchmarks"]
    document_lexical = prepare_search_corpus(documents)
    started = time.perf_counter()
    document_vectors = encode_corpus_documents(model, documents)
    document_seconds = time.perf_counter() - started
    token_lengths = model.tokenizer([d["text"] for d in documents], truncation=False,
                                   return_length=True, verbose=False)["length"]
    rows = []
    for benchmark in benchmarks:
        expected = benchmark["expected_document_ids"]
        query = benchmark["query"]
        tfidf = search_tfidf(query, document_lexical, len(documents))["results"]
        semantic = semantic_search(query, documents, document_vectors, model, len(documents))["results"]
        group = evaluation_group(benchmark)
        rows.append({"benchmark_id": benchmark["benchmark_id"], "query": query,
                     "type": benchmark["type"], "semantic_vs_keyword": benchmark["semantic_vs_keyword"],
                     "expected_document_ids": expected, "evaluation_group": group,
                     "tfidf": evaluate_expected_sources(tfidf, expected),
                     "semantic": evaluate_expected_sources(semantic, expected),
                     "notes": "Full Markdown inputs; MiniLM truncates at 256 word pieces. Ranks indicate source retrieval only, never generated answer correctness. "
                              + ("Included in single-source metrics." if group == "single_source_retrieval" else "Excluded from single-source aggregates; inspect evidence coverage/ranks separately.")})
    source_paths = [ROOT/"data/search_corpus/course_descriptions.csv", ROOT/"data/retrieval_benchmarks.json",
                    ROOT/"data/document_manifest.json"]
    manifest = json.loads(source_paths[2].read_text(encoding="utf-8"))
    source_paths += [ROOT/"data"/r["filename"] for r in manifest["academic_documents"]+manifest["question_papers"]]
    report = {"usage": "Measured instructor/testing output only. Not canonical facts or application answers. Regenerate with python -m workshop.evaluate_retrieval.",
              "model_id": SENTENCE_MODEL_ID, "model_revision": SENTENCE_MODEL_REVISION,
              "python": platform.python_version(), "packages": {p: version(p) for p in ["sentence-transformers","torch","transformers","numpy"]},
              "indexing_policy": "Course comparison: description only, same raw content supplied to both methods; classical preprocessing unchanged. Full-document benchmarks: unchanged complete Markdown, including headings/disclaimers. Encoder-native 256-word-piece truncation, no extraction/chunking.",
              "source_sha256": {str(p.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
              "performance": {"cpu_model_load_seconds": model_load, "course_count": len(courses), "course_matrix_shape": list(course_vectors.shape),
                              "course_corpus_encoding_seconds": corpus_seconds, "document_count": len(documents), "document_corpus_encoding_seconds": document_seconds},
              "course_comparisons": comparisons, "benchmarks": rows,
              "benchmark_input_lengths": [{"document_id": document["document_id"], "word_pieces_with_special_tokens": length,
                                            "encoder_truncates": length > model.max_seq_length}
                                           for document,length in zip(documents,token_lengths)],
              "single_source_metrics": {method: summarize_single_source_metrics(rows, method) for method in ["tfidf","semantic"]},
              "metric_exclusions": "Single-source aggregates include direct_fact/lexical_search/semantic_search with exactly one listed source. Multi-source, comparison, summary, analysis/paper queries report evidence ranks/coverage separately. A Top-3 hit is not recall of all required sources; paper analysis needs four papers. Source retrieval does not perform analysis, filtering, comparison or summarization."}
    destination = ROOT / "workshop/retrieval_comparison_results.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Measured {len(courses)} courses and {len(rows)} existing benchmarks; wrote {destination}")
    print(json.dumps(report["single_source_metrics"], indent=2))
    for row in rows:
        print(row["benchmark_id"], row["evaluation_group"], row["tfidf"]["ranks"], row["semantic"]["ranks"])


if __name__ == "__main__":
    main()
