"""Exercise 7: shared spaces, manual math, deterministic retrieval and evidence metrics."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest
from src.retrieval import (encode_corpus_documents, encode_search_query, calculate_semantic_similarities,
                           retrieve_semantic_results, semantic_search, score_documents,
                           prepare_search_corpus, search_tfidf)
from src.sentence_resources import (MODEL_CACHE, load_sentence_embedding_model,
                                    SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION)
from workshop.evaluate_retrieval import (load_course_documents, evaluate_expected_sources,
                                         evaluation_group, summarize_single_source_metrics)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def encoder():
    if not list(MODEL_CACHE.glob("models--*/snapshots/*")):
        pytest.skip("Prepare the shared model with python -m src.sentence_resources --download")
    return load_sentence_embedding_model()


def documents():
    return [{"document_id": "A", "title": "Language", "text": "Computers understand human language.", "semester": 6},
            {"document_id": "B", "title": "Network", "text": "Devices route packets across networks.", "semester": 4},
            {"document_id": "C", "title": "Database", "text": "Structured records use relational storage.", "semester": 3}]


def matrix():
    return np.eye(3, 384, dtype=np.float32)


def test_corpus_encoding_preserves_order_description_only():
    model = Mock()
    model.encode.return_value = matrix()
    rows = documents()
    original = deepcopy(rows)
    encoded = encode_corpus_documents(model, rows)
    assert encoded.shape == (3, 384)
    assert np.array_equal(encoded, matrix())
    assert model.encode.call_args.args[0] == [row["text"] for row in rows]
    assert rows == original
    assert "Language" not in model.encode.call_args.args[0]


def test_empty_corpus():
    model = Mock()
    encoded = encode_corpus_documents(model, [])
    assert encoded.shape == (0, 384)
    model.encode.assert_not_called()
    model.encode.return_value = matrix()[:1]
    result = semantic_search("valid query", [], encoded, model)
    assert result["scores"] == result["sorted_indices"] == result["results"] == []


@pytest.mark.parametrize("rows", [[{"text": "text"}], [{"document_id": "A", "text": "x"}]*2, [{"document_id": "A", "text": " "}]])
def test_invalid_documents(rows):
    with pytest.raises(ValueError):
        encode_corpus_documents(Mock(), rows)


def test_query_vector_shared_encoder():
    model = Mock()
    model.encode.return_value = matrix()[:1]
    vector = encode_search_query(model, "Keep Raw Case!")
    assert isinstance(vector, list) and len(vector) == 384
    assert vector == matrix()[0].tolist()
    model.encode.assert_called_once_with(["Keep Raw Case!"], batch_size=32, show_progress_bar=False,
                                        convert_to_numpy=True, normalize_embeddings=False)


@pytest.mark.parametrize("query,error", [("", ValueError), (" \n", ValueError), (7, TypeError)])
def test_blank_invalid_query(query, error):
    with pytest.raises(error):
        encode_search_query(Mock(), query)


def test_manual_similarity_order_and_zero_vectors():
    query = matrix()[0].tolist()
    with patch("src.retrieval.score_documents", wraps=score_documents) as manual:
        scores = calculate_semantic_similarities(query, matrix())
        manual.assert_called_once_with(query, matrix().tolist())
    assert scores == pytest.approx([1, 0, 0])
    assert calculate_semantic_similarities([0.0]*384, matrix()) == [0.0]*3
    assert calculate_semantic_similarities(query, np.empty((0,384))) == []


@pytest.mark.parametrize("vectors", [np.zeros((3,20)), np.zeros(384), np.full((3,384), np.nan), np.full((3,384), np.inf)])
def test_invalid_matrix(vectors):
    with pytest.raises(ValueError):
        calculate_semantic_similarities([1.0]*384, vectors)


def test_query_dimension_alignment():
    with pytest.raises(ValueError):
        calculate_semantic_similarities([1.0]*20, matrix())
    with pytest.raises(ValueError, match="rows"):
        semantic_search("query", documents(), matrix()[:2], Mock())


def test_rank_ties_metadata_top_k_and_no_mutation():
    rows = documents()
    original = deepcopy(rows)
    ranked = retrieve_semantic_results(rows, [0.2,0.8,0.8], 10)
    assert [r["document_id"] for r in ranked] == ["B","C","A"]
    assert [r["score"] for r in ranked] == [0.8,0.8,0.2]
    assert ranked[0]["semester"] == 4 and ranked[0]["text"] == rows[1]["text"]
    assert len(retrieve_semantic_results(rows,[1,0,0],1)) == 1
    assert retrieve_semantic_results(rows,[1,0,0],0) == []
    assert rows == original


@pytest.mark.parametrize("k", [-1, True, 1.2])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        retrieve_semantic_results(documents(), [1,0,0], k)
    with pytest.raises(ValueError):
        semantic_search("query", documents(), matrix(), Mock(), k)


def test_small_pipeline_intermediate_outputs():
    model = Mock()
    model.encode.return_value = matrix()[1:2]
    rows = documents()
    result = semantic_search("network query", rows, matrix(), model, 2)
    assert result["scores"] == pytest.approx([0,1,0])
    assert result["sorted_indices"] == [1,0,2]
    assert [r["document_id"] for r in result["results"]] == ["B","A"]
    assert result["query_vector"] == matrix()[1].tolist()


def test_real_corpus_order_finite_and_identical(encoder):
    rows = documents()
    vectors = encode_corpus_documents(encoder, rows)
    assert vectors.shape == (3,384) and np.isfinite(vectors).all()
    for index, row in enumerate(rows):
        assert vectors[index].tolist() == pytest.approx(encode_search_query(encoder,row["text"]), abs=1e-6)
    result = semantic_search(rows[0]["text"],rows,vectors,encoder,1)
    assert result["results"][0]["document_id"] == "A"
    assert result["results"][0]["score"] == pytest.approx(1, abs=1e-6)


def test_real_paraphrase_vs_unrelated(encoder):
    rows = [{"document_id":"A", "text":"Natural language processing helps computers understand human language."},
            {"document_id":"B", "text":"Database systems organize structured records."}]
    vectors = encode_corpus_documents(encoder, rows)
    result = semantic_search("Machines can analyze the meaning of human communication.",rows,vectors,encoder,2)
    assert result["scores"][0] > result["scores"][1]
    assert result["results"][0]["document_id"] == "A"


def test_real_course_corpus_no_ranking_quality_assumption(encoder):
    rows = load_course_documents()
    vectors = encode_corpus_documents(encoder,rows)
    assert vectors.shape == (20,384)
    output = semantic_search("Which subject teaches machines to work with human communication?",rows,vectors,encoder,5)
    lexical = search_tfidf("Which subject teaches machines to work with human communication?",prepare_search_corpus(rows),5)
    assert len(output["results"]) == len(lexical["results"]) == 5
    assert len(output["scores"]) == 20 and all(math.isfinite(score) for score in output["scores"])
    assert set(output["results"][0]) == set(lexical["results"][0])
    assert [r["score"] for r in output["results"]] == sorted([r["score"] for r in output["results"]],reverse=True)


def test_evaluation_ranks_coverage_and_multisource_not_top1():
    ranked = [{"document_id": word,"score": 1/(i+1)} for i,word in enumerate("ABCDE")]
    evaluated = evaluate_expected_sources(ranked,["B","E"])
    assert evaluated["ranks"] == {"B":2,"E":5}
    assert evaluated["any_expected_in_top_3"]
    assert evaluated["expected_source_coverage_at_3"] == 0.5
    assert not evaluated["all_expected_in_top_3"]
    assert evaluated["reciprocal_rank"] == 0.5
    assert evaluation_group({"type":"cross_document","expected_document_ids":["A","B"]}) == "multi_source_retrieval"
    assert evaluation_group({"type":"summarization","expected_document_ids":["A"]}) == "summary_evidence_only"
    assert evaluation_group({"type":"question_paper_analysis","expected_document_ids":["A"]}) == "analysis_papers_evidence_only"
    row = {"benchmark_id":"unit", "evaluation_group":"single_source_retrieval","tfidf":evaluated}
    excluded={**row,"evaluation_group":"multi_source_retrieval"}
    metrics=summarize_single_source_metrics([row,excluded],"tfidf")
    assert metrics["count"] == 1 and metrics["top_1_accuracy"] == 0
    assert metrics["top_3_hit_rate"] == 1 and metrics["mrr"] == 0.5


def test_measured_artifact_provenance_and_existing_benchmark_coverage():
    report=json.loads((ROOT/"workshop/retrieval_comparison_results.json").read_text(encoding="utf-8"))
    benchmarks=json.loads((ROOT/"data/retrieval_benchmarks.json").read_text(encoding="utf-8"))["benchmarks"]
    assert report["model_id"] == SENTENCE_MODEL_ID and report["model_revision"] == SENTENCE_MODEL_REVISION
    assert report["performance"]["course_matrix_shape"] == [20,384]
    assert [r["benchmark_id"] for r in report["benchmarks"]] == [b["benchmark_id"] for b in benchmarks]
    for path,digest in report["source_sha256"].items():
        data = (ROOT/path).read_bytes()
        if path == "data/document_manifest.json":
            # Task 11 adds a separate derived-PDF inventory. Preserve the exact
            # original academic/paper inventory used by the historical run.
            manifest = json.loads(data)
            manifest.pop("derived_sample_pdfs", None)
            data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").replace("\n", "\r\n").encode()
        assert hashlib.sha256(data).hexdigest() == digest
    for method in ["tfidf","semantic"]:
        recalculated = summarize_single_source_metrics(report["benchmarks"],method)
        for metric, value in report["single_source_metrics"][method].items():
            if isinstance(value, float):
                assert value == pytest.approx(recalculated[metric], abs=1e-12)
            else:
                assert value == recalculated[metric]
    for row in report["benchmarks"]:
        assert row["expected_document_ids"] == next(b["expected_document_ids"] for b in benchmarks if b["benchmark_id"]==row["benchmark_id"])
        assert all(1 <= rank <= 18 for method in ["tfidf","semantic"] for rank in row[method]["ranks"].values())


def test_student_boundaries_and_application_ground_truth_separation():
    source=(ROOT/"src/retrieval.py").read_text(encoding="utf-8")
    for todo in ["7.1","7.2","7.3","7.4"]:
        assert source.count(f"BEGIN STUDENT CORE {todo}") == 1
        assert source.count(f"END STUDENT CORE {todo}") == 1
        for path in ["docs/STUDENT_GUIDE.md","docs/INSTRUCTOR_GUIDE.md","workshop/TODO_INDEX.md"]:
            assert todo in (ROOT/path).read_text(encoding="utf-8")
    # No expected-ID/benchmark file access anywhere in the retrieval implementation.
    assert "expected_document_ids" not in source
    assert "retrieval_benchmarks.json" not in source
