"""Assistant integration tests: real paper text, deterministic RAG test backend."""
from pathlib import Path
import pytest
from src.academic_assistant import (route_question, parse_paper_request,
    load_analyzer_resources, run_academic_assistant, FREQUENCY_NOTE)
from src.local_llm import RecordingTestLLM
from src.document_processor import process_document
from tests.test_transparent_rag import tiny_retriever
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def resources(): return load_analyzer_resources(ROOT)

@pytest.mark.parametrize('question,capability', [
 ('minimum attendance', 'academic_qa'), ('NLP prerequisites', 'academic_qa'),
 ('show Transformer questions', 'question_papers'),
 ('most frequent topics in previous papers', 'question_papers'),
 ('compare topics across previous papers', 'question_papers'),
 ('summarize NLP Unit 4', 'study_support'),
 ('create revision questions for NLP Unit 4', 'study_support'),
 ('what should I revise from previous papers', 'question_papers'),
 ('Which courses should I study before Deep Learning?', 'academic_qa'),
 ('how many Word2Vec questions appeared?', 'question_papers'),
 ('What happened to my bicycle?', 'academic_qa'),
 ('Show self-attention questions.', 'question_papers'),
])
def test_route_priority(question, capability):
    route = route_question(question)
    assert route['capability'] == capability and route['reason']

@pytest.mark.parametrize('mode', ['academic_qa', 'question_papers', 'study_support'])
def test_explicit_override(mode):
    assert route_question('previous paper revision', mode)['capability'] == mode


def test_fallback_and_invalid_mode():
    assert route_question('hello')['fallback']
    with pytest.raises(ValueError): route_question('a', 'bad')

@pytest.mark.parametrize('query,topic,count', [
 ('Show Transformer questions.', 'transformers', 8),
 ('Show self-attention questions.', 'attention', 3),
 ('Show CBOW questions.', 'cbow', 1),
 ('Show Skip-Gram questions.', 'skip_gram', 1),
 ('Show GPT questions.', 'language_models', 2),
 ('How many Word2Vec questions appeared?', 'word2vec', 4),
])
def test_topic_reuse_and_observed_integrity(resources, query, topic, count):
    def forbidden(): raise AssertionError('Model loader called by deterministic papers!')
    result = run_academic_assistant(query, analyzer_loader=lambda: resources,
                                   llm_loader=forbidden, generation_enabled=True)
    assert result['data']['topic_id'] == topic and result['data']['count'] == count
    assert FREQUENCY_NOTE in result['warnings']
    assert result['data']['unmatched_count'] == 9
    assert result['data']['provenance'] == 'historical/synthetic-paper'
    assert all(q['source_path'] and q['paper_id'] and q['question_id'] for q in result['data']['questions'])
    assert result['answer_type'] == ('topic_count' if 'How many' in query else 'topic_filter')


def test_frequencies_and_comparison(resources):
    run = lambda q: run_academic_assistant(q, analyzer_loader=lambda: resources)
    ranking = run('Most frequent topics in previous papers')
    assert ranking['data']['rows'][0]['topic_id'] == 'transformers'
    assert ranking['data']['rows'][0]['count'] == 8
    compared = run('Compare topics across previous papers')
    assert compared['answer_type'] == 'paper_comparison'
    assert sum(r['total'] for r in compared['data']['rows']) == 48


def test_unknown_topic_never_forced(resources):
    result = run_academic_assistant('Show quantum teleportation questions', analyzer_loader=lambda: resources)
    assert result['answer_type'] == 'unsupported' and 'No supported topic' in result['answer']
    assert result['data']['request']['topic_id'] == 'UNMATCHED'

def unit_retriever(question, k):
    doc = {'document_id': 'S', 'title': 'Natural Language Processing Syllabus',
           'source_type': 'markdown', 'source_path': 'syllabus.md',
           'pages': [{'page_number': None, 'text': '### Unit 4 — Contextual language models\nBERT and GPT are contextual language model topics.'}]}
    chunks = process_document(doc, 100, 20)['chunks']
    return {'question_vector': [.1]*384, 'scores': [.8], 'sorted_indices': [0],
            'results': [{**c, 'rank': i, 'score': .8} for i, c in enumerate(chunks[:k], 1)]}


@pytest.mark.parametrize('question,kind', [
 ('What is the attendance requirement?', 'rag_generated'),
 ('Summarize NLP Unit 4.', 'study_generation'),
 ('Create five revision questions for NLP Unit 4.', 'study_generation'),
])
def test_rag_reuse_exact_prompt_and_provenance(question, kind):
    backend = RecordingTestLLM('TEST generated output')
    result = run_academic_assistant(question, retriever=unit_retriever if 'Unit' in question else tiny_retriever,
                                   llm_loader=lambda: backend, generation_enabled=True)
    assert result['answer_type'] == kind and result['answer'] == 'TEST generated output'
    assert backend.prompts == [result['data']['rag']['prompt']]
    assert result['sources'] == result['data']['rag']['attribution']['supplied_sources']
    if kind == 'study_generation':
        assert result['data']['provenance'] == 'generated'
        assert 'AI-Generated' in result['data']['heading']
        assert not result['data'].get('questions')
    if 'Create' in question: assert result['data']['heading'] == 'AI-Generated Revision Questions'


def test_generation_unavailable_does_not_disable_retrieval():
    def missing(): raise RuntimeError('Qwen absent')
    result = run_academic_assistant('attendance?', retriever=tiny_retriever,
                                   llm_loader=missing, generation_enabled=True)
    assert result['answer_type'] == 'retrieval_only' and result['answer'] is None
    assert result['sources'] and result['data']['rag']['prompt']
    assert any('Qwen absent' in w for w in result['warnings'])


def test_generation_failure_keeps_context():
    class Broken:
        def generate(self, prompt): raise RuntimeError('CPU error')
    result = run_academic_assistant('attendance', retriever=tiny_retriever,
                                   llm_loader=Broken, generation_enabled=True)
    assert result['answer_type'] == 'retrieval_only'
    assert result['data']['rag']['generation']['status'] == 'error'
    assert result['sources']


def test_models_missing_papers_independent(resources):
    def unavailable(*args): raise RuntimeError('MiniLM missing')
    result = run_academic_assistant('Show Transformer questions', retriever=unavailable,
        llm_loader=unavailable, generation_enabled=True, analyzer_loader=lambda: resources)
    assert result['data']['count'] == 8
    academic = run_academic_assistant('attendance', retriever=unavailable)
    assert academic['answer_type'] == 'unsupported' and 'Previous-paper analysis remains' in academic['answer']


def test_missing_papers_does_not_disable_rag():
    def missing(): raise OSError('No papers')
    result = run_academic_assistant('attendance', retriever=tiny_retriever, analyzer_loader=missing)
    assert result['answer_type'] == 'retrieval_only' and result['sources']
    paper = run_academic_assistant('Show Transformer questions', analyzer_loader=missing)
    assert paper['answer_type'] == 'unsupported' and 'independently available' in paper['answer']


def test_common_structure_blank_and_no_leakage(resources):
    results = [run_academic_assistant(''),
               run_academic_assistant('attendance', retriever=tiny_retriever),
               run_academic_assistant('Show Transformer questions', analyzer_loader=lambda: resources)]
    for result in results:
        assert {'question', 'capability', 'routing', 'answer_type', 'answer', 'data', 'sources', 'warnings', 'diagnostics'} <= result.keys()
    source = (ROOT / 'src/academic_assistant.py').read_text(encoding='utf-8')
    for forbidden in ['question_metadata.json', 'target_primary_topic_counts', 'retrieval_benchmarks.json', 'import streamlit']:
        assert forbidden not in source


def test_empty_results_and_context_budget(resources):
    taxonomy, analysis = resources
    result = run_academic_assistant('Show Semantic Similarity questions', analyzer_loader=lambda: resources)
    assert result['data']['count'] == 0 and result['data']['questions'] == []
    result = run_academic_assistant('attendance', retriever=tiny_retriever, top_k=8)
    assert result['sources'] and len(result['sources']) <= 5
