"""Task16 measured integration evaluation. Expectations stay outside production."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import re
import time
from src.academic_assistant import run_academic_assistant, load_analyzer_resources
from src.retrieval import load_canonical_documents, build_chunk_corpus, encode_document_chunks, semantic_chunk_search
from src.sentence_resources import load_sentence_embedding_model
from src.llm_resources import load_local_llm
ROOT = Path(__file__).resolve().parents[1]


def cases():
    # Curated evidence phrases assess supplied context, not generated accuracy.
    rows = [
        ('ATT', 'What is the minimum attendance required?', 'academic_qa', 'retrieval_only', None, None, ['75%']),
        ('EXAM', 'What marks are required to pass?', 'academic_qa', 'retrieval_only', None, None, ['50', '24']),
        ('SEM', 'In which semester is NLP offered?', 'academic_qa', 'retrieval_only', None, None, ['HCE-CSE603', r'Semester\s*[:|]?\s*6']),
        ('PRE', 'What are the prerequisites for NLP?', 'academic_qa', 'retrieval_only', None, None, ['Artificial Intelligence', 'Machine Learning']),
        ('DATE', 'When is the NLP end-semester examination?', 'academic_qa', 'retrieval_only', None, None, ['2027-05-07']),
        ('PARA', 'Where do connected devices learn the rules for exchanging messages?', 'academic_qa', 'retrieval_only', None, None, ['Computer Networks']),
        ('MULTI', 'What do I need to successfully complete NLP?', 'academic_qa', 'retrieval_only', None, None, []),
        ('COMPARE', 'Compare Machine Learning and Deep Learning.', 'academic_qa', 'retrieval_only', None, None, []),
        ('SUMMARY', 'Summarize NLP Unit 4.', 'study_support', 'retrieval_only', None, None, ['GPT', 'BERT']),
        ('REVISE', 'What should I revise for Attention?', 'study_support', 'retrieval_only', None, None, []),
        ('CREATE', 'Create five revision questions for NLP Unit 4.', 'study_support', 'retrieval_only', None, None, []),
        ('FILTER', 'Show Transformer questions from previous papers.', 'question_papers', 'topic_filter', 'transformers', 8, []),
        ('COUNT', 'How many Word2Vec questions appeared?', 'question_papers', 'topic_count', 'word2vec', 4, []),
        ('RANK', 'What are the most frequent topics in previous NLP papers?', 'question_papers', 'topic_frequency', None, None, []),
        ('PAPERS', 'Compare topics across previous papers.', 'question_papers', 'paper_comparison', None, None, []),
        ('AMBIG', 'What Transformer topics should I revise from previous papers?', 'question_papers', 'topic_filter', 'transformers', 8, []),
        ('ALIAS', 'Show self-attention questions.', 'question_papers', 'topic_filter', 'attention', 3, []),
        ('GPT', 'Show GPT questions.', 'question_papers', 'topic_filter', 'language_models', 2, []),
        ('UNKNOWN', 'Show quantum teleportation questions.', 'question_papers', 'unsupported', 'UNMATCHED', None, []),
        ('UNSUPPORTED', 'What is the recipe for chocolate cake?', 'academic_qa', 'retrieval_only', None, None, []),
    ]
    return [dict(zip(['case_id', 'question', 'expected_capability', 'expected_type', 'expected_topic', 'expected_count', 'evidence_patterns'], row)) for row in rows]


def evaluate(with_generation=False, selected=None):
    init = time.perf_counter()
    resources = load_analyzer_resources(ROOT)
    model = load_sentence_embedding_model()
    chunks = build_chunk_corpus(load_canonical_documents(ROOT), 100, 20)
    matrix = encode_document_chunks(model, chunks)
    backend = load_local_llm() if with_generation else None
    initialization = time.perf_counter() - init
    retrieve = lambda q, k: semantic_chunk_search(q, chunks, matrix, model, top_k=k)
    results = []
    for case in cases():
        if selected and case['case_id'] not in selected: continue
        result = run_academic_assistant(case['question'], retriever=retrieve,
            analyzer_loader=lambda: resources, llm_loader=lambda: backend,
            generation_enabled=with_generation)
        data = result['data']
        rag = data.get('rag', {})
        context = rag.get('context', {}).get('text', '')
        topic = data.get('topic_id', data.get('request', {}).get('topic_id'))
        row = {**case, 'actual_capability': result['capability'],
            'routing_correct': result['capability'] == case['expected_capability'],
            'routing': result['routing'], 'answer_type': result['answer_type'],
            'type_correct': result['answer_type'] == case['expected_type'] if not with_generation or result['capability'] == 'question_papers' else None,
            'actual_topic': topic, 'topic_correct': topic == case['expected_topic'] if case['expected_topic'] else None,
            'result_count': data.get('count'), 'count_correct': data.get('count') == case['expected_count'] if case['expected_count'] is not None else None,
            'source_documents': list(dict.fromkeys(s.get('document_id', s.get('paper_id')) for s in result['sources'])),
            'source_chunks': [s['chunk_id'] for s in result['sources'] if 'chunk_id' in s],
            'generation_used': rag.get('generation', {}).get('status') == 'generated',
            'answer': result['answer'], 'warnings': result['warnings'],
            'diagnostics': result['diagnostics'],
            'evidence_checks': {p: bool(re.search(p, context, re.I)) for p in case['evidence_patterns']},
            'notes': 'Evidence checks inspect supplied context only, not answer correctness. Unsupported questions still receive nearest evidence; relevance/abstention is not guaranteed.'}
        if result['answer_type'] == 'topic_frequency': row['top_topic'] = data['rows'][0]
        if result['answer_type'] == 'paper_comparison': row['comparison_total'] = sum(r['total'] for r in data['rows'])
        results.append(row)
    paper_rows = [r for r in results if r['expected_capability'] == 'question_papers']
    checks = [v for r in results for v in r['evidence_checks'].values()]
    def fraction(field, rows):
        values = [r[field] for r in rows if r[field] is not None]
        return {'passed': sum(values), 'total': len(values), 'fraction': sum(values)/len(values) if values else None}
    return {'task': 16, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'generation_requested': with_generation,
        'initialization_seconds': initialization, 'case_count': len(results),
        'metrics': {'routing': fraction('routing_correct', results),
                    'paper_intent': fraction('type_correct', paper_rows),
                    'paper_topic': fraction('topic_correct', paper_rows),
                    'paper_count': fraction('count_correct', paper_rows),
                    'supplied_evidence_phrase_checks': {'passed': sum(checks), 'total': len(checks)}},
        'scope': 'Curated integration checks, not general routing/model accuracy. No metadata or expectations enter production.',
        'results': results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-generation', action='store_true')
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--output', type=Path, default=ROOT/'workshop/academic_assistant_results.json')
    args = parser.parse_args()
    report = evaluate(args.with_generation, args.cases)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'cases': report['case_count'], 'metrics': report['metrics'], 'output': str(args.output)}, indent=2))

if __name__ == '__main__': main()
