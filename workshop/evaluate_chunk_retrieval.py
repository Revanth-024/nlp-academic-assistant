"""Instructor-only chunk evaluation. Expected evidence never enters search."""
from pathlib import Path
import hashlib
import json
import platform
import re
import time
from src.retrieval import (load_canonical_documents, build_chunk_corpus, encode_document_chunks,
    encode_user_question, calculate_chunk_similarities, retrieve_top_k_chunks)
from src.document_processor import model_token_lengths
from src.sentence_resources import (load_sentence_embedding_model, SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION)
ROOT = Path(__file__).resolve().parents[1]


def evaluation_cases():
    original = json.loads((ROOT/'data/retrieval_benchmarks.json').read_text(encoding='utf-8'))['benchmarks']
    cases = []
    for b in original:
        if any(not d.startswith('HCE-DOC') for d in b['expected_document_ids']):
            continue  # Question papers are outside the canonical 14-document corpus.
        category = ('comparison' if b['type']=='comparison' else
                    'summary_evidence' if b['type']=='summarization' else
                    'multi_source' if len(b['expected_document_ids'])>1 else
                    'semantic_paraphrase' if b['semantic_vs_keyword'] else 'direct_fact')
        cases.append({'query_id':b['benchmark_id'],'query':b['query'],'category':category,
                      'expected_documents':b['expected_document_ids'],'origin':'unchanged Task03 benchmark',
                      'single_source_metric':len(b['expected_document_ids'])==1 and b['type'] in ['direct_fact','lexical_search','semantic_search']})
    additional = [
        ('CHUNK-D01','direct_fact','What is the minimum attendance required?',['HCE-DOC08'],[r'75%',r'attendance']),
        ('CHUNK-D02','direct_fact','What is the minimum end-semester score required to pass?',['HCE-DOC09'],[r'24',r'end.semester']),
        ('CHUNK-D03','direct_fact','In which semester is Natural Language Processing offered?',['HCE-DOC04'],[r'semester\s*(?:\||:)?\s*6',r'HCE-CSE603']),
        ('CHUNK-D04','direct_fact','What are the prerequisites for Natural Language Processing?',['HCE-DOC04'],[r'(Artificial Intelligence|HCE-CSE501)',r'(Machine Learning|HCE-CSE502)']),
        ('CHUNK-D05','direct_fact','How many credits does the NLP course carry?',['HCE-DOC04'],[r'credits\s*(?:\||:)?\s*4',r'HCE-CSE603']),
        ('CHUNK-P01','semantic_paraphrase','When can low attendance stop a student from writing the final examination?',['HCE-DOC08','HCE-DOC09'],[r'(75%|shortage)',r'(examination|evaluation)']),
        ('CHUNK-P02','semantic_paraphrase','Which subjects must a student complete before taking NLP?',['HCE-DOC04','HCE-DOC13'],[r'(Artificial Intelligence|HCE-CSE501)',r'(Machine Learning|HCE-CSE502)']),
        ('CHUNK-P03','semantic_paraphrase','What subject teaches machines to work with human communication?',['HCE-DOC04'],[r'(language|NLP)',r'(processing|computational)']),
        ('CHUNK-M01','multi_source','What conditions must a student satisfy to complete the NLP course successfully?',['HCE-DOC04','HCE-DOC08','HCE-DOC09'],[]),
        ('CHUNK-C01','comparison','How are Machine Learning and Deep Learning related in the curriculum?',['HCE-DOC05','HCE-DOC06','HCE-DOC13','HCE-DOC02'],[]),
        ('CHUNK-U01','unsupported','What is the recipe for chocolate cake?',[],[]),
    ]
    for identity,category,query,expected,patterns in additional:
        cases.append({'query_id':identity,'category':category,'query':query,'expected_documents':expected,
            'evidence_patterns':patterns,'origin':'Task12 derived evaluation',
            'single_source_metric':len(expected)==1 and category in ['direct_fact','semantic_paraphrase']})
    # Curated evidence checks belong only to evaluation, not the retrieval API.
    curated = {'RET-01':[r'75%',r'attendance'], 'RET-02':[r'TF.IDF',r'Unit 2'],
               'RET-06':[r'Computer Networks',r'HCE-CSE'],
               'RET-11':[r'2027-04-26',r'2027-04-30'],
               'RET-12':[r'50',r'24',r'end.semester'],
               'RET-17':[r'Unit 3',r'(Word2Vec|distributed)']}
    for case in cases:
        if case['query_id'] in curated:
            case['evidence_patterns'] = curated[case['query_id']]
    return cases


def evidence_check(results, expected, patterns):
    """Conjunctive patterns must occur in ONE retrieved expected-source chunk."""
    if not patterns:
        return None
    matches = [c['chunk_id'] for c in results if c['document_id'] in expected and
               all(re.search(pattern,c['text'],flags=re.I) for pattern in patterns)]
    return {'patterns':patterns,'pass':bool(matches),'matching_chunk_ids':matches}


def measure_case(case, chunks, matrix, model):
    start=time.perf_counter(); vector=encode_user_question(model,case['query']); encoded=time.perf_counter()-start
    start=time.perf_counter(); scores=calculate_chunk_similarities(vector,matrix); scored=time.perf_counter()-start
    start=time.perf_counter(); ranked=retrieve_top_k_chunks(chunks,scores,len(chunks)); ranking=time.perf_counter()-start
    expected=case['expected_documents']
    ranks={d:next(c['rank'] for c in ranked if c['document_id']==d) for d in expected}
    first=min(ranks.values()) if ranks else None
    return {**case,'top_k':5,'results':ranked[:8], 'first_expected_document_rank':first,
        'expected_document_ranks':ranks,
        **{f'expected_document_hit_at_{k}': first is not None and first<=k for k in [1,3,5]},
        'source_coverage':{str(k):len({c['document_id'] for c in ranked[:k]} & set(expected))/len(expected) if expected else None for k in [3,5,8]},
        'evidence_check':{str(k):evidence_check(ranked[:k],expected,case.get('evidence_patterns',[])) for k in [3,5,8]},
        'score_distribution':{'highest':max(scores),'lowest':min(scores),'average':sum(scores)/len(scores)},
        'performance_seconds':{'question_encoding':encoded,'cosine':scored,'ranking_all_chunks':ranking},
        'first_expected_chunk':next((c for c in ranked if c['document_id'] in expected),None)}


def summarize(rows):
    applicable=[row for row in rows if row['single_source_metric']]
    return {'count':len(applicable),'query_ids':[r['query_id'] for r in applicable],
            **{f'document_hit_at_{k}':sum(r[f'expected_document_hit_at_{k}'] for r in applicable)/len(applicable) for k in [1,3,5]},
            'mrr':sum(1/r['first_expected_document_rank'] for r in applicable)/len(applicable)}


def main():
    start=time.perf_counter();documents=load_canonical_documents(ROOT);chunks=build_chunk_corpus(documents);processing=time.perf_counter()-start
    start=time.perf_counter();model=load_sentence_embedding_model();loaded=time.perf_counter()-start
    lengths=model_token_lengths(model.tokenizer,[c['text'] for c in chunks])
    if any(n>model.max_seq_length for n in lengths):
        raise ValueError('Chunks exceed the shared encoder window; inspect sources/settings before encoding.')
    start=time.perf_counter();matrix=encode_document_chunks(model,chunks);encoding=time.perf_counter()-start
    rows=[measure_case(case,chunks,matrix,model) for case in evaluation_cases()]
    history=json.loads((ROOT/'workshop/retrieval_comparison_results.json').read_text(encoding='utf-8'))
    comparisons=[]
    for identity in ['RET-02','RET-06','RET-09','RET-11','RET-12','RET-17']:
        old=next(b for b in history['benchmarks'] if b['benchmark_id']==identity)
        current=next(b for b in rows if b['query_id']==identity)
        comparisons.append({'query_id':identity,'query':current['query'],
            'whole_document_tfidf_ranks':old['tfidf']['ranks'],'whole_document_semantic_ranks':old['semantic']['ranks'],
            'chunk_expected_document_ranks':current['expected_document_ranks'],
            'first_expected_chunk':current['first_expected_chunk'], 'evidence_check_at_5':current['evidence_check']['5'],
            'caveat':'Historical whole-document corpus had 18 documents including papers; current corpus has 14 academic documents / 152 chunks. This is not a controlled truncation-only experiment.'})
    overlap_pairs=[]
    for row in rows:
        for i,a in enumerate(row['results'][:5]):
            for b in row['results'][i+1:5]:
                shared=min(a['word_end'],b['word_end'])-max(a['word_start'],b['word_start'])
                if a['document_id']==b['document_id'] and a['page_number']==b['page_number'] and shared>0:
                    overlap_pairs.append({'query_id':row['query_id'],'chunk_ids':[a['chunk_id'],b['chunk_id']],'ranks':[a['rank'],b['rank']],'shared_word_positions':shared})
    report={'usage':'Measured instructor/testing output only; application search must never use expected evidence.',
        'model_id':SENTENCE_MODEL_ID,'model_revision':SENTENCE_MODEL_REVISION,'python':platform.python_version(),
        'corpus':'14 canonical academic Markdown documents; no PDFs or papers',
        'settings':{'chunk_size':100,'overlap':20,'top_k':5}, 'document_count':len(documents),
        'chunk_count':len(chunks),'matrix_shape':list(matrix.shape),'token_max':max(lengths),'tokens_above_256':sum(n>256 for n in lengths),
        'source_sha256':{Path(d['source_path']).relative_to(ROOT).as_posix():hashlib.sha256(Path(d['source_path']).read_bytes()).hexdigest() for d in documents},
        'benchmark_sha256':hashlib.sha256((ROOT/'data/retrieval_benchmarks.json').read_bytes()).hexdigest(),
        'performance_seconds':{'processing':processing,'model_load':loaded,'chunk_encoding':encoding},
        'single_source_metrics':summarize(rows),'original_single_source_metrics':summarize([r for r in rows if r['origin']=='unchanged Task03 benchmark']),
        'metric_scope':'Document hits, not answer accuracy. Strict expected source IDs; related duplicate facts elsewhere do not count. Multi-source/comparison/summary/unsupported excluded from single-source aggregates. Evidence checks require all curated patterns in one expected-source chunk.',
        'excluded_original_queries':['RET-14','RET-18'],'excluded_reason':'Question papers are not in this corpus; no question analysis implemented.',
        'queries':rows,'whole_vs_chunk':comparisons,'overlap_pairs':overlap_pairs}
    (ROOT/'workshop/chunk_retrieval_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['performance_seconds','single_source_metrics','original_single_source_metrics','overlap_pairs']},indent=2))
    for row in rows:
        print(row['query_id'],row['first_expected_document_rank'],row['results'][0]['document_id'],round(row['results'][0]['score'],5),'coverage5',row['source_coverage']['5'],'evidence5',row['evidence_check']['5'])


if __name__=='__main__':
    main()
