from pathlib import Path
import copy
import hashlib
import json
from unittest.mock import patch
import pytest
from src.document_processor import process_document
from src.rag import (run_transparent_rag,inspect_source_labels,supplied_source_attribution)
from src.local_llm import RecordingTestLLM
ROOT=Path(__file__).resolve().parents[1]


def tiny_retriever(question,k):
    doc={'document_id':'D','title':'Policy','source_type':'markdown','source_path':'policy.md',
         'pages':[{'page_number':None,'text':'Students need 75% attendance to attend the final exam.'}]}
    chunks=process_document(doc,5,2)['chunks']
    results=[{**c,'rank':i,'score':.8-i*.1} for i,c in enumerate(chunks[:k],1)]
    return {'question_vector':[.1]*384,'scores':[.7]*len(chunks),'sorted_indices':list(range(len(chunks))),'results':results}


@pytest.mark.parametrize('answer,mentioned,valid,invalid,status',[
 ('75% [SOURCE 1]',['SOURCE 1'],['SOURCE 1'],[],'valid_labels'),
 ('Claim [SOURCE 99]',['SOURCE 99'],[],['SOURCE 99'],'unknown_labels'),
 ('[SOURCE 2] [SOURCE 1] [SOURCE 2]',['SOURCE 2','SOURCE 1'],['SOURCE 2','SOURCE 1'],[],'valid_labels'),
 ('No labels',[],[],[],'no_labels'),
 (None,[],[],[],'no_labels'),
 ('[source 01] [SOURCE 0]',['SOURCE 1','SOURCE 0'],['SOURCE 1'],['SOURCE 0'],'unknown_labels'),
])
def test_structural_label_inspector(answer,mentioned,valid,invalid,status):
    value=inspect_source_labels(answer,['SOURCE 1','SOURCE 2'])
    assert value['mentioned_labels']==mentioned and value['valid_labels']==valid
    assert value['invalid_labels']==invalid and value['status']==status
    assert 'not verified' in value['scope']
    if answer and answer.count('[SOURCE 2]')==2: assert value['occurrence_count']==3


def test_no_generation_complete_result_and_attribution():
    result=run_transparent_rag('attendance?',tiny_retriever,top_k=3)
    assert result['generation']['status']=='unavailable' and result['generation']['error'] is None
    assert result['generation']['answer'] is None and not result['generation']['available']
    assert len(result['retrieval']['query_embedding'])==384
    assert result['prompt'] and result['context']['text'] and result['diagnostics']
    expected=result['context']['included_sources']
    actual=result['attribution']['supplied_sources']
    assert actual==[{'label':label,**record} for label,record in expected.items()]
    assert [c['chunk_id'] for c in actual]==[c['chunk_id'] for c in result['retrieval']['results']]
    assert all(c['page_number'] is None for c in actual)
    assert result['diagnostics']['overlap_pairs']
    assert all(v>=0 for v in result['timings_seconds'].values())


def test_test_backend_exact_prompt_and_independent_source_display():
    backend=RecordingTestLLM('Known answer with no labels.')
    result=run_transparent_rag('attendance?',tiny_retriever,backend,top_k=3)
    assert backend.prompts==[result['prompt']] and result['generation']['answer']=='Known answer with no labels.'
    assert result['attribution']['supplied_sources']
    assert result['attribution']['model_label_inspection']['status']=='no_labels'
    assert any('did not emit' in flag for flag in result['diagnostics']['flags'])
    other=run_transparent_rag('attendance?',tiny_retriever,RecordingTestLLM('Wrong [SOURCE 99]'),top_k=3)
    assert other['diagnostics']['model_labels']['invalid_labels']==['SOURCE 99']
    assert other['attribution']['supplied_sources']==result['attribution']['supplied_sources']


def test_budget_exclusions_and_error_preserve_all_stages():
    result=run_transparent_rag('attendance?',tiny_retriever,top_k=3,max_chunks=1)
    assert len(result['attribution']['supplied_sources'])==1
    assert result['context']['excluded_sources']
    assert any('budget' in flag for flag in result['diagnostics']['flags'])
    class Broken:
        def generate(self,prompt): raise RuntimeError('test failure')
    error=run_transparent_rag('attendance?',tiny_retriever,Broken(),top_k=3)
    assert error['generation']['status']=='error' and 'test failure' in error['generation']['error']
    assert error['context']['text'] and error['prompt'] and error['attribution']['supplied_sources']


def test_multiple_documents_and_pdf_page_preserved():
    base=tiny_retriever('x',3)
    base['results'][1].update(document_id='P',source_type='pdf',page_number=2,chunk_id='P::chunk_000',source_path='p.pdf')
    original=copy.deepcopy(base)
    result=run_transparent_rag('question',lambda q,k:base,top_k=3)
    assert result['context']['statistics']['distinct_documents']==2
    assert result['attribution']['supplied_sources'][1]['page_number']==2
    assert base==original
    assert not any('one document' in flag for flag in result['diagnostics']['flags'])


@pytest.mark.parametrize('k',[-1,True,1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError):run_transparent_rag('q',tiny_retriever,top_k=k)


def test_real_attendance_k_evidence_without_generation():
    from src.retrieval import load_canonical_documents,build_chunk_corpus,encode_document_chunks,semantic_chunk_search
    from src.sentence_resources import load_sentence_embedding_model
    model=load_sentence_embedding_model();chunks=build_chunk_corpus(load_canonical_documents(ROOT));matrix=encode_document_chunks(model,chunks)
    retriever=lambda q,k:semantic_chunk_search(q,chunks,matrix,model,k)
    one=run_transparent_rag('What is the minimum attendance required?',retriever,top_k=1)
    five=run_transparent_rag('What is the minimum attendance required?',retriever,top_k=5)
    assert '75%' not in one['context']['text'] and '75%' in five['context']['text']
    tight=run_transparent_rag('What is the minimum attendance required?',retriever,top_k=5,max_chunks=1)
    assert any('75%' in c['text'] for c in tight['retrieval']['results'])
    assert '75%' not in tight['context']['text']
    assert one['context']['text']!=five['context']['text']


def test_evaluation_metadata_and_separation():
    report=json.loads((ROOT/'workshop/rag_evaluation_results.json').read_text(encoding='utf-8'))
    assert len(report['cases'])==12 and report['chunk_settings']['count']==152
    assert report['chunk_settings']['above_256']==0 and report['default_top_k']==5
    assert len({c['case_id'] for c in report['cases']})==12
    for path,digest in report['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((ROOT/'data/university_facts.json').read_bytes()).hexdigest()==report['expected_fact_contract_sha256']
    for case in report['cases']:
        included=case['included_sources']
        assert len(included)==case['context_statistics']['included_chunks']
        assert case['label_validity']==inspect_source_labels(case['answer'],[c['label'] for c in included])
    code=(ROOT/'src/rag.py').read_text(encoding='utf-8-sig')
    assert 'university_facts.json' not in code and 'rag_evaluation_results.json' not in code
