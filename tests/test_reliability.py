"""Task17 integration invariants; no model-quality or benchmark-win assertions."""
import ast
from collections import Counter
import json
from pathlib import Path
import re
from unittest.mock import patch
import pytest
from src.academic_assistant import inspect_unit_evidence, run_academic_assistant
from src.local_llm import RecordingTestLLM
from src.retrieval import load_canonical_documents, build_chunk_corpus, encode_document_chunks, semantic_chunk_search
from src.sentence_resources import load_sentence_embedding_model
from workshop.preflight import check_preflight
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6])
def test_unit_guard_general_and_no_score_manipulation(n):
    chunks=[{'document_title':'Natural Language Processing Syllabus','text':f'### Unit {n} — Topic\nEvidence text','score':.2},
            {'document_title':'Machine Learning Syllabus','text':'### Unit 4 — Other course','score':.9}]
    check=inspect_unit_evidence(f'Summarize NLP Unit {n}',chunks)
    assert check['generation_allowed'] and check['requested_units']==[n]
    assert [c['score'] for c in chunks]==[.2,.9]
    assert not inspect_unit_evidence(f'NLP Unit {n}',chunks[1:])['generation_allowed']

@pytest.mark.parametrize('question',['Semester 6 courses','What are HCE-CSE603 credits?','NLP prerequisites'])
def test_other_identifiers_do_not_trigger_unit_policy(question):
    assert not inspect_unit_evidence(question,[])['applied']


def test_guard_only_supplied_evidence_and_ambiguous_scope():
    chunks=[{'document_title':'Natural Language Processing Syllabus','text':'Unit 4 is mentioned, but is not a unit heading.'}]
    assert not inspect_unit_evidence('NLP Unit 4',chunks)['generation_allowed']
    chunks[0]['text']='### Unit 4 — Correct header'
    assert not inspect_unit_evidence('Summarize Unit 4',chunks)['generation_allowed']
    chunks.append({'document_title':'Machine Learning Syllabus','text':'### Unit 4 — Other header'})
    assert not inspect_unit_evidence('Compare ML and NLP Unit 4',chunks)['generation_allowed']

@pytest.fixture(scope='module')
def real_retriever():
    model=load_sentence_embedding_model();chunks=build_chunk_corpus(load_canonical_documents(ROOT))
    matrix=encode_document_chunks(model,chunks)
    return lambda q,k: semantic_chunk_search(q,chunks,matrix,model,k)

@pytest.mark.parametrize('question',['NLP Unit 1','NLP Unit 4','NLP Unit 5','What topics are covered in NLP Unit 4?'])
def test_real_unit_context_diagnostic(real_retriever,question):
    calls=[]
    def loader(): calls.append(True);return RecordingTestLLM('TEST ONLY')
    baseline=real_retriever(question,5)
    result=run_academic_assistant(question,retriever=real_retriever,llm_loader=loader,generation_enabled=True)
    rag=result['data']['rag'];guard=result['diagnostics']['unit_evidence']
    assert [(c['chunk_id'],c['score']) for c in rag['retrieval']['results']]==[(c['chunk_id'],c['score']) for c in baseline['results']]
    assert result['diagnostics']['original_query']==result['diagnostics']['retrieval_query']==question
    assert bool(calls)==guard['generation_allowed']
    if not guard['generation_allowed']:
        assert result['answer'] is None and any('Generation withheld' in w for w in result['warnings'])


def test_short_query_and_full_paraphrase_remain_visible(real_retriever):
    for question in ['NLP prerequisites','Which courses must a student complete before taking Natural Language Processing?']:
        result=run_academic_assistant(question,retriever=real_retriever)
        assert result['diagnostics']['retrieval_query']==question
        assert result['data']['rag']['retrieval']['results']
    short=run_academic_assistant('NLP prerequisites',retriever=real_retriever)
    assert any('Short queries' in w for w in short['warnings'])


def test_guard_prevents_loading_backend_without_unit_evidence():
    from tests.test_transparent_rag import tiny_retriever
    def forbidden(): raise AssertionError('Wrong-unit request loaded Qwen')
    result=run_academic_assistant('Summarize NLP Unit 4',retriever=tiny_retriever,
                                  llm_loader=forbidden,generation_enabled=True)
    assert result['answer_type']=='retrieval_only' and not result['diagnostics']['unit_evidence']['generation_allowed']


def test_preflight_current_and_optional_qwen():
    report=check_preflight(ROOT,True)
    assert report['passed']
    assert next(c for c in report['checks'] if c['check']=='Qwen')['status']=='OPTIONAL'


def test_preflight_missing_caches_and_data(tmp_path):
    inventory=check_preflight(tmp_path)
    assert not inventory['passed']
    status={c['check']:c['status'] for c in inventory['checks']}
    assert status['Question papers']=='FAIL' and status['MiniLM cache']=='WARNING' and status['Qwen']=='OPTIONAL'
    strict=check_preflight(tmp_path,True)
    assert next(c for c in strict['checks'] if c['check']=='MiniLM cache')['status']=='FAIL'


def test_preflight_missing_nltk_and_dependency():
    with patch('src.nltk_resources.resource_available',return_value=False):
        assert not check_preflight(ROOT)['passed']
    import importlib.metadata
    real=importlib.metadata.version
    def version(name):
        if name=='gensim': raise importlib.metadata.PackageNotFoundError(name)
        return real(name)
    with patch('importlib.metadata.version',side_effect=version):
        assert not check_preflight(ROOT)['passed']


def test_all_49_ids_and_mechanical_regions_synchronized():
    boundaries=[];parts=[]
    for file in (ROOT/'src').glob('*.py'):
        text=file.read_text(encoding='utf-8-sig')
        boundaries+=re.findall(r'STUDENT TODO BOUNDARY (\d+\.\d+)',text)
        begins=re.findall(r'BEGIN STUDENT CORE ([^\r\n]+)',text)
        ends=re.findall(r'END STUDENT CORE ([^\r\n]+)',text)
        assert begins==ends,(file.name,begins,ends)
        assert len(begins)==len(set(begins)),file.name
        parts+=re.findall(r'BEGIN STUDENT CORE (\d+\.\d+)',text)
    assert len(boundaries)==len(set(boundaries))==49
    assert set(parts)==set(boundaries)
    assert Counter(parts)['1.7']==2  # Two labelled subregions, one exercise ID.
    index=(ROOT/'workshop/TODO_INDEX.md').read_text(encoding='utf-8')
    indexed=re.findall(r'^\|\s*(\d+\.\d+)\s*\|',index,re.M)
    assert len(indexed)==49 and set(indexed)==set(boundaries)
    for file in ['docs/STUDENT_GUIDE.md','docs/INSTRUCTOR_GUIDE.md']:
        text=(ROOT/file).read_text(encoding='utf-8')
        assert all(re.search(r'(?<!\d)'+re.escape(i)+r'(?!\d)',text) for i in boundaries),file


def test_source_boundary_and_notebook_algorithm_reuse():
    public=set()
    for file in (ROOT/'src').glob('*.py'):
        tree=ast.parse(file.read_text(encoding='utf-8-sig'),feature_version=(3,11))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): assert all(not n.name.startswith('streamlit') for n in node.names)
            if isinstance(node,ast.ImportFrom): assert not (node.module or '').startswith('streamlit')
            if isinstance(node,ast.FunctionDef) and not node.name.startswith('_'):public.add(node.name)
    for file in (ROOT/'notebooks').glob('*.ipynb'):
        notebook=json.loads(file.read_text(encoding='utf-8'))
        code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
        assert 'from src.' in code,file.name
        assert not any(isinstance(n,ast.FunctionDef) and n.name in public for n in ast.walk(ast.parse(code))),file.name
