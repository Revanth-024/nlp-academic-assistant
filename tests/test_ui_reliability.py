"""UI integration checks for stale answers and expensive-resource cache reuse."""
import csv
from io import StringIO
from pathlib import Path
from unittest.mock import patch
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
from src import embeddings,retrieval
from src.local_llm import RecordingTestLLM
ROOT=Path(__file__).resolve().parents[1]


def page(name):
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=120)
    app.switch_page('pages/'+name+'.py').run(timeout=120)
    assert not app.exception,app.exception
    return app

@pytest.mark.parametrize('change',['question','mode','top_k','generation'])
def test_assistant_clears_stale_history(change):
    app=page('10_Academic_Assistant')
    app.text_input[0].set_value('Show Transformer questions').run()
    app.button[0].click().run()
    assert app.session_state['assistant_result']['data']['count']==8
    if change=='question':app.text_input[0].set_value(' ').run()
    elif change=='mode':app.radio[0].set_value('Study Support').run()
    elif change=='generation':app.checkbox[0].check().run()
    else:app.select_slider[0].set_value(3).run()
    assert 'assistant_result' not in app.session_state and not app.exception


def test_word2vec_query_reuses_models_settings_retrain():
    st.cache_resource.clear()
    with patch('src.embeddings.train_word2vec',wraps=embeddings.train_word2vec) as trained:
        app=page('04_Word_Embeddings_Lab');initial=trained.call_count
        assert initial>=1
        selected=app.selectbox(key='w2v_word')
        selected.set_value(selected.options[1]).run(timeout=120)
        assert trained.call_count==initial
        app.selectbox(key='w2v_vector_size').set_value(20).run(timeout=120)
        assert not app.exception and trained.call_count>initial


def test_semantic_query_cache_and_corpus_invalidation():
    st.cache_data.clear();st.cache_resource.clear()
    target=(ROOT/'data/search_corpus/course_descriptions.csv').resolve()
    original=Path.read_text
    with patch('src.retrieval.encode_corpus_documents',wraps=retrieval.encode_corpus_documents) as encoded:
        app=page('06_Semantic_Search_Lab');initial=encoded.call_count
        app.text_area(key='semantic_query').set_value('human communication').run(timeout=120)
        assert encoded.call_count==initial
        rows=list(csv.DictReader(StringIO(original(target,encoding='utf-8'))))
        rows[0]['description']+=' A temporary cache-invalidation test.'
        buffer=StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        def read(path,*args,**kwargs):return buffer.getvalue() if path.resolve()==target else original(path,*args,**kwargs)
        with patch.object(Path,'read_text',read):app.run(timeout=120)
        assert not app.exception and encoded.call_count==initial+1


def test_sentence_pair_similarity_reused_for_other_widget():
    st.cache_data.clear();st.cache_resource.clear()
    with patch('src.embeddings.sentence_similarity',wraps=embeddings.sentence_similarity) as similarity:
        app=page('05_Sentence_Embeddings_Lab');initial=similarity.call_count
        app.text_area(key='sentence_inspect').set_value('A different inspection sentence.').run(timeout=120)
        assert not app.exception and similarity.call_count==initial
        app.text_area(key='sentence_a').set_value('Students study natural language.').run(timeout=120)
        assert not app.exception and similarity.call_count==initial+1


def test_rag_stale_generation_and_chunk_cache():
    st.cache_data.clear();st.cache_resource.clear()
    with patch('src.retrieval.encode_document_chunks',wraps=retrieval.encode_document_chunks) as encoded, \
         patch('src.llm_resources.load_local_llm',return_value=RecordingTestLLM('TEST ONLY')):
        app=page('08_RAG_Lab');initial=encoded.call_count
        app.radio(key='rag_mode').set_value('Full Local Generation').run(timeout=120)
        app.button(key='rag_run').click().run(timeout=120)
        assert app.session_state['rag_result']['generation']['answer']=='TEST ONLY'
        app.select_slider(key='rag_k').set_value(3).run(timeout=120)
        assert app.session_state['rag_result']['generation']['answer'] is None and encoded.call_count==initial
        app.number_input(key='rag_context_chars').set_value(500).run(timeout=120)
        assert app.session_state['rag_result']['context']['excluded_sources'] and encoded.call_count==initial
        app.text_input[0].set_value(' ').run(timeout=120)
        assert 'rag_result' not in app.session_state
        app.text_input[0].set_value('attendance').run(timeout=120)
        app.radio(key='rag_corpus').set_value('Selected document / PDF experiment').run(timeout=120)
        changed=encoded.call_count
        assert changed>initial
        app.select_slider(key='rag_size').set_value(80).run(timeout=120)
        assert not app.exception and encoded.call_count>changed
        changed=encoded.call_count
        app.select_slider(key='rag_overlap').set_value(10).run(timeout=120)
        assert not app.exception and encoded.call_count>changed
