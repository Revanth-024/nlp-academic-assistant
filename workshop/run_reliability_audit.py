"""Offline Task17 integration audit; no Qwen or downloads by default.

Run python -m workshop.run_reliability_audit --pages.
Optional --with-generation exercises cached Qwen separately. Expected evidence
and prior evaluation artifacts are audit-only, never production answers.
"""
import argparse
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import re
import time
from unittest.mock import patch
from workshop.preflight import check_preflight
ROOT=Path(__file__).resolve().parents[1]


def run_audit(pages=False,with_generation=False):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    report={'environment':'Local CPU / network calls rejected', 'python_version':platform.python_version(),
            'utc':datetime.now(timezone.utc).isoformat(), 'test_summary':{'scope':'Focused audit checks; full pytest recorded separately.'},
            'resource_summary':check_preflight(ROOT,True),'offline_summary':{},'page_render_summary':[],
            'known_issue_summary':{},'retrieval_reliability':{},'cache_summary':{},'todo_consistency':{},
            'data_consistency':{},'performance_summary':{},'warnings':[],'matrix':[]}
    def check(area,scenario,operation,expected_error=False):
        start=time.perf_counter()
        try:
            value=operation()
            if expected_error: raise AssertionError('Expected validation error did not occur.')
            status,detail='PASS',str(value)[:160]
        except (ValueError,TypeError,RuntimeError,OSError) as error:
            status='PASS' if expected_error else 'FAIL';detail=str(error)[:240]
        except Exception as error: status,detail='FAIL',f'{type(error).__name__}: {error}'[:240]
        elapsed=time.perf_counter()-start
        report['matrix'].append({'area':area,'scenario':scenario,'status':status,'seconds':elapsed,'detail':detail})
        if scenario=='warm': report['performance_summary'][area]=elapsed
        return status
    with patch('requests.sessions.Session.request',side_effect=AssertionError('Audit forbids network')),patch('nltk.download',side_effect=AssertionError('Audit forbids downloads')):
        from src.preprocessing import lowercase_text,tokenize_words,remove_punctuation,remove_stopwords,lemmatize_words
        from src.classical_nlp import build_vocabulary,bag_of_words,generate_ngrams
        from src.embeddings import prepare_tokenized_sentences,train_word2vec,get_word_vector,encode_sentences,sentence_similarity
        from src.retrieval import (prepare_search_corpus,search_tfidf,load_canonical_documents,build_chunk_corpus,
                                  encode_document_chunks,semantic_chunk_search,encode_corpus_documents,semantic_search)
        from src.document_processor import extract_text_from_pdf,process_document,MAX_PDF_BYTES,model_token_lengths
        from src.sentence_resources import load_sentence_embedding_model
        from src.transformer_resources import load_transformer_pipeline
        from src.transformers_nlp import analyze_sentiment,recognize_entities,answer_from_context
        from src.llm_resources import load_local_llm
        from src.academic_assistant import load_analyzer_resources,run_academic_assistant,inspect_unit_evidence,route_question
        from src.rag import run_transparent_rag,select_context_chunks
        from pypdf import PdfWriter
        tokens=['students','study','language','students'];vocabulary=build_vocabulary([tokens])
        preprocess=lambda:lemmatize_words(remove_stopwords(tokenize_words(remove_punctuation(lowercase_text('Students are learning languages!')))))
        for scenario,op,error in [('normal',preprocess,False),('empty',lambda:tokenize_words(' '),False),('invalid',lambda:tokenize_words(None),True),('warm',preprocess,False)]:check('Preprocessing',scenario,op,error)
        for scenario,op,error in [('normal',lambda:bag_of_words(tokens,vocabulary),False),('empty',lambda:build_vocabulary([]),False),('invalid',lambda:generate_ngrams(tokens,0),True),('warm',lambda:bag_of_words(tokens,vocabulary),False)]:check('BoW',scenario,op,error)
        corpus=prepare_search_corpus([{'document_id':'A','text':'natural language processing'}, {'document_id':'B','text':'database records'}])
        for scenario,q in [('normal','language'),('empty',''),('OOV','zznotincorpus'),('warm','language')]:check('TF-IDF',scenario,lambda q=q:search_tfidf(q,corpus,50)['results'])
        check('TF-IDF','invalid',lambda:search_tfidf('language',corpus,-1),True)
        data=json.loads((ROOT/'data/examples/word2vec_corpus.json').read_text(encoding='utf-8'))
        prepared=prepare_tokenized_sentences([s['text'] for s in data['sentences']]);w2v=train_word2vec(prepared)
        for scenario,op,error in [('normal',lambda:get_word_vector(w2v,'language')[:4],False),('empty',lambda:prepare_tokenized_sentences([]),False),('invalid',lambda:train_word2vec([]),True),('missing vocabulary',lambda:get_word_vector(w2v,'zzunknown'),True),('warm',lambda:get_word_vector(w2v,'language')[:4],False)]:check('Word2Vec',scenario,op,error)
        init=time.perf_counter();model=load_sentence_embedding_model();documents=load_canonical_documents(ROOT);chunks=build_chunk_corpus(documents);matrix=encode_document_chunks(model,chunks)
        report['performance_summary']['encoder_and_chunk_initialization']=time.perf_counter()-init
        report['cache_summary']['MiniLM_same_object']=load_sentence_embedding_model() is model
        for scenario,op,error in [('normal',lambda:encode_sentences(model,['Students study NLP.']).shape,False),('empty',lambda:encode_sentences(model,[]).shape,False),('invalid',lambda:encode_sentences(model,[' ']),True),('missing resource',lambda:load_sentence_embedding_model(ROOT/'.cache/nonexistent-audit-resource'),True),('warm',lambda:sentence_similarity(model,'Students study language.','Learners study language.'),False)]:check('Sentence Embeddings',scenario,op,error)
        long='language '*1000
        check('Sentence Embeddings','long input',lambda:{'model_tokens':model_token_lengths(model.tokenizer,[long])[0], 'vector_shape':list(encode_sentences(model,[long]).shape),'truncation':'Explicit UI warning; encoder retains original256-token behavior.'})
        descriptions=[{'document_id':'NLP','text':'Computational methods for human language.'},{'document_id':'NET','text':'Devices exchange packets.'}]
        course_matrix=encode_corpus_documents(model,descriptions)
        for scenario,q in [('normal','human language'),('warm','human language')]:check('Semantic Search',scenario,lambda q=q:semantic_search(q,descriptions,course_matrix,model,50)['results'])
        check('Semantic Search','empty',lambda:semantic_search('',descriptions,course_matrix,model),True)
        check('Semantic Search','invalid',lambda:semantic_search('x',descriptions,course_matrix,model,-1),True)
        for task in ['sentiment','ner','qa']:
            pipeline=load_transformer_pipeline(task)
            report['cache_summary']['Transformer_'+task+'_same_object']=load_transformer_pipeline(task) is pipeline
            operation=(lambda:analyze_sentiment(pipeline,'The workshop is useful.')) if task=='sentiment' else (lambda:recognize_entities(pipeline,'Riya studies in London.')) if task=='ner' else (lambda:answer_from_context(pipeline,'Which semester?','NLP is offered in Semester6.'))
            check('Transformers '+task,'normal',operation);check('Transformers '+task,'warm',operation)
            check('Transformers '+task,'empty',lambda:analyze_sentiment(pipeline,''),True)
        check('Transformers','missing resource',lambda:load_transformer_pipeline('sentiment',ROOT/'.cache/nonexistent-audit-resource'),True)
        check('Transformers','invalid',lambda:load_transformer_pipeline('unknown'),True)
        resources=load_analyzer_resources(ROOT)
        for scenario,q in [('normal','Show Transformer questions'),('warm','Most frequent topics in previous papers'),('empty',''),('unknown topic','Show quantum questions')]:check('Analyzer',scenario,lambda q=q:run_academic_assistant(q,analyzer_loader=lambda:resources)['answer_type'])
        check('Analyzer','missing data',lambda:load_analyzer_resources(ROOT/'.cache/nonexistent-audit-resource'),True)
        retrieve=lambda q,k:semantic_chunk_search(q,chunks,matrix,model,k)
        check('Chunk Retrieval','warm',lambda:retrieve('What attendance is required?',5)['results'][0]['chunk_id'])
        check('Chunk Retrieval','empty',lambda:retrieve('',5),True)
        check('Chunk Retrieval','large Top-K',lambda:len(retrieve('attendance',200)['results']))
        for scenario,q in [('normal','What attendance is required?'),('warm','What attendance is required?')]:check('RAG',scenario,lambda q=q:run_transparent_rag(q,retrieve)['generation']['status'])
        check('RAG','empty',lambda:run_transparent_rag('',retrieve),True)
        check('RAG','invalid',lambda:run_transparent_rag('attendance',retrieve,top_k=-1),True)
        for scenario,q in [('normal','attendance requirement'),('warm','attendance requirement'),('empty',''),('unknown topic','Show quantum questions')]:check('Assistant',scenario,lambda q=q:run_academic_assistant(q,retriever=retrieve,analyzer_loader=lambda:resources)['answer_type'])
        check('Assistant','invalid mode',lambda:run_academic_assistant('attendance','bad'),True)
        check('Assistant','missing MiniLM',lambda:run_academic_assistant('Show Transformer questions',analyzer_loader=lambda:resources)['data']['count'])
        check('RAG','missing Qwen',lambda:run_academic_assistant('attendance',retriever=retrieve,generation_enabled=True,llm_loader=lambda:load_local_llm(ROOT/'.cache/nonexistent-audit-resource'))['answer_type'])
        # PDF upload checks use in-memory bytes, never a filesystem upload.
        def pdf_bytes(writer):
            buffer=BytesIO();writer.write(buffer);return buffer.getvalue()
        writer=PdfWriter();writer.add_blank_page(width=100,height=100)
        check('PDF','malformed',lambda:extract_text_from_pdf(b'invalid'),True)
        check('PDF','scanned/blank',lambda:extract_text_from_pdf(pdf_bytes(writer)),True)
        check('PDF','oversized',lambda:extract_text_from_pdf(b'0'*(MAX_PDF_BYTES+1)),True)
        writer.encrypt('secret');check('PDF','encrypted',lambda:extract_text_from_pdf(pdf_bytes(writer)),True)
        writer=PdfWriter()
        for _ in range(51):writer.add_blank_page(width=100,height=100)
        check('PDF','over50pages',lambda:extract_text_from_pdf(pdf_bytes(writer)),True)
        manifest=json.loads((ROOT/'data/document_manifest.json').read_text(encoding='utf-8'));pdfs=[]
        for entry in manifest['derived_sample_pdfs']:
            extracted=extract_text_from_pdf(ROOT/'data'/entry['filename'])
            text='\n'.join(p['text'] for p in extracted['pages'])
            pdfs.append({'filename':entry['filename'],'pages':len(extracted['pages']),
                'disclaimer': 'Synthetic educational data created for NLP workshop purposes.' in text,
                'fictional':'fictional' in text.lower(),
                'source_hash_matches':hashlib.sha256((ROOT/'data'/entry['canonical_filename']).read_bytes()).hexdigest()==entry['canonical_source_sha256']})
        report['data_consistency']['derived_PDFs']=pdfs
        report['data_consistency']['canonical_document_disclaimers']=all('Synthetic educational data created for NLP workshop purposes.' in p.read_text(encoding='utf-8') for p in list((ROOT/'data/academic_documents').glob('*.md'))+list((ROOT/'data/question_papers').glob('*.md')))
        # Full rankings expose structural identifier misses, not only successes.
        queries=['NLP prerequisites','What are the prerequisites for NLP?','Which courses must a student complete before taking Natural Language Processing?']+[f'NLP Unit {n}' for n in range(1,6)]+['Summarize NLP Unit 4.','What topics are covered in NLP Unit 4?','Create revision questions for NLP Unit 4.']
        issue_rows=[]
        for query in queries:
            all_results=retrieve(query,len(chunks))['results'];unit=re.search(r'\bUnit\s+(\d+)\b',query,re.I)
            expected=[]
            for c in all_results:
                if unit: hit=c['document_id']=='HCE-DOC04' and bool(re.search(r'(?mi)^\s*#{1,6}\s+Unit\s+'+unit.group(1)+r'\b',c['text']))
                else: hit=c['document_id'] in {'HCE-DOC04','HCE-DOC13'} and 'Artificial Intelligence' in c['text'] and 'Machine Learning' in c['text']
                if hit:expected.append({'rank':c['rank'],'chunk_id':c['chunk_id'],'score':c['score']})
            supplied,_=select_context_chunks(all_results[:5]);guard=inspect_unit_evidence(query,supplied)
            issue_rows.append({'query':query,'expected_evidence':expected,'top5':[{k:c[k] for k in ['rank','chunk_id','document_id','score','text']} for c in all_results[:5]],'final_guard':guard,'ranking_changed':False})
        report['retrieval_reliability']['investigation']=issue_rows
        report['known_issue_summary']={'short_queries':'Expected NLP prerequisites rank8/10 versus2 with full course-name question; wording/underspecification changes embedding geometry. No query expansion added.',
            'unit_queries':'Exact unit identifiers weakly represented by MiniLM; generic/neighboring unit text ranks higher. Requested heading exists in chunks belowK5;100-word windows can split heading/body.',
            'policy':'Assistant-only generation guard based on matching syllabus course scope and exact supplied Unit heading. No ranking or score change; no fabricated unit text. Missing/ambiguous evidence keeps retrieval visible and withholds Qwen.'}
        # Re-run existing evaluator logic and compare raw ranks/context/prompt.
        from workshop.evaluate_chunk_retrieval import evaluation_cases,measure_case,summarize
        current=[measure_case(c,chunks,matrix,model) for c in evaluation_cases()]
        old=json.loads((ROOT/'workshop/chunk_retrieval_results.json').read_text(encoding='utf-8'))
        comparison=[]
        for row in current:
            prior=next(r for r in old['queries'] if r['query_id']==row['query_id'])
            comparison.append({'query_id':row['query_id'],'top5_unchanged':[c['chunk_id'] for c in row['results'][:5]]==[c['chunk_id'] for c in prior['results'][:5]],'expected_rank_unchanged':row['first_expected_document_rank']==prior['first_expected_document_rank']})
        report['retrieval_reliability']['Task12']={'metrics':summarize(current),'comparison':comparison}
        from workshop.evaluate_rag import evaluation_cases as rag_cases,evidence_state
        facts=json.loads((ROOT/'data/university_facts.json').read_text(encoding='utf-8'))
        old=json.loads((ROOT/'workshop/rag_evaluation_results.json').read_text(encoding='utf-8'));rag_rows=[]
        for case in rag_cases(facts):
            result=run_transparent_rag(case['question'],retrieve,top_k=case.get('top_k',5),max_chunks=case.get('max_chunks',5))
            prior=next(r for r in old['cases'] if r['case_id']==case['case_id'])
            rag_rows.append({'case_id':case['case_id'],'topK_unchanged':[c['chunk_id'] for c in result['retrieval']['results']]==[c['chunk_id'] for c in prior['retrieved_sources']],
                'context_chunks_unchanged':[c['chunk_id'] for c in result['attribution']['supplied_sources']]==[c['chunk_id'] for c in prior['included_sources']],
                'prompt_unchanged':hashlib.sha256(result['prompt'].encode()).hexdigest()==prior['prompt_sha256'],
                'context_evidence':evidence_state(list(result['context']['included_sources'].values()),case)})
        report['retrieval_reliability']['Task14']={'comparison':rag_rows,'generation_scope':'Historical failures retained; no prompt/model/RAG algorithm changes. Optional generation is separate.'}
        from workshop.evaluate_academic_assistant import evaluate
        assistant=evaluate();report['retrieval_reliability']['Task16']=assistant
        from workshop.evaluate_question_analyzer import evaluate as analyzer_evaluate
        analyzer_report=analyzer_evaluate();report['analyzer_regression']=analyzer_report['summary'] if 'summary' in analyzer_report else {k:v for k,v in analyzer_report.items() if k not in ['questions','per_topic_metrics']}
        boundaries=[]
        for file in (ROOT/'src').glob('*.py'):boundaries+=re.findall(r'STUDENT TODO BOUNDARY (\d+\.\d+)',file.read_text(encoding='utf-8-sig'))
        report['todo_consistency']={'IDs':sorted(boundaries,key=lambda x:tuple(map(int,x.split('.')))),'unique_count':len(set(boundaries)),'duplicate_IDs':len(boundaries)-len(set(boundaries)), 'core_regions':50,'note':'1.7 has two labelled subregions under one exercise ID; no implementations stripped.'}
        if with_generation:
            start=time.perf_counter();llm=load_local_llm();report['cache_summary']['Qwen_same_object']=load_local_llm() is llm
            report['performance_summary']['Qwen_load']=time.perf_counter()-start
            result=run_academic_assistant('What is the minimum attendance required?',retriever=retrieve,generation_enabled=True,llm_loader=lambda:llm)
            report['optional_generation']={'answer':result['answer'],'timings':result['diagnostics']['timings_seconds'],'note':'Actual optional local model output; not a correctness guarantee.'}
        if pages:
            from streamlit.testing.v1 import AppTest
            for file in [ROOT/'app.py',*sorted((ROOT/'pages').glob('*.py'))]:
                app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=120)
                if file.name!='app.py':app.switch_page('pages/'+file.name).run(timeout=120)
                report['page_render_summary'].append({'page':file.name,'status':'PASS' if not app.exception else 'FAIL','errors':[str(e.value) for e in app.exception]})
        report['offline_summary']={'requests_rejected':True,'downloads_rejected':True,'all_checks_local':True,'Qwen_requested':with_generation}
    report['test_summary'].update(passed=sum(c['status']=='PASS' for c in report['matrix']),failed=sum(c['status']=='FAIL' for c in report['matrix']))
    report['warnings'].extend(['No query expansion or structural reranking was added. Pure retrieval misses remain measured.', 'Unit heading presence is necessary but insufficient for complete coverage; mixed-unit requests are withheld.', 'Optional generation remains slow and factually unreliable. Historical source labels were absent; sources are not verified citations.'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pages',action='store_true');parser.add_argument('--with-generation',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'workshop/reliability_audit_results.json')
    args=parser.parse_args();report=run_audit(args.pages,args.with_generation)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'matrix':report['test_summary'],'pages':report['page_render_summary'],'output':str(args.output)},indent=2))
    return 1 if report['test_summary']['failed'] or any(p['status']=='FAIL' for p in report['page_render_summary']) else 0

if __name__=='__main__':raise SystemExit(main())
