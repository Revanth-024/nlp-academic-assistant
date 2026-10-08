"""Compact Task14 evaluator; expectations/checks never enter production RAG."""
import argparse
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import time
from src.rag import run_transparent_rag,build_grounded_prompt
from src.retrieval import load_canonical_documents,build_chunk_corpus,encode_document_chunks,semantic_chunk_search
from src.document_processor import model_token_lengths
from src.sentence_resources import load_sentence_embedding_model,SENTENCE_MODEL_ID,SENTENCE_MODEL_REVISION
from src.llm_resources import load_local_llm,LocalModelUnavailable,MODEL_ID,MODEL_REVISION
ROOT=Path(__file__).resolve().parents[1]
ABSTENTION_PHRASES=['insufficient','cannot determine','not provided','not available in context','does not contain','no specific recipe','not enough information','no recipe']


def evaluation_cases(facts):
    nlp=next(c for c in facts['courses'] if c['code']=='HCE-CSE603')
    attendance=str(facts['attendance_policy']['minimum_percent'])
    exam=facts['examination']['schemes']['course_standard']
    cases=[]
    for k in [1,3,5]:
        cases.append({'case_id':f'ATT-K{k}','category':'retrieval_failure' if k==1 else 'direct_fact',
            'question':'What is the minimum attendance required?','top_k':k,'preferred_documents':['HCE-DOC08'],
            'evidence_patterns':[attendance+'%',r'attendance'],'answer_patterns':[attendance],
            'known_incorrect_patterns':[r'\b80\s*%'],
            'human_note':'Read attendance evidence, not just preferred-source rank. At K=1 the percentage was absent in Task13; do not assume the new model output.'})
    cases += [
        {'case_id':'EXAM','category':'numeric_fact','question':'What is the minimum end-semester score required to pass?',
         'preferred_documents':['HCE-DOC09'],'evidence_patterns':[str(exam['minimum_total_to_pass']),str(exam['minimum_end_semester_to_pass']),r'end.semester'],
         'answer_patterns':[str(exam['minimum_total_to_pass']),str(exam['minimum_end_semester_to_pass'])],
         'human_note':'Task-specific check expects both pass conditions; inspect whether the answer distinguishes the end-semester minimum.'},
        {'case_id':'SEMESTER','category':'direct_fact','question':'In which semester is Natural Language Processing offered?',
         'preferred_documents':['HCE-DOC04'],'evidence_patterns':[r'HCE-CSE603',r'semester\s*(?:\||:)?\s*'+str(nlp['semester'])],
         'answer_patterns':[r'(semester\s*6|sixth semester|semester six)']},
        {'case_id':'PREREQUISITES','category':'semantic_paraphrase','question':'Which subjects must a student complete before taking NLP?',
         'preferred_documents':['HCE-DOC04','HCE-DOC13'],'evidence_patterns':[r'(Artificial Intelligence|HCE-CSE501)',r'(Machine Learning|HCE-CSE502)'],
         'answer_patterns':[r'(Artificial Intelligence|\bAI\b)',r'(Machine Learning|\bML\b)'],
         'human_note':'Distinguish prerequisites from same-semester lab corequisites. Patterns are not entailment.'},
        {'case_id':'CREDITS','category':'indirect_support','question':'How many credits does the NLP course carry?',
         'preferred_documents':['HCE-DOC04'],'evidence_patterns':[r'HCE-CSE603',r'credits\s*(?:\||:)?\s*'+str(nlp['credits'])],
         'indirect_patterns':[r'Natural Language Processing Laboratory',r'four.credit theory'],
         'answer_patterns':[r'\b'+str(nlp['credits'])+r'\s*credits?\b'],
         'human_note':'Expected syllabus miss is not a fact miss: Lab Guidelines can supply the corresponding four-credit theory relationship.'},
        {'case_id':'MULTI','category':'multi_source','question':'When can low attendance stop a student from writing the final examination?',
         'preferred_documents':['HCE-DOC08','HCE-DOC09'],'evidence_patterns':[r'attendance shortage',r'ineligible'], 'additional_fact_patterns':[attendance+'%'],
         'answer_patterns':[attendance], 'known_incorrect_patterns':[r'fail(?:ing)? (?:either |the )?pass condition',r'24 out of 20'],
         'human_note':'The shortage-to-ineligibility relationship is supplied; the exact percentage may be absent. Attendance eligibility differs from passing. Inspect causality; pattern checks cannot judge the whole answer.'},
        {'case_id':'CAKE','category':'unsupported_question','question':'What is the recipe for chocolate cake?',
         'preferred_documents':[],'evidence_patterns':[],'answer_patterns':[],
         'human_note':'The canonical knowledge base has no recipe. Nearest chunks are not recipe support; abstention phrase detection is case-specific.'},
        {'case_id':'RET-06','category':'semantic_paraphrase','question':'Where do connected devices learn the rules for exchanging messages?',
         'preferred_documents':['HCE-DOC02'],'evidence_patterns':[r'Computer Networks',r'HCE-CSE402'],
         'answer_patterns':[r'Computer Networks'],'known_incorrect_patterns':[r'machine learning courses'],
         'human_note':'Inspect curriculum profile and the actual generated response; no rank improvement guarantees answer correctness.'},
        {'case_id':'COMPARE','category':'comparison','question':'Compare the focus of Machine Learning and Deep Learning.',
         'preferred_documents':['HCE-DOC05','HCE-DOC06'],'evidence_patterns':[],'answer_patterns':[],
         'human_note':'Two syllabus sources are desired. Comparison quality needs human inspection, not one substring.'},
        {'case_id':'BUDGET','category':'context_failure','question':'What is the minimum attendance required?',
         'top_k':5,'max_chunks':1,'preferred_documents':['HCE-DOC08'],
         'evidence_patterns':[attendance+'%',r'attendance'],'answer_patterns':[attendance],
         'known_incorrect_patterns':[r'\b80\s*%'],'human_note':'Explicit budget experiment: useful attendance evidence can be retrieved but excluded. Defaults remain 5 context chunks elsewhere.'},
    ]
    return cases


def patterns_match(text,patterns):
    return bool(patterns) and all(re.search(p,text,flags=re.I) for p in patterns)


def evidence_state(chunks,case):
    """Curated case-specific evidence checks, never called by production."""
    preferred=set(case['preferred_documents'])
    found=set(c['document_id'] for c in chunks)
    coverage=len(preferred & found)/len(preferred) if preferred else None
    direct=any(patterns_match(c['text'],case['evidence_patterns']) for c in chunks)
    indirect=any(patterns_match(c['text'],case.get('indirect_patterns',[])) for c in chunks)
    if case['category']=='unsupported_question': classification='insufficient'
    elif case['category']=='comparison': classification='direct_support' if coverage==1 else 'insufficient'
    elif direct: classification='direct_support'
    elif indirect: classification='indirect_support'
    else: classification='insufficient'
    return {'classification':classification,'required_evidence_available':direct or indirect or (case['category']=='comparison' and coverage==1),
            'preferred_document_coverage':coverage,'preferred_document_retrieved':bool(preferred & found),
            'scope':'Curated source/text checks, not general semantic or factual evaluation.'}


def answer_checks(answer,case):
    if answer is None: return {'status':'not_generated','expected_fact_check':None,'known_incorrect_relationship':None,'abstention_style':None}
    return {'status':'task-specific deterministic checks',
            'expected_fact_check':patterns_match(answer,case['answer_patterns']) if case['answer_patterns'] else None,
            'expected_pattern_matches':{p:bool(re.search(p,answer,flags=re.I)) for p in case['answer_patterns']},
            'known_incorrect_relationship':any(re.search(p,answer,flags=re.I) for p in case.get('known_incorrect_patterns',[])),
            'abstention_style':any(phrase in answer.lower() for phrase in ABSTENTION_PHRASES),
            'scope':'Numbers/phrases can appear in wrong claims; these are not general correctness or hallucination scores.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate',action='store_true',help='Use explicitly preloaded optional Qwen; never download')
    parser.add_argument('--output',type=Path,default=ROOT/'workshop/rag_evaluation_results.json')
    args=parser.parse_args()
    facts=json.loads((ROOT/'data/university_facts.json').read_text(encoding='utf-8'))
    start=time.perf_counter();documents=load_canonical_documents(ROOT);chunks=build_chunk_corpus(documents)
    model=load_sentence_embedding_model()
    lengths=model_token_lengths(model.tokenizer,[c['text'] for c in chunks])
    if max(lengths)>model.max_seq_length: raise ValueError('Canonical chunks exceed the encoder window.')
    matrix=encode_document_chunks(model,chunks);initialization=time.perf_counter()-start
    llm=None
    if args.generate:
        try: llm=load_local_llm()
        except LocalModelUnavailable as error: parser.exit(1,str(error)+'\n')
    def retriever(question,k):
        return semantic_chunk_search(question,chunks,matrix,model,k)
    records=[]
    for case in evaluation_cases(facts):
        result=run_transparent_rag(case['question'],retriever,llm,top_k=case.get('top_k',5),
            max_chunks=case.get('max_chunks',5),generation_model=MODEL_ID if llm else None)
        retrieved=result['retrieval']['results'];included=list(result['context']['included_sources'].values())
        record={**case,'top_k':case.get('top_k',5),'retrieved_sources':retrieved,
            'included_sources':result['attribution']['supplied_sources'],'context_statistics':result['context']['statistics'],
            'excluded_sources':result['context']['excluded_sources'],'retrieval_evidence':evidence_state(retrieved,case),
            'context_evidence':evidence_state(included,case),'evidence_classification':evidence_state(included,case)['classification'],
            'answer':result['generation']['answer'],'generation_status':result['generation']['status'],
            'generation_error':result['generation']['error'],'deterministic_checks':answer_checks(result['generation']['answer'],case),
            'model_source_labels':result['attribution']['model_label_inspection']['mentioned_labels'],
            'label_validity':result['attribution']['model_label_inspection'],
            'diagnostics':result['diagnostics'],'timings_seconds':result['timings_seconds'],
            'prompt_sha256':hashlib.sha256(result['prompt'].encode()).hexdigest()}
        records.append(record)
        print(case['case_id'],record['evidence_classification'],repr(record['answer']),record['deterministic_checks'],flush=True)
    generated=[r for r in records if r['answer'] is not None]
    applicable=[r for r in generated if r['deterministic_checks']['expected_fact_check'] is not None]
    report={'usage':'Measured instructor evaluation; production never reads expected answers/checks.',
        'utc_time':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'packages':{name:version(name) for name in ['torch','transformers','sentence-transformers']},
        'generation_enabled':args.generate,'model_id':MODEL_ID,'model_revision':MODEL_REVISION,
        'generation_settings':{'device':'cpu','dtype':'float32','do_sample':False,'num_beams':1,'max_new_tokens':96,'max_input_tokens':3072},
        'embedding_model':SENTENCE_MODEL_ID,'embedding_revision':SENTENCE_MODEL_REVISION,
        'chunk_settings':{'size':100,'overlap':20,'count':len(chunks),'token_max':max(lengths),'above_256':sum(n>256 for n in lengths)},
        'context_defaults':{'max_chunks':5,'max_characters':8000},'default_top_k':5,
        'prompt_template_sha256':hashlib.sha256(build_grounded_prompt('<<QUESTION>>','<<CONTEXT>>').encode()).hexdigest(),
        'source_sha256':{Path(d['source_path']).relative_to(ROOT).as_posix():hashlib.sha256(Path(d['source_path']).read_bytes()).hexdigest() for d in documents},
        'expected_fact_contract_sha256':hashlib.sha256((ROOT/'data/university_facts.json').read_bytes()).hexdigest(),
        'initialization_seconds':initialization,'abstention_phrase_set':ABSTENTION_PHRASES,'cases':records,
        'summary':{'case_count':len(records),'retrieval_evidence_available':sum(r['retrieval_evidence']['required_evidence_available'] for r in records),
            'context_evidence_available':sum(r['context_evidence']['required_evidence_available'] for r in records),
            'context_excluded_chunks':sum(r['context_statistics']['excluded_chunks'] for r in records),
            'generated_count':len(generated),'expected_fact_check_cases':len(applicable),
            'expected_fact_check_passes':sum(r['deterministic_checks']['expected_fact_check'] for r in applicable),
            'known_incorrect_relationship_cases':sum(r['deterministic_checks'].get('known_incorrect_relationship') is True for r in generated),
            'source_label_emission_count':sum(r['label_validity']['has_source_labels'] for r in generated),
            'source_label_emission_rate':sum(r['label_validity']['has_source_labels'] for r in generated)/len(generated) if generated else None,
            'unknown_label_count':sum(len(r['label_validity']['invalid_labels']) for r in generated)},
        'limitations':'Curated evidence and case-specific substring checks, not generalized answer accuracy. Human notes are review guidance; raw outputs are unedited. Preferred-source misses can have indirect support.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Saved',args.output,report['summary'],flush=True)


if __name__=='__main__':
    main()
