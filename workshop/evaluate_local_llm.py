"""Measured Task13 prompt/backend experiments; expected outputs never supply answers."""
from pathlib import Path
import json
import platform
import re
import time
from src.retrieval import load_canonical_documents,build_chunk_corpus,encode_document_chunks,semantic_chunk_search
from src.sentence_resources import load_sentence_embedding_model
from src.rag import prepare_rag_request,run_rag
from src.llm_resources import load_local_llm,model_directory,MODEL_ID,MODEL_REVISION,MAX_NEW_TOKENS,MAX_INPUT_TOKENS
ROOT=Path(__file__).resolve().parents[1]


def main():
    embedder=load_sentence_embedding_model()
    chunks=build_chunk_corpus(load_canonical_documents(ROOT));matrix=encode_document_chunks(embedder,chunks)
    start=time.perf_counter();llm=load_local_llm();load_seconds=time.perf_counter()-start
    parameter_bytes=sum(p.numel()*p.element_size() for p in llm.model.parameters())
    cases=[('attendance','What is the minimum attendance required?'),
           ('numeric','What is the minimum end-semester score required to pass?'),
           ('unsupported','What is the recipe for chocolate cake?'),
           ('retrieval_failure','How many credits does the NLP course carry?'),
           ('multi_source','When can low attendance stop a student from writing the final examination?')]
    results=[]
    for identity,question in cases:
        search=semantic_chunk_search(question,chunks,matrix,embedder,5)
        request=prepare_rag_request(question,search['results'],requested_top_k=5)
        weak='Answer the question using this context.\n\n'+request['context']+'\n\nQuestion: '+question+'\nAnswer:'
        for mode,prompt in [('weak',weak),('grounded',request['prompt'])]:
            start=time.perf_counter();answer=llm.generate(prompt);elapsed=time.perf_counter()-start
            labels=re.findall(r'\[SOURCE (\d+)\]',answer)
            row={'case':identity,'question':question,'top_k':5,'prompt_mode':mode,
                 'answer':answer,'generation_seconds':elapsed,'context_statistics':request['statistics'],
                 'included_chunk_ids':request['included_chunk_ids'], 'source_mapping':request['sources'],
                 'emitted_labels':labels,'all_labels_exist':all('SOURCE '+label in request['sources'] for label in labels) if labels else None,
                 'citation_support':'Not automatically verified. Instructor must inspect claims against actual text.',
                 'generation_input_tokens':len(llm.tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True))}
            results.append(row);print(identity,mode,round(elapsed,2),repr(answer),flush=True)
    attendance=cases[0][1]
    for k in [1,3]:
        search=semantic_chunk_search(attendance,chunks,matrix,embedder,k)
        start=time.perf_counter();result=run_rag(attendance,search['results'],llm,requested_top_k=k);elapsed=time.perf_counter()-start
        results.append({'case':'attendance','question':attendance,'top_k':k,'prompt_mode':'grounded','answer':result['answer'],
                        'generation_seconds':elapsed,'context_statistics':result['statistics'],'included_chunk_ids':result['included_chunk_ids']})
        print('attendance K',k,repr(result['answer']),flush=True)
    try:
        import psutil
        rss=psutil.Process().memory_info().rss
    except ImportError:
        rss=None
        if platform.system()=='Windows':
            import ctypes
            class MemoryCounters(ctypes.Structure):
                _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
            counters=MemoryCounters();counters.cb=ctypes.sizeof(counters)
            ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
            handle=ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(counters),counters.cb):
                rss=counters.WorkingSetSize
    report={'model_id':MODEL_ID,'revision':MODEL_REVISION,'license':'Apache-2.0','backend':'Existing Transformers/PyTorch, float32 CPU',
        'python':platform.python_version(),'cache_bytes':sum(p.stat().st_size for p in model_directory().rglob('*') if p.is_file()),
        'model_parameter_bytes':parameter_bytes,'process_rss_bytes_after_generations':rss,
        'rss_scope':'Whole process, including MiniLM, generator, PyTorch and allocator; not generator-only peak.',
        'model_load_seconds':load_seconds,'generation_parameters':{'do_sample':False,'num_beams':1,'max_new_tokens':MAX_NEW_TOKENS,'max_input_tokens':MAX_INPUT_TOKENS,'dtype':'float32','device':'cpu'},
        'context_budget':{'max_chunks':5,'max_characters':8000,'policy':'whole-block rank prefix; no deduplication'},'experiments':results,
        'candidate_selection':'SmolLM2-360M: 723674912-byte weights, 8192 positions. Smol135: 269060552 bytes, smaller capacity. FLAN-T5-small: 307867048 bytes, 512-token configured baseline; long-input behavior would need separate testing. Qwen0.5B: 988097824 bytes, larger download/CPU memory. Smol360 was downloaded/tested first and failed grounding. Qwen0.5B was then evaluated with identical prompt/corpus settings; the Smol trial is retained separately.',
        'case_interpretation':'Credit case is a strict expected-syllabus source miss; actual Top-5 also contains lab-guideline support for four theory credits. Attendance K=1 is a genuine evidence gap.',
        'usage':'Measured instructor output, not benchmark truth, verified citations, or production answer guarantees.'}
    (ROOT/'workshop/local_llm_evaluation_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Model load',load_seconds,'cache',report['cache_bytes'],'RAM',parameter_bytes,rss,flush=True)


if __name__=='__main__':
    main()
