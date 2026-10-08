"""Instructor chunk measurements; cached tokenizer only, no embeddings."""
from pathlib import Path
import hashlib
import json
from src.document_processor import load_markdown_document, process_document, model_token_lengths
from src.sentence_resources import load_sentence_tokenizer, SENTENCE_MODEL_ID, SENTENCE_MODEL_REVISION
ROOT = Path(__file__).resolve().parents[1]

def distribution(values):
    return {"count": len(values), "min": min(values, default=0), "average": sum(values)/len(values) if values else 0, "max": max(values, default=0), "above_256": sum(v>256 for v in values)}

def main():
    tokenizer=load_sentence_tokenizer()
    manifest=json.loads((ROOT/'data/document_manifest.json').read_text())
    documents=[load_markdown_document(ROOT/'data'/e['filename'],document_id=e['document_id'],title=e['title']) for e in manifest['academic_documents']]
    report={'model_id':SENTENCE_MODEL_ID,'model_revision':SENTENCE_MODEL_REVISION,'token_counting':'Untruncated WordPieces including special tokens; tokenizer only','default':{'chunk_size':100,'overlap':20},'source_sha256':{Path(d['source_path']).relative_to(ROOT).as_posix():hashlib.sha256(Path(d['source_path']).read_bytes()).hexdigest() for d in documents},'documents':[],'experiments':[]}
    full,words,tokens=[],[],[]
    for d in documents:
        p=process_document(d); lengths=model_token_lengths(tokenizer,[c['text'] for c in p['chunks']]); total=model_token_lengths(tokenizer,[p['cleaned_pages'][0]['text']])[0]
        full.append(total); tokens.extend(lengths); words.extend(c['word_end']-c['word_start'] for c in p['chunks'])
        report['documents'].append({'document_id':d['document_id'],'statistics':p['statistics'],'full_model_tokens':total,'chunk_model_tokens':distribution(lengths),'oversized_chunk_ids':[c['chunk_id'] for c,n in zip(p['chunks'],lengths) if n>256]})
    report.update(full_document_tokens=distribution(full),default_chunk_words=distribution(words),default_chunk_tokens=distribution(tokens))
    for size,overlap in [(60,20),(100,0),(100,20),(100,40),(120,20),(240,20)]:
        words,tokens,duplicates=[],[],0
        for d in documents:
            p=process_document(d,size,overlap); words.extend(c['word_end']-c['word_start'] for c in p['chunks']); tokens.extend(model_token_lengths(tokenizer,[c['text'] for c in p['chunks']])); duplicates+=p['statistics']['duplicated_word_positions']
        report['experiments'].append({'chunk_size':size,'overlap':overlap,'words':distribution(words),'model_tokens':distribution(tokens),'duplicated_word_positions':duplicates})
    (ROOT/'workshop/chunk_analysis_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['default_chunk_words','default_chunk_tokens','full_document_tokens','experiments']},indent=2))

if __name__=='__main__':
    main()
