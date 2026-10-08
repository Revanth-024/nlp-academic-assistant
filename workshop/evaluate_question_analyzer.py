"""Evaluation only: observed paper-text predictions versus synthetic ground truth."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from src.question_analyzer import (load_topic_taxonomy,load_question_papers,analyze_questions,
    count_topic_frequencies,rank_topics,filter_questions_by_topic,UNMATCHED,SPECIFICITY)
ROOT=Path(__file__).resolve().parents[1]


def evaluate():
    taxonomy=load_topic_taxonomy(ROOT/'data/university_facts.json')
    start=time.perf_counter();papers=load_question_papers(ROOT/'data/question_papers');extraction=time.perf_counter()-start
    records=[q for p in papers for q in p['questions']]
    start=time.perf_counter();questions=analyze_questions(records,taxonomy);analysis=time.perf_counter()-start
    start=time.perf_counter();observed=count_topic_frequencies(questions,taxonomy);ranking=rank_topics(observed,taxonomy);aggregation=time.perf_counter()-start
    # Load labels AFTER production analysis. No labels are passed to matching.
    truth=json.loads((ROOT/'data/question_papers/question_metadata.json').read_text(encoding='utf-8'))
    expected={(q['paper_id'],q['question_id']):q for q in truth['questions']}
    if set(expected)!={(q['paper_id'],q['question_id']) for q in questions}: raise ValueError('Prediction/truth identities differ.')
    rows=[];confusions=Counter();ground=Counter()
    for q in questions:
        target=expected[(q['paper_id'],q['question_id'])]
        if q['text']!=target['question_text'] or q['marks']!=target['marks']:
            raise ValueError('Extracted text/marks differ from validation metadata.')
        topic=target['primary_topic_id'];ground[topic]+=1
        correct=q['primary_topic']==topic
        if not correct: confusions[(topic,q['primary_topic'])]+=1
        rows.append({**q,'ground_truth_primary':topic,'correct':correct})
    metrics=[]
    for topic in taxonomy:
        id=topic['id']
        tp=sum(q['primary_topic']==q['ground_truth_primary']==id for q in rows)
        fp=sum(q['primary_topic']==id and q['ground_truth_primary']!=id for q in rows)
        fn=sum(q['ground_truth_primary']==id and q['primary_topic']!=id for q in rows)
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        metrics.append({'topic_id':id,'true_positives':tp,'false_positives':fp,'false_negatives':fn,
            'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.0})
    report={'usage':'Measured evaluation only. User-facing counts/predictions come from production paper text, never this artifact.',
        'utc_time':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
        'taxonomy_source':'data/university_facts.json#/nlp_topic_taxonomy','taxonomy_schema_version':json.loads((ROOT/'data/university_facts.json').read_text())['schema_version'],
        'topic_count':len(taxonomy),'specificity_policy':SPECIFICITY,
        'rule_scope':'Canonical names/aliases; separator normalization, word boundaries, limited final-word singular/plural variation. Related IDs are never propagated. No new synonyms/question-specific hacks.',
        'paper_ids':[p['paper_id'] for p in papers],'questions_per_paper':{p['paper_id']:len(p['questions']) for p in papers},
        'paper_marks':{p['paper_id']:sum(q['marks'] for q in p['questions']) for p in papers},
        'total_questions':len(rows),'correct':sum(q['correct'] for q in rows),'incorrect':sum(not q['correct'] for q in rows),
        'unmatched':observed[UNMATCHED],'matched_but_incorrect':sum(not q['correct'] and q['primary_topic']!=UNMATCHED for q in rows),
        'accuracy':sum(q['correct'] for q in rows)/len(rows),'ambiguous':sum(q['is_ambiguous'] for q in rows),
        'observed_frequencies':observed,'ground_truth_frequencies':{t['id']:ground[t['id']] for t in taxonomy},
        'frequency_differences':{t['id']:observed[t['id']]-ground[t['id']] for t in taxonomy},'ranking':ranking,
        'per_topic_metrics':metrics,'macro_f1':sum(m['f1'] for m in metrics)/len(metrics),
        'confusion_pairs':[{'actual':a,'predicted':p,'count':count} for (a,p),count in sorted(confusions.items())],
        'questions':rows,'transformer_filter':filter_questions_by_topic(questions,'transformers',taxonomy),
        'attention_filter':filter_questions_by_topic(questions,'attention',taxonomy),
        'timings_seconds':{'extraction':extraction,'classification':analysis,'aggregation_and_ranking':aggregation},
        'source_sha256':{Path(p['source_path']).relative_to(ROOT).as_posix():hashlib.sha256(Path(p['source_path']).read_bytes()).hexdigest() for p in papers},
        'taxonomy_contract_sha256':hashlib.sha256((ROOT/'data/university_facts.json').read_bytes()).hexdigest(),
        'ground_truth_sha256':hashlib.sha256((ROOT/'data/question_papers/question_metadata.json').read_bytes()).hexdigest(),
        'interpretation':'Unmatched paraphrases/components are honest taxonomy-coverage limits. Accuracy is measured on four synthetic papers, not general unseen-paper performance or exam prediction.'}
    return report


if __name__=='__main__':
    report=evaluate()
    (ROOT/'workshop/question_analyzer_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('Questions',report['total_questions'],'correct',report['correct'],'incorrect',report['incorrect'],'unmatched',report['unmatched'],'accuracy',report['accuracy'])
    print('Observed',report['observed_frequencies']);print('Truth',report['ground_truth_frequencies'])
    print('Timing',report['timings_seconds']);print('Confusions',report['confusion_pairs'])
