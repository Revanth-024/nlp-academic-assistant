"""Exercise 10 invariants: boundaries, provenance, extraction and measured fit."""
from io import BytesIO
from pathlib import Path
import hashlib
import json
import pytest
from pypdf import PdfReader, PdfWriter
from src.document_processor import *
from src.sentence_resources import load_sentence_tokenizer
ROOT=Path(__file__).resolve().parents[1]


def test_cleaning_preserves_evidence():
    assert clean_document_text('  # NLP\r\n\r\n\r\nAttendance:\t75%.  \rMarks: 40 + 60 = 100! ') == '# NLP\n\nAttendance: 75%.\nMarks: 40 + 60 = 100!'
    assert clean_document_text('')==''
    assert clean_document_text(' \t\n ')==''
    assert clean_document_text('Dr. Nivora Pellin — NLP')=='Dr. Nivora Pellin — NLP'
    with pytest.raises(TypeError): clean_document_text(None)


@pytest.mark.parametrize('splitter',[split_text_into_chunks,lambda text,size:create_overlapping_chunks(text,size,0)])
def test_nonoverlapping_final_partial(splitter):
    chunks=splitter('a b c d e f g',3)
    assert [c['text'] for c in chunks]==['a b c','d e f','g']
    assert [(c['word_start'],c['word_end']) for c in chunks]==[(0,3),(3,6),(6,7)]
    assert splitter('',3)==[]
    assert len(splitter('a b c',3))==1
    assert len(splitter('a',3))==1


@pytest.mark.parametrize('size',[0,-1,True,1.5])
def test_invalid_size(size):
    with pytest.raises(ValueError): split_text_into_chunks('a b',size)


@pytest.mark.parametrize('overlap',[-1,3,4,True,1.5])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError): create_overlapping_chunks('a b c',3,overlap)


def test_overlap_progress_and_no_redundant_tail():
    chunks=create_overlapping_chunks('a b c d e f g h',4,1)
    assert [c['text'] for c in chunks]==['a b c d','d e f g','g h']
    assert len(create_overlapping_chunks('a b c d',4,3))==1
    assert create_overlapping_chunks('',4,1)==[]
    assert create_overlapping_chunks('a b c',3,0)==split_text_into_chunks('a b c',3)
    assert create_overlapping_chunks('a b c d',3,2)==create_overlapping_chunks('a b c d',3,2)


def test_exact_offsets_and_metadata_no_mutation():
    text='# Title\n\nA  B\nC.'
    chunks=split_text_into_chunks(text,2)
    source={'document_id':'D1','title':'Title','source_path':'data/a.md','source_type':'markdown'}
    labeled=attach_chunk_metadata(chunks,source)
    assert 'chunk_id' not in chunks[0]
    for i,c in enumerate(labeled):
        assert c['text']==text[c['char_start']:c['char_end']]
        assert c['text'].split()==text.split()[c['word_start']:c['word_end']]
        assert c['chunk_id']==f'D1::chunk_{i:03d}'
        assert c['page_number'] is None and c['document_title']=='Title'
    with pytest.raises(ValueError): attach_chunk_metadata(chunks,source,page_number=1)
    with pytest.raises(ValueError): attach_chunk_metadata(chunks,{**source,'title':''})
    with pytest.raises(ValueError): attach_chunk_metadata(chunks,source,start_index=-1)
    source['source_type']='pdf'
    with pytest.raises(ValueError): attach_chunk_metadata(chunks,source)
    assert attach_chunk_metadata(chunks,source,page_number=2,start_index=7)[0]['chunk_id']=='D1::chunk_007'


def test_process_page_boundaries_and_statistics():
    doc={'document_id':'PDF1','title':'Test','source_path':'test.pdf','source_type':'pdf','pages':[{'page_number':1,'text':'a b c d e'},{'page_number':2,'text':''},{'page_number':3,'text':'f g h'}]}
    result=process_document(doc,3,1)
    assert [c['page_number'] for c in result['chunks']]==[1,1,3]
    assert [c['chunk_index'] for c in result['chunks']]==[0,1,2]
    assert result['chunks'][2]['word_start']==0
    assert result['statistics']['document_word_count']==8
    assert result['statistics']['duplicated_word_positions']==1
    assert len(result['cleaned_pages'])==3
    assert doc['pages'][1]['text']==''
    empty=process_document({**doc,'pages':[{'page_number':1,'text':''}]})
    assert empty['statistics']['chunk_count']==0 and empty['statistics']['average_words']==0


def test_markdown_loading(tmp_path):
    path=tmp_path/'test.md';path.write_text('\ufeff# Title\nNLP: 75%.',encoding='utf-8')
    doc=load_markdown_document(path,document_id='D')
    assert doc['title']=='Title' and doc['source_type']=='markdown'
    assert doc['pages'][0]['page_number'] is None
    with pytest.raises(DocumentExtractionError): load_markdown_document(tmp_path/'missing.md')


@pytest.mark.parametrize('filename,pages,terms',[
 ('04_nlp_syllabus.pdf',3,['HCE-CSE603','HCE-CSE501','HCE-CSE502','Nivora Pellin','Unit 4','GPT']),
 ('08_attendance_policy.pdf',2,['75%','No condonation']),
 ('09_examination_regulations.pdf',2,['40 marks','60 marks','100 marks','50 total marks','24 end-semester'])])
def test_derived_pdf_facts(filename,pages,terms):
    path=ROOT/'data/sample_pdfs'/filename
    doc=extract_text_from_pdf(path)
    assert len(doc['pages'])==pages
    text='\n'.join(p['text'] for p in doc['pages'])
    assert 'Synthetic educational data created for NLP workshop purposes.' in text
    assert 'fictional' in text.lower() and 'Hindu College of Engineering' in text
    for term in terms: assert term in text
    assert [p['page_number'] for p in doc['pages']]==list(range(1,pages+1))
    assert all(c['page_number'] in range(1,pages+1) for c in process_document(doc)['chunks'])
    uploaded=extract_text_from_pdf(path.read_bytes(),source_name=filename)
    assert uploaded['source_path']=='upload://'+filename
    assert uploaded['document_id']==extract_text_from_pdf(path.read_bytes())['document_id']


def pdf_bytes(writer):
    stream=BytesIO();writer.write(stream);return stream.getvalue()


def test_bad_blank_encrypted_large_pdfs():
    with pytest.raises(DocumentExtractionError,match='valid'): extract_text_from_pdf(b'invalid')
    writer=PdfWriter();writer.add_blank_page(width=100,height=100)
    with pytest.raises(DocumentExtractionError,match='OCR'): extract_text_from_pdf(pdf_bytes(writer))
    writer.encrypt('secret')
    with pytest.raises(DocumentExtractionError,match='unencrypted'): extract_text_from_pdf(pdf_bytes(writer))
    with pytest.raises(DocumentExtractionError,match='10 MiB'): extract_text_from_pdf(b'0'*(MAX_PDF_BYTES+1))
    writer=PdfWriter()
    for _ in range(51): writer.add_blank_page(width=100,height=100)
    with pytest.raises(DocumentExtractionError,match='50 pages'): extract_text_from_pdf(pdf_bytes(writer))


def test_empty_physical_page_keeps_number():
    reader=PdfReader(ROOT/'data/sample_pdfs/08_attendance_policy.pdf')
    writer=PdfWriter();writer.add_blank_page(width=100,height=100);writer.add_page(reader.pages[0])
    doc=extract_text_from_pdf(pdf_bytes(writer))
    assert doc['pages'][0]=={'page_number':1,'text':''}
    assert doc['pages'][1]['page_number']==2
    assert {c['page_number'] for c in process_document(doc)['chunks']}=={2}


def test_manifest_canonical_sources_unchanged():
    manifest=json.loads((ROOT/'data/document_manifest.json').read_text())
    assert len(manifest['academic_documents'])==14 and len(manifest['question_papers'])==4
    assert len(manifest['derived_sample_pdfs'])==3
    for entry in manifest['derived_sample_pdfs']:
        assert hashlib.sha256((ROOT/'data'/entry['canonical_filename']).read_bytes()).hexdigest()==entry['canonical_source_sha256']
        assert (ROOT/'data'/entry['filename']).exists()
        assert entry['canonical_document_id'] in {d['document_id'] for d in manifest['academic_documents']}


def test_default_chunk_fit_cached_tokenizer_and_measured_artifact():
    tokenizer=load_sentence_tokenizer()  # Explicit preflight required, never downloads.
    report=json.loads((ROOT/'workshop/chunk_analysis_results.json').read_text())
    assert report['default']=={'chunk_size':DEFAULT_CHUNK_SIZE,'overlap':DEFAULT_OVERLAP}
    words,lengths=[],[]
    for relative,digest in report['source_sha256'].items():
        path=ROOT/relative
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
        doc=load_markdown_document(path);chunks=process_document(doc)['chunks']
        words.extend(c['word_end']-c['word_start'] for c in chunks)
        lengths.extend(model_token_lengths(tokenizer,[c['text'] for c in chunks]))
    assert len(lengths)==report['default_chunk_tokens']['count']
    assert max(lengths)<=256 and max(lengths)==report['default_chunk_tokens']['max']
    assert max(words)==DEFAULT_CHUNK_SIZE
    assert model_token_lengths(tokenizer,[])==[]


@pytest.mark.parametrize('todo',['10.1','10.2','10.3','10.4'])
def test_todo_documentation_synchronization(todo):
    for relative in ['src/document_processor.py','workshop/TODO_INDEX.md','docs/STUDENT_GUIDE.md','docs/INSTRUCTOR_GUIDE.md']:
        assert todo in (ROOT/relative).read_text(encoding='utf-8-sig')
