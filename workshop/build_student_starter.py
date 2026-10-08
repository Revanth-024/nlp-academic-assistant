"""Deterministic instructor-only starter builder. Reference files stay untouched."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[1]
MARKER=re.compile(r'^(?P<indent>[ \t]*)# (?P<kind>BEGIN|END) STUDENT CORE (?P<label>\d+\.\d+[^\r\n]*)\r?$',re.M)
STUDENT_TESTS=['test_preprocessing.py','test_classical_nlp.py','test_embeddings.py','test_sentence_embeddings.py',
               'test_retrieval.py','test_semantic_retrieval.py','test_transformers_nlp.py',
               'test_document_processor.py','test_chunk_retrieval.py','test_rag.py','test_question_analyzer.py']
PUBLIC_DOCS=['STUDENT_GUIDE.md','SYNTHETIC_DATA_DESIGN.md','GITHUB_PORTFOLIO_GUIDE.md','DEPLOYMENT_GUIDE.md','KNOWN_LIMITATIONS.md']
PUBLIC_WORKSHOP=['TODO_INDEX.md','SETUP_CHECKLIST.md','WORKSHOP_SCHEDULE.md','preflight.py']


def core_regions(text):
    """Explicit paired markers, exact labels/indent; reject nesting/duplicates."""
    matches=list(MARKER.finditer(text));regions=[];opened=None;seen=set()
    for marker in matches:
        label=marker['label'].strip()
        if marker['kind']=='BEGIN':
            if opened is not None or label in seen:raise ValueError('Nested or duplicate student core: '+label)
            opened=marker;seen.add(label)
        else:
            if opened is None or opened['label'].strip()!=label or opened['indent']!=marker['indent']:
                raise ValueError('Mismatched student core: '+label)
            regions.append({'label':label,'id':label.split()[0],'indent':marker['indent'],
                            'start':opened.end(),'end':marker.start()});opened=None
    if opened is not None:raise ValueError('Unclosed student core: '+opened['label'])
    return regions


def strip_cores(text):
    """Keep scaffolding and interfaces; remove only bounded educational bodies."""
    regions=core_regions(text)
    for region in reversed(regions):
        message='STUDENT TODO '+region['label']+': implement this exercise in the marked core; see docs/STUDENT_GUIDE.md.'
        replacement='\n'+region['indent']+'# Implement only this educational core; surrounding setup stays provided.\n'
        replacement+=region['indent']+'raise NotImplementedError('+repr(message)+')\n'
        text=text[:region['start']]+replacement+text[region['end']:]
    text=text.replace('STUDENT TODO BOUNDARY ', 'STUDENT TODO ')
    text=re.sub(r'Complete (?:working )?reference(?: implementation)?(?: below)?', 'Student scaffold',text,flags=re.I)
    ast.parse(text)
    return text,[region['id'] for region in regions]


def curate_tests(text):
    """AST/name-based exclusion of reference-marker/evaluator-only checks."""
    text=text.replace('evidence()+evidence()', "'duplicate-fixture'")
    text=text.replace('def test_bad_retrieval_items(bad):', 'def test_bad_retrieval_items(bad):\n    if bad == "duplicate-fixture": bad = evidence() + evidence()')
    tree=ast.parse(text);removed=[];names=set()
    for node in tree.body:
        if isinstance(node,ast.ImportFrom) and (node.module or '').startswith(('workshop.','tests.')):
            removed.append(node);names.update(n.asname or n.name for n in node.names)
    changed=True
    while changed:
        changed=False
        for node in tree.body:
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) or node in removed:continue
            instructor=any(word in node.name for word in ['synchron','boundary','boundaries','evaluator','artifact','real_canonical_evidence'])
            depends=any(isinstance(n,ast.Name) and n.id in names for n in ast.walk(node))
            if instructor or depends:
                removed.append(node);names.add(node.name);changed=True
    lines=text.splitlines(keepends=True)
    for node in sorted(removed,key=lambda n:n.lineno,reverse=True):
        start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])-1
        del lines[start:node.end_lineno]
    return ''.join(lines)


def scaffold_page(text):
    """Only expected incomplete exercises are caught; coding bugs remain errors."""
    text=text.replace('Working reference state', 'Student starter state').replace('Complete reference cores','Student exercise cores')
    return ('"""Generated student UI: infrastructure retained, expected TODOs explained."""\n'
            'import streamlit as st\ntry:\n'+''.join('    '+line if line.strip() else line for line in text.splitlines(keepends=True))+
            '\nexcept NotImplementedError as error:\n    st.info(str(error))\n'
            '    st.caption("Complete the indicated TODO in src/, run its exercise tests, then rerun this page. Other errors are not hidden by this wrapper.")\n')


def scaffold_notebook(text):
    notebook=json.loads(text)
    notebook['cells'].insert(0,{'cell_type':'markdown','metadata':{},'source':['# Student starter\n','Run cells in order. An incomplete TODO prints guidance and stops subsequent cells. After implementing it, restart the kernel and run again.']})
    for cell in notebook['cells']:
        if cell['cell_type']!='code':continue
        code=''.join(cell['source'])
        if 'from workshop.evaluate_retrieval import' in code:
            tree=ast.parse(code);lines=code.splitlines(keepends=True)
            for node in reversed(tree.body):
                if isinstance(node,ast.ImportFrom) and node.module=='workshop.evaluate_retrieval':
                    lines[node.lineno-1:node.end_lineno]=['from workshop.course_data import load_course_documents, DEFAULT_QUERIES\n']
            code=''.join(lines)
        if any(name in code for name in ['load_benchmark_documents(', 'evaluate_expected_sources(', 'evaluation_group(', 'summarize_single_source_metrics(']):
            cell['source']=['# Instructor benchmark evaluation omitted; inspect your own completed search results.\n'];continue
        cell['execution_count']=None;cell['outputs']=[]
        if re.search(r'(?:_results\.json|evaluation_results\.json)',code):
            cell['source']=['# Instructor measured report omitted. Run your own completed functions to inspect results.\n'];continue
        wrapped='if not globals().get("_workshop_todo_incomplete", False):\n    try:\n'
        wrapped+=''.join('        '+line if line.strip() else line for line in (code+'\n').splitlines(keepends=True))
        wrapped+='    except NotImplementedError as error:\n        print(str(error))\n        print("Implement the indicated TODO, restart the kernel, and run again.")\n        _workshop_todo_incomplete = True\nelse:\n    print("Earlier TODO incomplete; later cells are paused.")\n'
        cell['source']=wrapped.splitlines(keepends=True)
    return json.dumps(notebook,indent=2,ensure_ascii=False)+'\n'


def allowed_files(root):
    files=[]
    for name in ['requirements.txt','pytest.ini','.gitignore','.env.example','README.md','app.py']:
        files.append(root/name)
    for folder,suffixes in [('src',{'.py'}),('pages',{'.py'}),('notebooks',{'.ipynb'}),('data',{'.md','.json','.csv','.pdf'})]:
        files.extend(p for p in (root/folder).rglob('*') if p.is_file() and not p.is_symlink() and (p.suffix in suffixes or p.name=='.gitkeep') and '__pycache__' not in p.parts)
    files.extend(root/'docs'/name for name in PUBLIC_DOCS)
    files.extend(root/'workshop'/name for name in PUBLIC_WORKSHOP)
    files.extend(root/'tests'/name for name in STUDENT_TESTS+['__init__.py'])
    files.extend(p for p in (root/'assets').rglob('.gitkeep'))
    if any(not p.is_file() for p in files):raise ValueError('A required release file is missing.')
    return sorted(set(files),key=lambda p:p.relative_to(root).as_posix())


def safe_output(root,output):
    root=root.resolve();output=output.resolve()
    if output==root or output in root.parents or output.is_relative_to(root/'src') or output.is_relative_to(root/'data'):
        raise ValueError('Output must not replace canonical source or its source/data directories.')
    if output.is_relative_to(root) and not output.is_relative_to(root/'dist'):
        raise ValueError('Inside this repository, output must be under ignored dist/.')
    if output.exists():
        manifest_path=output/'STUDENT_STARTER_MANIFEST.json'
        if not manifest_path.is_file():raise ValueError('Existing output is not an unchanged generated starter. Choose a new directory.')
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
        expected=set(manifest['files'])|{'STUDENT_STARTER_MANIFEST.json'}
        if actual!=expected:raise ValueError('Output contains added/removed files; preserve student work and choose a new directory.')
        for name,digest in manifest['files'].items():
            path=(output/name).resolve()
            if not path.is_relative_to(output) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Output has changed; refusing to overwrite student work.')
    return output


def build_starter(output,root=ROOT):
    root=Path(root).resolve();output=safe_output(root,Path(output));files=allowed_files(root)
    source_digest=hashlib.sha256();transformed=[];contents={}
    for tooling in ['build_student_starter.py','validate_student_starter.py']:
        source_digest.update(tooling.encode()+b'\0'+(root/'workshop'/tooling).read_bytes()+b'\0')
    for file in files:
        relative=file.relative_to(root).as_posix();raw=file.read_bytes()
        source_digest.update(relative.encode()+b'\0'+raw+b'\0')
        if file.suffix=='.py':
            text=raw.decode('utf-8-sig')
            if relative.startswith('src/'):
                text,ids=strip_cores(text);transformed+=ids
            elif relative.startswith('pages/'):text=scaffold_page(text)
            elif relative.startswith('tests/'):text=curate_tests(text)
            elif relative=='workshop/preflight.py':
                text=text.replace("except Exception as error: checks.append", "except NotImplementedError as error: checks.append({'check': name, 'status': 'WARNING', 'detail': str(error)})\n        except Exception as error: checks.append")
            contents[relative]=text.encode('utf-8')
        elif file.suffix=='.ipynb':contents[relative]=scaffold_notebook(raw.decode('utf-8-sig')).encode('utf-8')
        else:contents[relative]=raw
    if len(set(transformed))!=49 or len(transformed)!=50 or Counter(transformed)['1.7']!=2:
        raise ValueError('Expected49 exercise IDs /50 labelled regions; reference markers drifted.')
    contents['app.py']=('"""Generated student workshop home."""\nimport streamlit as st\nst.set_page_config(page_title="NLP Workshop Starter",page_icon="📚")\nst.title("From Classical NLP to Generative AI")\nst.header("Building an Intelligent Academic Assistant")\nst.info("Student starter:49 algorithm TODOs are intentionally incomplete. Pages show the first TODO required; infrastructure is provided.")\nst.write("UNDERSTAND → IMPLEMENT → EXPERIMENT → COMPARE → INTEGRATE → DEPLOY")\nst.write("Begin with START_HERE.md, docs/STUDENT_GUIDE.md and workshop/SETUP_CHECKLIST.md. Run tests/test_starter_smoke.py to check the scaffolding; exercise correctness tests become green as you implement the cores.")\n').encode()
    contents['README.md']=('''# NLP-Based Intelligent Academic Assistant — Student Starter

A2-Day Hands-On NLP Workshop: **From Classical NLP to Generative AI: Building an Intelligent Academic Assistant**.

This generated starter contains49 intended algorithm TODOs, shared src APIs, ten educational pages, eight notebooks and fictional academic data. It is not the completed reference implementation. Infrastructure/model loading is provided; complete NLP cores in src/.

Begin with [START_HERE](START_HERE.md), [Student Guide](docs/STUDENT_GUIDE.md), [Setup Checklist](workshop/SETUP_CHECKLIST.md) and [Schedule](workshop/WORKSHOP_SCHEDULE.md).

Hindu College of Engineering is fictional. **Synthetic educational data created for NLP workshop purposes.**

Read [Known Limitations](docs/KNOWN_LIMITATIONS.md), [Deployment Guide](docs/DEPLOYMENT_GUIDE.md) and [Portfolio Guide](docs/GITHUB_PORTFOLIO_GUIDE.md). Model caches are not bundled; setup is explicit and free, Qwen optional. Repository license awaits owner selection; no legal license was silently assigned.

```text
python -m workshop.preflight
python -m streamlit run app.py
python -m pytest tests/test_starter_smoke.py
python -m pytest tests/test_preprocessing.py
```

The last command intentionally fails until its TODOs are complete; do not treat incomplete exercises as broken infrastructure. Formula/expected-output tests validate your implementation, not a hidden answer engine. No instructor solutions or measured development reports are included.
''').encode()
    helper_source=(root/'workshop/evaluate_retrieval.py').read_text(encoding='utf-8-sig')
    helper_tree=ast.parse(helper_source)
    helper_parts=[ast.get_source_segment(helper_source,n) for n in helper_tree.body if
                  (isinstance(n,ast.FunctionDef) and n.name=='load_course_documents') or
                  (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DEFAULT_QUERIES' for t in n.targets))]
    contents['workshop/course_data.py']=('\"\"\"Generated provided CSV infrastructure; no evaluator or algorithm solution.\"\"\"\nimport csv\nfrom pathlib import Path\nROOT=Path(__file__).resolve().parents[1]\n\n'+'\n\n'.join(helper_parts)+'\n').encode('utf-8')
    source_digest.update(b'workshop/evaluate_retrieval.py\0'+helper_source.encode('utf-8'))
    contents['START_HERE.md']=('''# Start Here

Build NLP components progressively: Words → Numbers → Meaning → Retrieval → Context → Generation → Intelligent Application.

1. After cloning/unpacking, use64-bit Python3.11 (3.12 also tested). Create/activate `.venv` and install `requirements.txt`; follow workshop/SETUP_CHECKLIST.md for platform commands.
2. Explicitly preload NLTK, MiniLM and task models before class. No paid credentials. Qwen is optional.
3. Run `python -m workshop.preflight`; run `python -m pytest tests/test_starter_smoke.py`.
4. First exercise: `src/preprocessing.py`, **STUDENT TODO1.1**. Preserve signatures/markers and change only the marked core.
5. Run `python -m streamlit run app.py`. Pages catch expected NotImplementedError messages, not unrelated coding bugs.
6. Run `python -m pytest tests/test_preprocessing.py`; later commands are in docs/STUDENT_GUIDE.md. Correctness tests fail while cores are incomplete—this is expected.
7. Experiment in notebook01, inspect each stage, then `git add src/preprocessing.py` and `git commit -m "Add text preprocessing"`. Continue through the schedule and commit journey.

After fixing an incomplete notebook TODO, restart the kernel and run cells again. Some later exercises depend on earlier ones; follow the first displayed TODO instead of changing infrastructure. Ask the instructor for a bounded recovery checkpoint when stuck.
''').encode()
    index_source=(root/'workshop/TODO_INDEX.md').read_text(encoding='utf-8-sig')
    rows=re.findall(r'^\|\s*\d+\.\d+\s*\|.*$',index_source,re.M)
    contents['workshop/TODO_INDEX.md']=('# Student TODO Index\n\n49 unique exercise IDs /50 labelled core regions. Implement these in sequence;1.7 has vocabulary/frequency parts. UI/resources are provided. See docs/STUDENT_GUIDE.md for goals, formulas, examples and hints.\n\n| TODO | Concept | File | Difficulty | Day |\n|---|---|---|---|---|\n'+'\n'.join(rows)+'\n').encode('utf-8')
    contents['docs/STUDENT_GUIDE.md']=('> **Generated student starter:** implementations are intentionally incomplete. Expected outputs below describe completed exercises; reference-state history does not mean your TODOs are already solved. Follow START_HERE.md.\n\n'.encode()+contents['docs/STUDENT_GUIDE.md'])
    student_guide=contents['docs/STUDENT_GUIDE.md'].decode('utf-8')
    student_guide=student_guide.replace('python -m pytest tests/test_transparent_rag.py','python -m pytest tests/test_rag.py')
    student_guide=student_guide.replace('python -m workshop.evaluate_rag','Open the RAG Lab and ask the instructor for recorded evaluation results.')
    student_guide=student_guide.replace('python -m workshop.evaluate_academic_assistant','python -m streamlit run app.py')
    contents['docs/STUDENT_GUIDE.md']=student_guide.encode('utf-8')
    contents['tests/test_starter_smoke.py']=('''"""Setup smoke; independent of unfinished algorithm outputs."""
from pathlib import Path
import ast,json,re
ROOT=Path(__file__).resolve().parents[1]
def test_modules_and_todo_inventory():
    ids=[]
    for file in (ROOT/'src').glob('*.py'):
        text=file.read_text(encoding='utf-8-sig');ast.parse(text)
        ids+=re.findall(r'STUDENT TODO (\\d+\\.\\d+)',text)
    assert len(set(ids))==49
    assert not (ROOT/'docs/INSTRUCTOR_GUIDE.md').exists()
def test_supplied_data_and_pages():
    assert len(list((ROOT/'pages').glob('*.py')))==10
    assert len(list((ROOT/'notebooks').glob('*.ipynb')))==8
    assert len(list((ROOT/'data/academic_documents').glob('*.md')))==14
    assert json.loads((ROOT/'data/university_facts.json').read_text())['institution']['name']=='Hindu College of Engineering'
''').encode()
    output.mkdir(parents=True,exist_ok=True)
    for relative,content in contents.items():
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True).stdout.strip() or None
    if commit=='HEAD':commit=None
    manifest={'format_version':1,'reference_commit':commit,'reference_content_sha256':source_digest.hexdigest(),
        'release_recommendation':'v1.0-workshop','python_support':['3.11','3.12'],'todo_count':49,'transformed_regions':50,
        'file_count':len(contents)+1,'excluded_categories':['instructor guide','evaluators/measurement reports','packaging tooling','local environments/caches/models','reference-only tests'],
        'validation':'Static validation PASS; use the separate validator --runtime for local UI/import/notebook checks.',
        'files':{name:hashlib.sha256(value).hexdigest() for name,value in sorted(contents.items())}}
    (output/'STUDENT_STARTER_MANIFEST.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    from workshop.validate_student_starter import validate_static
    validate_static(output,root)
    return manifest


def write_archive(directory,destination):
    """Sorted files /fixed timestamps; convenient distribution, not remote upload."""
    import zipfile
    directory=Path(directory).resolve();destination=Path(destination).resolve()
    if destination.is_relative_to(directory):raise ValueError('Archive must be outside the starter tree.')
    from workshop.validate_student_starter import validate_static
    validate_static(directory)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for file in sorted(p for p in directory.rglob('*') if p.is_file()):
            entry=zipfile.ZipInfo(file.relative_to(directory).as_posix(),date_time=(1980,1,1,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED;entry.external_attr=0o644 <<16
            archive.writestr(entry,file.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=ROOT/'dist/nlp-academic-assistant-student');parser.add_argument('--archive',type=Path,help='Optional deterministic ZIP; no caches/models.')
    args=parser.parse_args();manifest=build_starter(args.output)
    if args.archive:write_archive(args.output,args.archive)
    print(f"Built{manifest['file_count']} files;49 TODO IDs /50 labelled core regions. {args.output}")

if __name__=='__main__':main()

