"""Instructor-only validation of generated starter, without expecting solved cores."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1]
ABSOLUTE=re.compile(r'(?:[A-Za-z]:[\\/](?:Users|Temp)|/(?:home|Users)/)')


def interfaces(text):
    result={}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            result[node.name]=(ast.dump(node.args,include_attributes=False),ast.dump(node.returns,include_attributes=False) if node.returns else None,ast.get_docstring(node))
    return result


def validate_static(directory,reference=ROOT):
    from workshop.build_student_starter import core_regions
    directory=Path(directory).resolve();reference=Path(reference).resolve()
    manifest=json.loads((directory/'STUDENT_STARTER_MANIFEST.json').read_text(encoding='utf-8'))
    files={p.relative_to(directory).as_posix():p for p in directory.rglob('*') if p.is_file()}
    expected=set(manifest['files'])|{'STUDENT_STARTER_MANIFEST.json'}
    if set(files)!=expected or len(files)!=manifest['file_count']:raise ValueError('Unexpected/missing files or manifest count mismatch.')
    ids=[];regions=0
    for relative,path in files.items():
        if path.is_symlink():raise ValueError('Student output must contain ordinary files, not symlinks.')
        if any(part in {'.git','.cache','.nltk_data','.venv','venv','__pycache__','.pytest_cache','work'} for part in path.relative_to(directory).parts):raise ValueError('Local artifacts in package.')
        if relative!='STUDENT_STARTER_MANIFEST.json' and hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['files'][relative]:raise ValueError('Generated content differs from manifest: '+relative)
        if path.suffix in {'.py','.md','.json','.csv','.ipynb','.txt'}:
            text=path.read_text(encoding='utf-8-sig')
            if ABSOLUTE.search(text):raise ValueError('Development-machine path in '+relative)
        if path.suffix=='.py':ast.parse(text,feature_version=(3,11))
        if relative.startswith('src/') and path.suffix=='.py':
            ids+=re.findall(r'^\s*# STUDENT TODO (\d+\.\d+)',text,re.M)
            for region in core_regions(text):
                body=text[region['start']:region['end']]
                parsed=ast.parse(textwrap.dedent(body))
                if len(parsed.body)!=1 or not isinstance(parsed.body[0],ast.Raise):raise ValueError('Reference body remains in '+relative)
                call=parsed.body[0].exc
                if not isinstance(call,ast.Call) or not isinstance(call.func,ast.Name) or call.func.id!='NotImplementedError':raise ValueError('Missing educational stub.')
                regions+=1
            canonical=reference/relative
            if canonical.is_file() and interfaces(text)!=interfaces(canonical.read_text(encoding='utf-8-sig')):raise ValueError('Function signature/docstring changed: '+relative)
    if len(ids)!=49 or len(set(ids))!=49 or regions!=50:raise ValueError('Expected49 unique TODO headers /50 stub regions.')
    forbidden=['docs/INSTRUCTOR_GUIDE.md','workshop/INSTRUCTOR_START.md','workshop/build_student_starter.py','workshop/validate_student_starter.py']
    if any((directory/p).exists() for p in forbidden):raise ValueError('Instructor-only material leaked.')
    if any(p.name.startswith(('solution_','answers')) or p.name.endswith('_results.json') for p in files.values()):raise ValueError('Solution/development report leaked.')
    return {'status':'PASS','todo_count':49,'stub_regions':50,'file_count':len(files),'interfaces':'Preserved','solutions':'Bounded bodies and instructor material absent'}


def validate_runtime(directory,with_notebooks=True,resource_root=None):
    """Run in a separate interpreter rooted in the starter to avoid src leakage."""
    directory=Path(directory).resolve()
    program='''
import ast,contextlib,importlib,io,json,sys
from pathlib import Path
from unittest.mock import patch
root=Path.cwd();sys.path.insert(0,str(root))
for file in (root/'src').glob('*.py'):
    module=importlib.import_module('src.'+file.stem)
    assert Path(module.__file__).resolve().is_relative_to(root)
if RESOURCE_ROOT:
    from src import sentence_resources,transformer_resources,llm_resources,nltk_resources
    base=Path(RESOURCE_ROOT)
    sentence_resources.MODEL_CACHE=base/'.cache/sentence_transformers'
    transformer_resources.MODEL_CACHE=base/'.cache/transformer_tasks'
    llm_resources.MODEL_CACHE=base/'.cache/local_llm'
    nltk_resources.LOCAL_DATA_DIR=base/'.nltk_data'
    nltk_resources.configure_resource_path()
from streamlit.testing.v1 import AppTest
pages=[]
with patch('requests.sessions.Session.request',side_effect=AssertionError('No downloads in starter')),patch('nltk.download',side_effect=AssertionError('No NLTK downloads')):
    for file in [root/'app.py',*sorted((root/'pages').glob('*.py'))]:
        app=AppTest.from_file(str(root/'app.py')).run(timeout=120)
        if file.name!='app.py':app.switch_page('pages/'+file.name).run(timeout=120)
        assert not app.exception,(file.name,app.exception)
        messages=[str(item.value) for category in [app.info,app.warning,app.error,app.caption] for item in category]
        pages.append({'page':file.name,'status':'PASS','TODO_guidance':any('TODO' in m for m in messages),'resource_guidance':any('model' in m.lower() or 'preload' in m.lower() for m in messages)})
    notebooks=[]
    if NOTEBOOKS:
        for file in sorted((root/'notebooks').glob('*.ipynb')):
            book=json.loads(file.read_text(encoding='utf-8'));namespace={'__name__':'__main__','display':lambda value:None};messages=io.StringIO()
            with contextlib.redirect_stdout(messages):
                for cell in book['cells']:
                    if cell['cell_type']=='code':exec(compile(''.join(cell['source']),file.name,'exec'),namespace)
            assert namespace.get('_workshop_todo_incomplete'),file.name
            notebooks.append({'notebook':file.name,'status':'Expected TODO pause','guidance':'STUDENT TODO' in messages.getvalue()})
print('STARTER_RUNTIME_JSON:'+json.dumps({'imports':'PASS','pages':pages,'notebooks':notebooks}))
'''.replace('if NOTEBOOKS:', 'if '+str(bool(with_notebooks))+':').replace('RESOURCE_ROOT', repr(str(Path(resource_root).resolve())) if resource_root else 'None')
    environment=os.environ.copy();environment.pop('PYTHONPATH',None)
    environment['HF_HUB_OFFLINE']='1';environment['TRANSFORMERS_OFFLINE']='1';environment['PYTHONDONTWRITEBYTECODE']='1'
    run=subprocess.run([sys.executable,'-B','-c',program],cwd=directory,env=environment,capture_output=True,text=True,encoding='utf-8',timeout=360)
    if run.returncode:raise ValueError('Starter runtime validation failed:\n'+run.stdout[-3000:]+'\n'+run.stderr[-4000:])
    line=next(line for line in run.stdout.splitlines() if line.startswith('STARTER_RUNTIME_JSON:'))
    return json.loads(line.split(':',1)[1])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--runtime',action='store_true');parser.add_argument('--resource-root',type=Path,help='Explicit preloaded resource tree for validation only; never copied into starter.');parser.add_argument('--report',type=Path)
    args=parser.parse_args();report={'static':validate_static(args.directory)}
    if args.runtime:report['runtime']=validate_runtime(args.directory,resource_root=args.resource_root)
    if args.report:args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

