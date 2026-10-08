"""Task18 packaging invariants; no change to frozen NLP behavior."""
import ast
import hashlib
import json
from pathlib import Path
import pytest
from workshop.build_student_starter import core_regions,strip_cores,build_starter,safe_output,scaffold_page
from workshop.validate_student_starter import validate_static,interfaces
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('ending',['\n','\r\n'])
def test_marker_parser_preserves_signatures_and_guidance(ending):
    source=ending.join(['def lower(text: str) -> str:', '    """Normalize case only."""',
                       '    # STUDENT TODO BOUNDARY 1.1 — Lowercase', '    # Difficulty: ★ Guided',
                       '    # Goal: normalize case. Hint: string method.', '    # BEGIN STUDENT CORE 1.1',
                       '    return text.lower()', '    # END STUDENT CORE 1.1', ''])
    student,ids=strip_cores(source)
    assert ids==['1.1'] and interfaces(source)==interfaces(student)
    assert 'return text.lower()' not in student and 'NotImplementedError' in student and 'Hint:' in student

@pytest.mark.parametrize('source',[
 '# BEGIN STUDENT CORE 1.1\na=1\n', '# END STUDENT CORE 1.1\n',
 '# BEGIN STUDENT CORE 1.1\n# END STUDENT CORE 1.2\n',
 '# BEGIN STUDENT CORE 1.1\n# BEGIN STUDENT CORE 1.2\n# END STUDENT CORE 1.2\n# END STUDENT CORE 1.1\n',
 '# BEGIN STUDENT CORE 1.1\n# END STUDENT CORE 1.1\n# BEGIN STUDENT CORE 1.1\n# END STUDENT CORE 1.1\n',
 ' # BEGIN STUDENT CORE 1.1\n# END STUDENT CORE 1.1\n',
])
def test_bad_markers_rejected(source):
    with pytest.raises(ValueError):core_regions(source)

@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    directory=tmp_path_factory.mktemp('starter')/'student'
    manifest=build_starter(directory)
    return directory,manifest


def test_all_49_transformations_manifest_and_interfaces(bundle):
    directory,manifest=bundle
    result=validate_static(directory)
    assert result['todo_count']==manifest['todo_count']==49 and manifest['transformed_regions']==50
    assert manifest['file_count']==len(list(p for p in directory.rglob('*') if p.is_file()))
    assert len(manifest['reference_content_sha256'])==64
    for path in (directory/'src').glob('*.py'):
        assert interfaces(path.read_text(encoding='utf-8-sig'))==interfaces((ROOT/'src'/path.name).read_text(encoding='utf-8-sig'))


def test_instructor_exclusions_and_student_scaffolding(bundle):
    directory,_=bundle
    assert not (directory/'docs/INSTRUCTOR_GUIDE.md').exists()
    assert not list((directory/'workshop').glob('evaluate*.py'))
    assert not list((directory/'workshop').glob('*results.json'))
    for name in ['START_HERE.md','docs/STUDENT_GUIDE.md','workshop/preflight.py','data/university_facts.json','tests/test_starter_smoke.py']:
        assert (directory/name).exists()
    assert not (directory/'.cache').exists() and not (directory/'.venv').exists()


def test_deterministic_rebuild_and_no_source_mutation(bundle,tmp_path):
    directory,manifest=bundle
    second=tmp_path/'second';again=build_starter(second)
    assert manifest==again
    assert {p.relative_to(directory).as_posix():p.read_bytes() for p in directory.rglob('*') if p.is_file()}=={p.relative_to(second).as_posix():p.read_bytes() for p in second.rglob('*') if p.is_file()}
    assert build_starter(directory)==manifest
    assert 'return text.lower()' in (ROOT/'src/preprocessing.py').read_text()


def test_modified_student_work_not_overwritten(tmp_path):
    directory=tmp_path/'student';build_starter(directory)
    file=directory/'src/preprocessing.py';file.write_text(file.read_text()+'\n# My student work\n')
    with pytest.raises(ValueError,match='changed'):build_starter(directory)
    assert 'My student work' in file.read_text()

@pytest.mark.parametrize('path',['src','data','pages','docs',''])
def test_refuse_canonical_output_targets(path):
    with pytest.raises(ValueError):safe_output(ROOT,ROOT/path)


def test_tampered_manifest_and_reference_body_leak_detected(tmp_path):
    directory=tmp_path/'student';build_starter(directory)
    file=directory/'src/preprocessing.py';file.write_bytes((ROOT/'src/preprocessing.py').read_bytes())
    with pytest.raises(ValueError):validate_static(directory)


def test_generated_page_catches_only_expected_incompleteness():
    wrapped=scaffold_page('raise NotImplementedError("TODO")\n')
    tree=ast.parse(wrapped)
    assert isinstance(tree.body[-1],ast.Try)
    assert [h.type.id for h in tree.body[-1].handlers]==['NotImplementedError']


def test_student_notebooks_pause_without_solution_cells(bundle):
    directory,_=bundle
    for file in (directory/'notebooks').glob('*.ipynb'):
        notebook=json.loads(file.read_text())
        cells=[c for c in notebook['cells'] if c['cell_type']=='code']
        assert all(c['execution_count'] is None and c['outputs']==[] for c in cells)
        assert any('_workshop_todo_incomplete' in ''.join(c['source']) for c in cells)
        assert not any('_results.json' in ''.join(c['source']) for c in cells)

def test_archive_equivalent_bytes_and_hygiene(bundle,tmp_path):
    from workshop.build_student_starter import write_archive
    import zipfile
    directory,_=bundle
    first,second=tmp_path/'first.zip',tmp_path/'second.zip'
    write_archive(directory,first);write_archive(directory,second)
    assert first.read_bytes()==second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        assert 'START_HERE.md' in archive.namelist()
        assert not any('.cache/' in name or 'INSTRUCTOR_GUIDE' in name for name in archive.namelist())


def test_absolute_path_rejected_even_with_updated_hash(tmp_path):
    directory=tmp_path/'student';build_starter(directory)
    file=directory/'START_HERE.md';file.write_text(file.read_text(encoding='utf-8')+'\nC:/Users/developer/private\n',encoding='utf-8')
    manifest_path=directory/'STUDENT_STARTER_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));manifest['files']['START_HERE.md']=hashlib.sha256(file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='Development-machine path'):validate_static(directory)


def test_reference_body_detected_even_with_updated_hash(tmp_path):
    directory=tmp_path/'student';build_starter(directory)
    file=directory/'src/preprocessing.py';text=file.read_text(encoding='utf-8')
    from workshop.build_student_starter import core_regions
    region=core_regions(text)[0]
    text=text[:region['start']]+'\n    return text.lower()\n'+text[region['end']:]
    file.write_text(text,encoding='utf-8')
    manifest_path=directory/'STUDENT_STARTER_MANIFEST.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files']['src/preprocessing.py']=hashlib.sha256(file.read_bytes()).hexdigest();manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError,match='Reference body remains'):validate_static(directory)
