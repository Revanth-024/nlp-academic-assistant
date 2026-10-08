"""Validate authored dataset content against facts and held-out annotations."""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DOCUMENT_NAMES = [
    "01_student_academic_regulations.md", "02_btech_cse_curriculum.md",
    "03_semester_course_structure.md", "04_nlp_syllabus.md",
    "05_machine_learning_syllabus.md", "06_deep_learning_syllabus.md",
    "07_academic_calendar.md", "08_attendance_policy.md",
    "09_examination_regulations.md", "10_laboratory_guidelines.md",
    "11_placement_faq.md", "12_student_handbook.md",
    "13_course_prerequisites.md", "14_faculty_course_information.md",
]
QUESTION_PATTERN = re.compile(
    r"^## (Q\d+) — (\d+) marks\s*\n\n(.*?)(?=\n## |\Z)",
    re.MULTILINE | re.DOTALL,
)


def load_json(relative):
    return json.loads((DATA / relative).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def facts():
    return load_json("university_facts.json")


@pytest.fixture(scope="module")
def manifest():
    return load_json("document_manifest.json")


@pytest.fixture(scope="module")
def metadata():
    return load_json("question_papers/question_metadata.json")


@pytest.fixture(scope="module")
def benchmarks():
    return load_json("retrieval_benchmarks.json")["benchmarks"]


def document_text(filename):
    return (DATA / "academic_documents" / filename).read_text(encoding="utf-8")


def table_rows(text):
    """Read simple authored Markdown tables without inspecting prose wording."""
    rows = []
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if not all(re.fullmatch(r"[:\- ]+", cell) for cell in cells):
                rows.append(cells)
    return rows


def section(text, heading):
    pattern = rf"^{re.escape(heading)}\s*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    assert match, f"Missing section: {heading}"
    return match.group(1)


def resolve_pointer(value, pointer):
    assert pointer.startswith("/")
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


@pytest.mark.parametrize("filename", DOCUMENT_NAMES)
def test_required_academic_document_exists_and_is_fictional(facts, filename):
    text = document_text(filename)
    assert text.strip() and text.startswith("# ")
    assert "\n## " in text
    assert facts["synthetic_data"]["required_document_statement"] in text[:600]
    assert facts["institution"]["name"] in text
    assert re.search(r"\bfictional\b", text[:600], re.IGNORECASE)


def test_manifest_covers_every_document_once(manifest):
    academic = manifest["academic_documents"]
    expected = {f"academic_documents/{name}" for name in DOCUMENT_NAMES}
    assert len(academic) == len(expected)
    assert {d["filename"] for d in academic} == expected
    assert {p.name for p in (DATA / "academic_documents").glob("*.md")} == set(DOCUMENT_NAMES)
    inventory = academic + manifest["question_papers"]
    assert len({d["document_id"] for d in inventory}) == len(inventory)
    assert len({d["filename"] for d in inventory}) == len(inventory)
    for item in inventory:
        path = DATA / item["filename"]
        assert path.resolve().is_relative_to(DATA.resolve())
        assert path.is_file()
        assert item["title"] and item["category"] and item["description"]
        assert item["canonical_fact_areas"] and item["intended_workshop_uses"]
        assert path.read_text(encoding="utf-8").splitlines()[0] == f"# {item['title']}"
        assert "content" not in item


def test_manifest_fact_areas_resolve(facts, manifest):
    for item in manifest["academic_documents"] + manifest["question_papers"]:
        for pointer in item["canonical_fact_areas"]:
            assert resolve_pointer(facts, pointer) is not None


@pytest.mark.parametrize("filename,credit_column,mode_column", [
    ("02_btech_cse_curriculum.md", 2, "course_type"),
    ("03_semester_course_structure.md", 2, "delivery"),
])
def test_full_curriculum_tables_match_canonical_courses(facts, filename, credit_column, mode_column):
    text = document_text(filename)
    catalog = {c["code"]: c for c in facts["courses"]}
    observed = set()
    for semester in facts["program"]["semesters"]:
        heading = f"### Semester {semester['number']}"
        match = re.search(rf"^{re.escape(heading)}\s*\n(.*?)(?=^### |^## |\Z)", text, re.MULTILINE | re.DOTALL)
        assert match
        rows = [r for r in table_rows(match.group(1)) if r[0] in catalog]
        assert [r[0] for r in rows] == semester["course_codes"]
        for row in rows:
            course = catalog[row[0]]
            assert row[1] == course["name"]
            assert int(row[credit_column]) == course["credits"]
            assert row[3] == course[mode_column]
            assert course["semester"] == semester["number"]
            observed.add(row[0])
    assert observed == set(catalog)


@pytest.mark.parametrize("filename,course_code", [
    ("04_nlp_syllabus.md", "HCE-CSE603"),
    ("05_machine_learning_syllabus.md", "HCE-CSE502"),
    ("06_deep_learning_syllabus.md", "HCE-CSE701"),
])
def test_syllabus_identity_prerequisites_and_faculty(facts, filename, course_code):
    course = next(c for c in facts["courses"] if c["code"] == course_code)
    text = document_text(filename)
    fields = {row[0]: row[1] for row in table_rows(section(text, "## Course information"))}
    assert fields["Course code"] == course["code"]
    assert fields["Course name"] == course["name"]
    assert int(fields["Semester"]) == course["semester"]
    assert int(fields["Credits"]) == course["credits"]
    assert fields["Academic year"] == facts["institution"]["academic_year"]
    roster = {f["id"]: f for f in facts["faculty"]}
    assert fields["Faculty"] == "; ".join(roster[f]["name"] for f in course["faculty_ids"])
    prerequisites = section(text, "## Prerequisites")
    for prerequisite in course["prerequisites"]:
        assert prerequisite in prerequisites
    assert course["laboratory_course_code"] in prerequisites
    assert "corequisite" in prerequisites.lower()
    assert course["description"] in section(text, "## Course description")


def test_nlp_five_unit_syllabus_content(facts):
    course = next(c for c in facts["courses"] if c["code"] == "HCE-CSE603")
    text = document_text("04_nlp_syllabus.md")
    headings = re.findall(r"^### Unit (\d+) — (.+)$", text, re.MULTILINE)
    assert headings == [(str(u["number"]), u["title"]) for u in course["units"]]
    topic_names = {t["id"]: t["name"] for t in facts["nlp_topic_taxonomy"]}
    for unit in course["units"]:
        match = re.search(rf"^### Unit {unit['number']} — .*?\n(.*?)(?=^### |^## |\Z)", text, re.MULTILINE | re.DOTALL)
        assert match
        for topic in unit["topics"]:
            assert topic in match.group(1)
        for topic_id in unit["topic_ids"]:
            assert topic_names[topic_id] in match.group(1)


def test_prerequisite_register_distinguishes_corequisites(facts):
    rows = [r for r in table_rows(document_text("13_course_prerequisites.md")) if r[0].startswith(tuple(c["name"] for c in facts["courses"])) and len(r) == 4]
    catalog = {c["code"]: c for c in facts["courses"]}
    assert len(rows) == len(catalog)
    observed = set()
    for row in rows:
        course_code = re.search(r"\((HCE-CSE\d{3})\)$", row[0]).group(1)
        course = catalog[course_code]
        observed.add(course_code)
        assert int(row[1]) == course["semester"]
        assert re.findall(r"HCE-CSE\d{3}", row[2]) == course["prerequisites"]
        assert re.findall(r"HCE-CSE\d{3}", row[3]) == course["corequisite_course_codes"]
    assert observed == set(catalog)


def test_faculty_roster_and_assignments_match_contract(facts):
    text = document_text("14_faculty_course_information.md")
    catalog = {c["code"]: c for c in facts["courses"]}
    rows = [r for r in table_rows(text) if r[0].startswith("FAC-")]
    assert len(rows) == len(facts["faculty"])
    for member in facts["faculty"]:
        row = next(r for r in rows if r[0] == member["id"])
        assert row[1:4] == [member["name"], member["designation"], member["department"]]
        heading = f"### {member['name']} — {member['id']}"
        match = re.search(rf"^{re.escape(heading)}\s*\n(.*?)(?=^### |^## |\Z)", text, re.MULTILINE | re.DOTALL)
        assert match
        assignments = [r for r in table_rows(match.group(1)) if r[0] in catalog]
        assert [r[0] for r in assignments] == member["courses_taught"]
        for course_row in assignments:
            course = catalog[course_row[0]]
            assert course_row[1] == course["name"]
            assert int(course_row[2]) == course["semester"]
            assert int(course_row[3]) == course["credits"]
            assert course_row[4] == course["delivery"]


def test_no_unknown_course_codes_in_academic_documents(facts):
    valid = {c["code"] for c in facts["courses"]}
    for filename in DOCUMENT_NAMES:
        assert set(re.findall(r"HCE-CSE\d{3}", document_text(filename))) <= valid


def test_attendance_threshold_is_consistent(facts):
    for filename in [DOCUMENT_NAMES[i] for i in (0, 7, 8, 9)]:
        text = document_text(filename)
        assert {float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", text)} == {facts["attendance_policy"]["minimum_percent"]}
    policy = document_text("08_attendance_policy.md").lower()
    assert "no condonation" in policy
    assert "re-enrollment" in policy
    assert "unrounded" in policy


def test_examination_values_and_grades_are_consistent(facts):
    scheme = facts["examination"]["schemes"]["course_standard"]
    text = document_text("09_examination_regulations.md")
    assert re.search(rf"[Ii]nternal assessment contributes {scheme['internal_assessment_marks']} marks", text)
    assert re.search(rf"end-semester evaluation contributes {scheme['end_semester_marks']} marks", text)
    assert re.search(rf"making {scheme['total_marks']} marks", text)
    assert re.search(rf"at least {scheme['minimum_total_to_pass']} total marks", text)
    assert re.search(rf"at least {scheme['minimum_end_semester_to_pass']} end-semester marks", text)
    assert "no separate internal minimum" in text.lower()
    grade_rows = table_rows(section(text, "## Absolute grading"))[1:]
    assert grade_rows == [[f"{b['minimum_marks']}–{b['maximum_marks']}", b["grade"], str(b["grade_points"])] for b in facts["examination"]["grading"]["bands"]]
    for filename in ["01_student_academic_regulations.md", "04_nlp_syllabus.md", "05_machine_learning_syllabus.md", "06_deep_learning_syllabus.md", "10_laboratory_guidelines.md"]:
        content = document_text(filename)
        assert re.search(rf"{scheme['internal_assessment_marks']} internal(?: assessment)? marks", content)
        assert re.search(rf"{scheme['end_semester_marks']} end-semester marks", content)
        assert re.search(rf"at least {scheme['minimum_total_to_pass']} total marks", content)
        assert re.search(rf"at least {scheme['minimum_end_semester_to_pass']} end-semester marks", content)


def test_calendar_contains_only_and_all_approved_dates(facts):
    def dates(value):
        if isinstance(value, dict):
            return set().union(*(dates(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(*(dates(v) for v in value))
        if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return {value}
        return set()

    approved = dates(facts["academic_calendar"])
    text = document_text("07_academic_calendar.md")
    assert set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text)) == approved
    for filename in DOCUMENT_NAMES:
        assert set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", document_text(filename))) <= approved
    nlp_date = facts["academic_calendar"]["terms"][1]["course_examinations"]["HCE-CSE603"]
    assert nlp_date in section(text, "## Specific NLP examination")
    assert "HCE-CSE603" in section(text, "## Specific NLP examination")


def test_course_description_csv_matches_canonical_values(facts):
    with (DATA / "search_corpus/course_descriptions.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert set(reader.fieldnames) == {"course_code", "course_name", "semester", "credits", "description", "keywords"}
        rows = list(reader)
    assert 15 <= len(rows) <= 20
    assert len({r["course_code"] for r in rows}) == len(rows)
    catalog = {c["code"]: c for c in facts["courses"]}
    required = {"Natural Language Processing", "Machine Learning", "Deep Learning", "Artificial Intelligence", "Database Management Systems", "Computer Networks", "Operating Systems", "Data Structures", "Software Engineering", "Probability and Statistics"}
    assert required <= {r["course_name"] for r in rows}
    for row in rows:
        course = catalog[row["course_code"]]
        assert row["course_name"] == course["name"]
        assert int(row["semester"]) == course["semester"]
        assert int(row["credits"]) == course["credits"]
        assert row["description"] == course["description"]
        assert row["keywords"].split("; ") == course["keywords"]
        assert row["description"].strip() and row["keywords"].strip()


@pytest.mark.parametrize("filename,collection", [
    ("preprocessing_examples.json", "examples"),
    ("bow_examples.json", "collections"),
    ("similarity_examples.json", "pairs"),
])
def test_example_inputs_are_valid_and_have_no_solutions(facts, filename, collection):
    examples = load_json(f"examples/{filename}")
    assert examples["is_fictional"] is True
    assert examples["disclaimer"] == facts["synthetic_data"]["required_document_statement"]
    entries = examples[collection]
    assert entries and len({e["id"] for e in entries}) == len(entries)
    for entry in entries:
        assert not {"answer", "solution", "expected_output", "vector", "score"} & set(entry)
        if collection == "examples":
            assert entry["text"].strip()
        elif collection == "collections":
            assert len(entry["texts"]) >= 2 and all(t.strip() for t in entry["texts"])
        else:
            assert entry["text_a"].strip() and entry["text_b"].strip()


def test_four_papers_and_question_text_match_ground_truth(facts, manifest, metadata):
    design = facts["question_paper_design"]
    paper_items = manifest["question_papers"]
    assert {p["document_id"] for p in paper_items} == set(design["paper_ids"])
    assert len(paper_items) == 4
    assert set(metadata["paper_files"]) == set(design["paper_ids"])
    assert len(metadata["questions"]) == 48
    keys = [(q["paper_id"], q["question_id"]) for q in metadata["questions"]]
    assert len(keys) == len(set(keys))
    valid_topics = {t["id"] for t in facts["nlp_topic_taxonomy"]}
    metadata_by_key = {(q["paper_id"], q["question_id"]): q for q in metadata["questions"]}
    all_texts = []
    for item in paper_items:
        paper_id = item["document_id"]
        assert metadata["paper_files"][paper_id] == item["filename"]
        assert Path(item["filename"]).name == f"nlp_question_paper_{design['paper_ids'].index(paper_id) + 1:02}.md"
        text = (DATA / item["filename"]).read_text(encoding="utf-8")
        assert facts["synthetic_data"]["required_document_statement"] in text[:600]
        assert "fictional" in text[:600].lower()
        assert paper_id in text and design["course_code"] in text
        assert "primary_topic_id" not in text and "secondary_topic_ids" not in text
        questions = QUESTION_PATTERN.findall(text)
        assert [q[0] for q in questions] == [f"Q{i}" for i in range(1, 13)]
        assert len(questions) == design["primary_topic_slots_per_paper"]
        assert sum(int(q[1]) for q in questions) == facts["examination"]["schemes"]["course_standard"]["end_semester_marks"]
        for qid, marks, body in questions:
            annotated = metadata_by_key[(paper_id, qid)]
            assert annotated["marks"] == int(marks) > 0
            assert annotated["question_text"] == body.strip()
            assert annotated["primary_topic_id"] in valid_topics
            assert set(annotated["secondary_topic_ids"]) <= valid_topics - {annotated["primary_topic_id"]}
            assert len(annotated["secondary_topic_ids"]) == len(set(annotated["secondary_topic_ids"]))
            all_texts.append(body.strip())
    assert len(all_texts) == len(set(all_texts))
    assert set(p for p, _ in keys) == set(design["paper_ids"])


def test_primary_topic_targets_match_per_paper_and_aggregate(facts, metadata):
    targets = facts["question_paper_design"]["topic_targets"]
    aggregate = Counter(q["primary_topic_id"] for q in metadata["questions"])
    assert aggregate == {t["topic_id"]: sum(t["question_counts_by_paper"].values()) for t in targets}
    for paper_id in facts["question_paper_design"]["paper_ids"]:
        actual = Counter(q["primary_topic_id"] for q in metadata["questions"] if q["paper_id"] == paper_id)
        assert actual == Counter({t["topic_id"]: t["question_counts_by_paper"][paper_id] for t in targets})


def test_question_wording_includes_variety_and_implicit_topics(metadata):
    texts = [q["question_text"] for q in metadata["questions"]]
    assert len({text.split()[0] for text in texts}) >= 7
    attention = [q["question_text"] for q in metadata["questions"] if q["primary_topic_id"] == "attention"]
    assert any("attention" not in text.lower() for text in attention)
    assert any("term frequency inverse document frequency" in text.lower() for text in texts)


def test_ground_truth_usage_is_explicit(metadata):
    assert "ground truth" in metadata["usage"].lower()
    assert "independently" in metadata["usage"].lower()
    benchmark_file = load_json("retrieval_benchmarks.json")
    assert "document content" in benchmark_file["usage"].lower()
    design_doc = (ROOT / "docs/SYNTHETIC_DATA_DESIGN.md").read_text(encoding="utf-8")
    assert "question_metadata.json" in design_doc
    assert "must not" in design_doc.lower()


def test_benchmarks_have_valid_sources_references_and_types(facts, manifest, benchmarks):
    inventory = {d["document_id"]: d for d in manifest["academic_documents"] + manifest["question_papers"]}
    assert 15 <= len(benchmarks) <= 20
    assert len({b["benchmark_id"] for b in benchmarks}) == len(benchmarks)
    assert len({b["query"] for b in benchmarks}) == len(benchmarks)
    required_types = {"direct_fact", "lexical_search", "semantic_search", "comparison", "cross_document", "summarization", "question_paper_analysis", "question_filtering"}
    assert required_types <= {b["type"] for b in benchmarks}
    course_ids = {c["code"] for c in facts["courses"]}
    topic_ids = {t["id"] for t in facts["nlp_topic_taxonomy"]}
    for benchmark in benchmarks:
        assert benchmark["query"].strip() and benchmark["teaching_purpose"].strip()
        ids = benchmark["expected_document_ids"]
        assert ids and len(ids) == len(set(ids))
        assert set(ids) <= set(inventory)
        assert "expected_chunk_ids" not in benchmark
        for reference in benchmark["expected_canonical_refs"]:
            if reference["kind"] == "course":
                assert reference["id"] in course_ids
            elif reference["kind"] == "topic":
                assert reference["id"] in topic_ids
            else:
                assert reference["kind"] == "fact"
                assert resolve_pointer(facts, reference["id"]) is not None


def test_canonical_benchmarks_are_preserved(facts, benchmarks):
    mapped = {b["benchmark_id"]: b for b in benchmarks}
    for canonical in facts["retrieval_design"]["benchmarks"]:
        benchmark = mapped[canonical["id"]]
        assert benchmark["query"] == canonical["query"]
        assert benchmark["type"] == canonical["query_type"]
        assert benchmark["expected_canonical_refs"] == canonical["expected_refs"]
        assert benchmark["semantic_vs_keyword"] == canonical["semantic_lexical_contrast"]


def test_benchmark_source_coverage_and_semantic_contrasts(facts, manifest, benchmarks):
    inventory = {d["document_id"]: d for d in manifest["academic_documents"] + manifest["question_papers"]}
    contrast = [b for b in benchmarks if b["semantic_vs_keyword"]]
    assert len(contrast) >= 4
    for benchmark in contrast:
        assert benchmark["semantic_rationale"]
        for doc_id in benchmark["expected_document_ids"]:
            text = (DATA / inventory[doc_id]["filename"]).read_text(encoding="utf-8")
            assert benchmark["query"].lower() not in text.lower()
    mapped = {b["benchmark_id"]: b for b in benchmarks}
    assert {inventory[i]["category"] for i in mapped["RET-09"]["expected_document_ids"]} == {"academic_calendar", "course_syllabus"}
    assert set(mapped["RET-14"]["expected_document_ids"]) == set(facts["question_paper_design"]["paper_ids"])
    for bid in ("RET-03", "RET-05", "RET-06"):
        course_code = next(r["id"] for r in mapped[bid]["expected_canonical_refs"] if r["kind"] == "course")
        course = next(c for c in facts["courses"] if c["code"] == course_code)
        sources = "\n".join((DATA / inventory[i]["filename"]).read_text(encoding="utf-8") for i in mapped[bid]["expected_document_ids"])
        assert course["description"] in sources
    assert "HCE-DOC06" in mapped["RET-16"]["expected_document_ids"]
    assert "HCE-DOC04" in mapped["RET-17"]["expected_document_ids"]
    assert set(mapped["RET-18"]["expected_document_ids"]) == set(facts["question_paper_design"]["paper_ids"])
