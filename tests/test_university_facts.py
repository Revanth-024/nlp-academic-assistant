"""Check canonical academic facts independently of the generated corpus."""
import json
from datetime import date
from pathlib import Path

import pytest

FACTS_PATH = Path(__file__).resolve().parents[1] / "data" / "university_facts.json"


def reject_duplicate_keys(pairs):
    """JSON parsers normally hide conflicting repeated object keys."""
    result = {}
    for key, value in pairs:
        assert key not in result, f"Duplicate JSON key: {key}"
        result[key] = value
    return result


@pytest.fixture(scope="module")
def facts():
    return json.loads(
        FACTS_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )


def resolve_pointer(facts, pointer):
    """Resolve benchmark references without depending on a third-party schema tool."""
    assert pointer.startswith("/")
    value = facts
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def test_institution_and_disclaimer(facts):
    assert facts["institution"]["name"] == "Hindu College of Engineering"
    assert facts["institution"]["is_fictional"] is True
    assert facts["synthetic_data"]["is_fictional"] is True
    assert "fictional" in facts["synthetic_data"]["required_institution_disclaimer"]
    assert facts["synthetic_data"]["required_document_statement"] == (
        "Synthetic educational data created for NLP workshop purposes."
    )
    assert facts["institution"]["location"]["town"].strip()


def test_complete_program_structure(facts):
    program = facts["program"]
    assert program["name"] == "B.Tech Computer Science and Engineering"
    assert program["duration_years"] == 4
    assert program["number_of_semesters"] == 8
    semesters = program["semesters"]
    assert sorted(s["number"] for s in semesters) == list(range(1, 9))
    catalog = {c["code"]: c for c in facts["courses"]}
    listed = []
    for semester in semesters:
        assert semester["course_codes"]
        for course_code in semester["course_codes"]:
            assert catalog[course_code]["semester"] == semester["number"]
            listed.append(course_code)
    assert len(listed) == len(set(listed))
    assert set(listed) == set(catalog)


def test_course_identifiers_and_fields(facts):
    courses = facts["courses"]
    assert len({c["code"] for c in courses}) == len(courses)
    assert len({c["name"] for c in courses}) == len(courses)
    deliveries = {"theory": "theory", "elective": "theory", "lab": "laboratory",
                  "project": "practical", "seminar": "practical"}
    for course in courses:
        assert course["code"].strip() and course["name"].strip()
        assert type(course["semester"]) is int
        assert 1 <= course["semester"] <= 8
        assert type(course["credits"]) in (int, float)
        assert 0 < course["credits"] <= 10
        assert course["delivery"] == deliveries[course["course_type"]]
        assert course["assessment_scheme_id"] in facts["examination"]["schemes"]


def test_canonical_search_descriptions_and_keywords(facts):
    described = [c for c in facts["courses"] if "description" in c]
    assert described
    assert len({c["description"] for c in described}) == len(described)
    for course in described:
        assert course["description"].strip()
        assert course["keywords"]
        assert len(course["keywords"]) == len(set(course["keywords"]))
        assert all(keyword.strip() for keyword in course["keywords"])


def test_prerequisites_are_existing_earlier_courses(facts):
    courses = {c["code"]: c for c in facts["courses"]}
    for course in courses.values():
        assert len(course["prerequisites"]) == len(set(course["prerequisites"]))
        for prerequisite in course["prerequisites"]:
            assert prerequisite in courses
            assert courses[prerequisite]["semester"] < course["semester"]
        for corequisite in course["corequisite_course_codes"]:
            assert corequisite in courses
            assert courses[corequisite]["semester"] == course["semester"]
            assert corequisite != course["code"]


@pytest.mark.parametrize("name,code,semester", [
    ("Machine Learning", "HCE-CSE502", 5),
    ("Natural Language Processing", "HCE-CSE603", 6),
    ("Deep Learning", "HCE-CSE701", 7),
])
def test_canonical_ai_course_mapping(facts, name, code, semester):
    matches = [c for c in facts["courses"] if c["name"] == name]
    assert len(matches) == 1
    course = matches[0]
    assert course["code"] == code
    assert course["semester"] == semester
    assert course["description"].strip()
    assert course["learning_objectives"]
    assert course["faculty_ids"]


def test_ai_dependencies_and_laboratory_relationships(facts):
    courses = {c["code"]: c for c in facts["courses"]}
    assert "HCE-CSE502" in courses["HCE-CSE701"]["prerequisites"]
    assert {"HCE-CSE501", "HCE-CSE502"} <= set(courses["HCE-CSE603"]["prerequisites"])
    for course in courses.values():
        if "laboratory_course_code" in course:
            lab = courses[course["laboratory_course_code"]]
            assert lab["course_type"] == "lab"
            assert lab["semester"] == course["semester"]
            assert course["code"] in lab["corequisite_course_codes"]
    assert courses["HCE-CSE701"]["major_topics"]


def test_nlp_syllabus_has_five_units_covering_taxonomy(facts):
    nlp = next(c for c in facts["courses"] if c["name"] == "Natural Language Processing")
    assert [u["number"] for u in nlp["units"]] == [1, 2, 3, 4, 5]
    topic_ids = {t["id"] for t in facts["nlp_topic_taxonomy"]}
    covered = set()
    for unit in nlp["units"]:
        assert unit["title"].strip() and unit["topics"]
        assert set(unit["topic_ids"]) <= topic_ids
        covered.update(unit["topic_ids"])
    assert covered == topic_ids


def test_unit_four_distinguishes_bert_and_gpt_without_new_primary_topics(facts):
    nlp = next(c for c in facts["courses"] if c["code"] == "HCE-CSE603")
    unit4 = nlp["units"][3]
    assert unit4["number"] == 4
    assert "GPT" in unit4["topics"]
    assert "Masked language modeling" in unit4["topics"]
    assert "Autoregressive / next-token language modeling" in unit4["topics"]
    language_models = next(t for t in facts["nlp_topic_taxonomy"] if t["id"] == "language_models")
    assert "GPT" in language_models["aliases"]
    models = {m["name"]: m for m in language_models["subtopics"]}
    assert models["BERT"]["orientation"] == "encoder-oriented"
    assert models["BERT"]["objective"] == "masked language modeling"
    assert models["GPT"]["orientation"] == "decoder-oriented"
    assert models["GPT"]["objective"] == "autoregressive / next-token prediction"
    assert {t["id"] for t in facts["nlp_topic_taxonomy"]} == {t["topic_id"] for t in facts["question_paper_design"]["topic_targets"]}


def test_faculty_relationships_are_bidirectional(facts):
    faculty = facts["faculty"]
    assert len({f["id"] for f in faculty}) == len(faculty)
    assert len({f["name"] for f in faculty}) == len(faculty)
    roster = {f["id"]: f for f in faculty}
    courses = {c["code"]: c for c in facts["courses"]}
    for member in faculty:
        assert member["is_fictional"] is True
        assert member["department"] == facts["program"]["department"]
        assert member["designation"] and member["specializations"]
        assert len(member["courses_taught"]) == len(set(member["courses_taught"]))
        for course_code in member["courses_taught"]:
            assert course_code in courses
            assert member["id"] in courses[course_code]["faculty_ids"]
    for course in courses.values():
        assert len(course["faculty_ids"]) == len(set(course["faculty_ids"]))
        for faculty_id in course["faculty_ids"]:
            assert faculty_id in roster
            assert course["code"] in roster[faculty_id]["courses_taught"]


def test_one_attendance_minimum_shared_with_labs(facts):
    attendance = facts["attendance_policy"]
    assert type(attendance["minimum_percent"]) in (int, float)
    assert 0 < attendance["minimum_percent"] <= 100
    assert attendance["minimum_percent"] == 75
    assert attendance["condonation_available"] is False
    assert facts["laboratory_policy"]["attendance_policy_ref"] == "/attendance_policy"
    assert facts["examination"]["eligibility"]["attendance_policy_ref"] == "/attendance_policy"

    def count_thresholds(value):
        if isinstance(value, dict):
            return sum(key == "minimum_percent" for key in value) + sum(
                count_thresholds(item) for item in value.values()
            )
        if isinstance(value, list):
            return sum(count_thresholds(item) for item in value)
        return 0

    assert count_thresholds(facts) == 1


def test_examination_totals_and_pass_conditions(facts):
    for scheme in facts["examination"]["schemes"].values():
        assert scheme["internal_assessment_marks"] + scheme["end_semester_marks"] == scheme["total_marks"]
        assert sum(c["marks"] for c in scheme["internal_components"]) == scheme["internal_assessment_marks"]
        assert all(c["marks"] > 0 for c in scheme["internal_components"])
        assert 0 < scheme["minimum_total_to_pass"] <= scheme["total_marks"]
        assert 0 < scheme["minimum_end_semester_to_pass"] <= scheme["end_semester_marks"]
        assert scheme["minimum_end_semester_to_pass"] <= scheme["minimum_total_to_pass"]
        assert scheme["minimum_internal_to_pass"] is None
        assert set(scheme["evaluation_modes"]) == {"theory", "laboratory", "practical"}
    ref = facts["laboratory_policy"]["assessment_scheme_ref"]
    assert resolve_pointer(facts, ref) == facts["examination"]["schemes"]["course_standard"]


def test_grading_bands_cover_passing_marks_without_overlap(facts):
    scheme = facts["examination"]["schemes"]["course_standard"]
    bands = facts["examination"]["grading"]["bands"]
    covered = []
    for band in bands:
        assert band["minimum_marks"] <= band["maximum_marks"]
        assert 0 < band["grade_points"] <= 10
        covered.extend(range(band["minimum_marks"], band["maximum_marks"] + 1))
    assert len(covered) == len(set(covered))
    assert set(covered) == set(range(scheme["minimum_total_to_pass"], scheme["total_marks"] + 1))


def test_calendar_dates_and_term_order(facts):
    terms = facts["academic_calendar"]["terms"]
    assert len({t["id"] for t in terms}) == len(terms)
    assert sorted(s for term in terms for s in term["semester_numbers"]) == list(range(1, 9))
    start_year, end_year = map(int, facts["institution"]["academic_year"].split("-"))
    assert end_year == start_year + 1
    previous_break_end = None
    courses = {c["code"]: c for c in facts["courses"]}
    milestone_ids = []
    for term in terms:
        commencement = date.fromisoformat(term["commencement"])
        teaching_end = date.fromisoformat(term["teaching_end"])
        assert start_year <= commencement.year <= end_year
        if previous_break_end:
            assert previous_break_end < commencement
        ranges = {}
        for key in ("internal_examinations", "laboratory_examinations", "end_semester_examinations", "semester_break"):
            period = term[key]
            start, end = date.fromisoformat(period["start"]), date.fromisoformat(period["end"])
            assert start <= end
            assert start_year <= start.year <= end.year <= end_year
            ranges[key] = (start, end)
        internal, lab, end_sem, semester_break = (ranges[k] for k in ("internal_examinations", "laboratory_examinations", "end_semester_examinations", "semester_break"))
        assert commencement < internal[0] <= internal[1] <= teaching_end < lab[0]
        assert lab[1] < end_sem[0] <= end_sem[1] < semester_break[0]
        previous_break_end = semester_break[1]
        for milestone in term["milestones"]:
            assert commencement <= date.fromisoformat(milestone["date"]) <= teaching_end
            milestone_ids.append(milestone["id"])
            if "course_code" in milestone:
                assert courses[milestone["course_code"]]["semester"] in term["semester_numbers"]
        for course_code, exam_date in term["course_examinations"].items():
            assert courses[course_code]["semester"] in term["semester_numbers"]
            assert end_sem[0] <= date.fromisoformat(exam_date) <= end_sem[1]
    assert len(milestone_ids) == len(set(milestone_ids))


def test_laboratory_and_placement_requirements(facts):
    labs = facts["laboratory_policy"]
    for key in ("record_submission", "experiment_completion", "demonstration", "missing_work_rule"):
        assert labs[key].strip()
    assert labs["safety_and_discipline"]
    placement = facts["placement_policy"]["recruitment_eligibility"]
    assert 0 <= placement["minimum_cgpa"] <= 10
    assert placement["maximum_pending_courses"] >= 0
    assert set(placement["eligible_semesters"]) <= set(range(1, 9))


def test_topic_taxonomy_is_unique_and_connected(facts):
    topics = facts["nlp_topic_taxonomy"]
    ids = {t["id"] for t in topics}
    assert len(ids) == len(topics)
    assert len({t["name"] for t in topics}) == len(topics)
    for topic in topics:
        assert topic["aliases"] and all(alias.strip() for alias in topic["aliases"])
        assert len(topic["aliases"]) == len(set(topic["aliases"]))
        assert set(topic["related_topic_ids"]) <= ids - {topic["id"]}


def test_question_paper_targets_are_feasible_and_varied(facts):
    design = facts["question_paper_design"]
    assert design["status"] == "generated_with_validated_targets"
    assert len(design["paper_ids"]) == len(set(design["paper_ids"])) == 4
    targets = design["topic_targets"]
    assert len({t["topic_id"] for t in targets}) == len(targets)
    assert {t["topic_id"] for t in targets} == {t["id"] for t in facts["nlp_topic_taxonomy"]}
    totals = {}
    for target in targets:
        counts = target["question_counts_by_paper"]
        assert set(counts) == set(design["paper_ids"])
        assert all(type(count) is int and count >= 0 for count in counts.values())
        totals[target["topic_id"]] = sum(counts.values())
        assert totals[target["topic_id"]] > 0
    for paper_id in design["paper_ids"]:
        assert sum(t["question_counts_by_paper"][paper_id] for t in targets) == design["primary_topic_slots_per_paper"]
    assert totals["transformers"] > totals["tf_idf"] > totals["bert"]
    assert totals["attention"] > totals["bert"] > totals["semantic_search"]
    assert totals["word_embeddings"] > totals["text_preprocessing"]
    assert totals["word2vec"] > totals["tokenization"]
    assert design["course_code"] == "HCE-CSE603"


def test_benchmark_ids_categories_and_references(facts):
    design = facts["retrieval_design"]
    benchmarks = design["benchmarks"]
    assert 12 <= len(benchmarks) <= 15
    assert len({b["id"] for b in benchmarks}) == len(benchmarks)
    assert len({b["query"] for b in benchmarks}) == len(benchmarks)
    query_types = {"direct_fact", "lexical_search", "semantic_search", "comparison",
                   "cross_document", "summarization", "question_paper_analysis"}
    assert {b["query_type"] for b in benchmarks} == query_types
    course_ids = {c["code"] for c in facts["courses"]}
    topic_ids = {t["id"] for t in facts["nlp_topic_taxonomy"]}
    for benchmark in benchmarks:
        assert benchmark["query"].strip() and benchmark["teaching_purpose"].strip()
        assert benchmark["expected_source_categories"]
        assert set(benchmark["expected_source_categories"]) <= set(design["source_categories"])
        assert benchmark["expected_refs"]
        for reference in benchmark["expected_refs"]:
            if reference["kind"] == "course":
                assert reference["id"] in course_ids
            elif reference["kind"] == "topic":
                assert reference["id"] in topic_ids
            else:
                assert reference["kind"] == "fact"
                assert resolve_pointer(facts, reference["id"]) is not None


def test_semantic_contrasts_have_generation_guidance(facts):
    contrasts = [b for b in facts["retrieval_design"]["benchmarks"] if b["semantic_lexical_contrast"]]
    assert len(contrasts) >= 4
    for benchmark in contrasts:
        assert benchmark["query_type"] == "semantic_search"
        assert benchmark["source_phrasing_hint"].strip()
        assert benchmark["semantic_rationale"].strip()
        assert benchmark["query"].casefold() not in benchmark["source_phrasing_hint"].casefold()
