# Instructor Start

1. Use the complete reference repository. Read docs/INSTRUCTOR_GUIDE.md, workshop/WORKSHOP_SCHEDULE.md and docs/KNOWN_LIMITATIONS.md.
2. Before class, install requirements and explicitly preload NLTK, MiniLM and the three task models. Qwen is optional:

```text
python -m src.nltk_resources --download
python -m src.sentence_resources --download
python -m src.transformer_resources --download
python -m src.llm_resources --download
python -m workshop.preflight --full-classroom --smoke
python -m pytest
python -m workshop.run_reliability_audit --pages
```

The Qwen command is optional; leave it out on constrained machines. Task17 baseline:551 reference tests on Python3.11/3.12, zero skips in prepared environments,63 offline checks. See RELEASE_VALIDATION.md for the final packaging-inclusive totals.

3. Generate and validate the student package:

```text
python -m workshop.build_student_starter
python -m workshop.validate_student_starter dist/nlp-academic-assistant-student
python -m workshop.validate_student_starter dist/nlp-academic-assistant-student --runtime --resource-root .
```

`--resource-root .` explicitly borrows only cached resources from this reference tree for validation; source imports are still the generated stub modules. No caches are distributed. An unchanged output may be rebuilt deterministically; modified student work is never overwritten—choose a new output directory.

4. Distribute the validated generated starter, not an independently maintained copy. Check START_HERE and its setup smoke. Correctness tests fail until TODOs are implemented. Keep the full Instructor Guide/evaluators out of normal student distribution.
5. Run `python -m streamlit run app.py` for reference demonstrations. Inspect intermediate representations; show retrieval-only mode before optional Qwen. Unit-specific missing evidence is visibly withheld in the assistant, not repaired by benchmark hacks.
6. Recovery: ask students to commit working progress; compare the relevant bounded core with the reference, or provide a checkpoint for the current exercise after teaching it. Save their edits first. Restore only the current section/file after checking which earlier work it contains; never reset their entire repository casually. Once an owner publishes checkpoints/tags, document their actual identities rather than inventing them now. No duplicated solution modules.
7. Before public distribution, owner selects a repository license, reviews third-party terms, reviews the final validation matrix and publishes manually if desired. No remote/tag/deployment action is performed by this task.
