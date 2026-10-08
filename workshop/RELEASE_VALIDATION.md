# Release Validation — Task18

**Functional architecture frozen after Task17. Task18 changes are limited to packaging, documentation, validation and deployment preparation.**

## Measured reference / starter matrix

| Area | Reference | Generated student starter |
|---|---|---|
|Dependencies / installation|Existing validated3.11/3.12; new isolated3.11 installation also succeeds|Same requirements; tested with the fresh isolated environment |
|Preflight|PASS, full-classroom local inference smoke; no downloads|Inventory works; unimplemented smoke operations report expected TODO warnings; caches must be explicitly preloaded |
|Tests|**574 passed** on3.11.17 /3.12.14; zero skips; one existing Transformer/NumPy deprecation warning|**2 setup smoke tests pass;340 correctness tests collect**; unsolved cores intentionally fail correctness tests |
|Packaging tests|**23 passed**|49 unique TODO IDs /50 stub regions verified |
|Streamlit|Home +all10 pages render in offline audit|Home +all10 pages render on both versions and fresh3.11; live starter launch and TODO1.1 guidance verified |
|Data|14 academic Markdown documents, four papers,20 descriptions, three derived PDFs; existing consistency suites pass|Same canonical data retained |
|Source APIs|Complete implementations|All function signatures/docstrings preserved; bounded bodies absent |
|Instructor material|Instructor Guide, evaluators, measurement artifacts and release tools present|Excluded; provided CSV loader extracted separately for notebook use |
|Guides|Student/Instructor/start/schedule/portfolio/deployment/limitations present|Student/public guides, START_HERE, public TODO registry and setup/schedule present |
|Notebooks|All8 execute offline, including cached optional Qwen|All8 pause educationally at a TODO; later cells are held until restart after implementation |
|Offline|**63 checks passed**, no requests/downloads; all task-model inference local|Imports/pages/notebook pauses checked offline with explicitly borrowed preloaded resources; no models bundled |
|Determinism|Canonical source unchanged by builds|Equivalent files/manifests on two builds; fixed-timestamp ZIP equivalent; modified student work protected |
|Paths / secrets|No development-machine paths in source/guides/notebooks; measured report paths normalized; no required credential or key-shaped secrets found|No absolute local paths, instructor solutions, caches, weights or environment files |
|Licensing|**Owner decision pending: no repository LICENSE selected**|Same owner decision; third-party model/library terms remain applicable |
|Cloud|Root requirements/entrypoint/paths/fallbacks validated locally|Not live deployed; use completed student work or reference for an actual demonstration |
|Browser|1280px width verified|Live TODO guidance verified; narrower layouts remain unverified |

## Fresh-environment scope

A new isolated64-bit Python3.11.17 virtual environment was created without system-site packages. Requirements installed offline using previously downloaded local wheel files plus14 distribution-record-based rehydrated wheels for the exact validated versions. A separate frozen runtime constraint list was used only for validation; no dependency version changed in the project. This is a real isolated install, not global-package inheritance, but **not a pristine online PyPI installation or Linux/cloud validation**.

`pip check` reports no broken requirements. Fresh-environment full-classroom preflight/smoke succeeds using the already-preloaded reference resource tree; pretrained weights were reused explicitly, not redownloaded or bundled. Fresh-environment starter imports,11 page renders and eight expected notebook pauses succeed. Model downloading instructions are the previously tested explicit setup commands.

## Runtime / timing scope

Reference full suites:3.11 **574 passed in159.82s**,3.12 **574 passed in144.90s**. The prior551 tests remain green;23 packaging tests were added. Core skips0 in these prepared environments. Optional model checks on an unprepared machine require preload and may be unavailable; no fake model is substituted.

The Task18 reference notebook run executes all8 blocks. Its RAG notebook took about265s including cold/competing validation load and optional model work; an actual warm attendance generation took about22s in that run. Earlier Task17 notebook and latency measurements remain historical observations, not guaranteed classroom runtimes. Install/preload before class; Qwen is optional and should not hold up every student.

## Release ownership / next actions

1. Select the repository license and review upstream terms before licensed public distribution.
2. Review known limitations and the matrix; narrower-than1280 layout and live hosted resource feasibility remain unverified.
3. Generate the starter with `python -m workshop.build_student_starter --archive dist/nlp-academic-assistant-student.zip`.
4. Validate with `python -m workshop.validate_student_starter dist/nlp-academic-assistant-student --runtime --resource-root .` and check START_HERE/setup smoke.
5. Publish your chosen reference/starter repositories manually, using your actual remote URLs. No push, tag, release, account change, upload or deployment occurred in this task. `v1.0-workshop` is a future recommendation only.
6. If deploying, follow docs/DEPLOYMENT_GUIDE.md and the official provider instructions; verify actual hosted capabilities before claiming a live demo.
