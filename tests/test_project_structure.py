"""Verify that the workshop's required foundation files exist."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("directory", [
    "pages", "src", "data/academic_documents", "data/question_papers",
    "data/search_corpus", "data/sample_pdfs", "notebooks", "docs",
    "workshop", "assets/architecture", "assets/screenshots", "tests",
])
def test_required_directory_exists(directory):
    assert (ROOT / directory).is_dir()


@pytest.mark.parametrize("module", [
    "__init__", "preprocessing", "classical_nlp", "embeddings", "retrieval",
    "document_processor", "rag", "local_llm", "question_analyzer", "utils",
])
def test_source_module_exists(module):
    assert (ROOT / "src" / f"{module}.py").is_file()


@pytest.mark.parametrize("page", [
    "01_Preprocessing_Lab", "02_BoW_NGrams_Lab", "03_TFIDF_Search_Lab",
    "04_Word_Embeddings_Lab", "05_Sentence_Embeddings_Lab",
    "06_Semantic_Search_Lab", "07_Transformer_Lab", "08_RAG_Lab",
    "09_Question_Paper_Analyzer", "10_Academic_Assistant",
])
def test_streamlit_page_exists(page):
    assert (ROOT / "pages" / f"{page}.py").is_file()


@pytest.mark.parametrize("document", [
    "docs/INSTRUCTOR_GUIDE.md", "docs/STUDENT_GUIDE.md",
    "workshop/TODO_INDEX.md", "workshop/SETUP_CHECKLIST.md",
])
def test_workshop_document_exists(document):
    assert (ROOT / document).is_file()
