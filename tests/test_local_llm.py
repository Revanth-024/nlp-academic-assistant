from pathlib import Path
from unittest.mock import patch
import pytest
from src.local_llm import RecordingTestLLM,LocalGenerationError
from src.llm_resources import (load_local_llm,local_model_status,LocalModelUnavailable,
                               MODEL_ID,MODEL_REVISION,MAX_INPUT_TOKENS,MAX_NEW_TOKENS)


def test_recording_test_backend_preserves_exact_prompt_and_response():
    model=RecordingTestLLM('test response')
    prompt='Rules\n\nContext\nSOURCE 1'
    assert model.generate(prompt)=='test response' and model.generate(prompt)=='test response'
    assert model.prompts==[prompt,prompt]
    with pytest.raises(ValueError): model.generate(' ')


def test_missing_model_no_download(tmp_path):
    assert not local_model_status(tmp_path)['files_present']
    with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
        with pytest.raises(LocalModelUnavailable,match='prompt construction remain available'):
            load_local_llm(tmp_path)
    assert MODEL_ID=='Qwen/Qwen2.5-0.5B-Instruct'
    assert len(MODEL_REVISION)==40


@pytest.mark.skipif(not local_model_status()['files_present'],
                    reason='Optional local generator is not installed; explicit preflight enables this integration check.')
def test_real_cached_cpu_generation_offline_once():
    with patch('requests.sessions.Session.request',side_effect=AssertionError('No network')):
        backend=load_local_llm()
        assert str(next(backend.model.parameters()).device)=='cpu'
        assert not backend.model.training
        prompt='Answer briefly: What is two plus two?'
        answer=backend.generate(prompt)
        assert answer.strip() and answer==backend.generate(prompt)
        assert load_local_llm() is backend
        assert MAX_INPUT_TOKENS==3072 and MAX_NEW_TOKENS==96
        with pytest.raises(LocalGenerationError,match='no silent truncation'):
            backend.generate('word '*4000)
