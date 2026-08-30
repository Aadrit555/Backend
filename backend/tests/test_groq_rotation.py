"""Tests for Groq API key rotation resilience."""
import pytest
from unittest.mock import patch, MagicMock
from groq import RateLimitError
import httpx

import backend.orchestrator.groq_client as groq_client
from backend.orchestrator.groq_client import chat


@pytest.fixture(autouse=True)
def setup_keys():
    """Ensure 3 keys are configured and reset index, then restore original."""
    orig_1 = groq_client.settings.groq_api_key_1
    orig_2 = groq_client.settings.groq_api_key_2
    orig_3 = groq_client.settings.groq_api_key_3
    
    groq_client.settings.groq_api_key_1 = "key1"
    groq_client.settings.groq_api_key_2 = "key2"
    groq_client.settings.groq_api_key_3 = "key3"
    groq_client._configured_keys = []
    groq_client._active_key_index = 0
    
    yield
    
    groq_client.settings.groq_api_key_1 = orig_1
    groq_client.settings.groq_api_key_2 = orig_2
    groq_client.settings.groq_api_key_3 = orig_3
    groq_client._configured_keys = []
    groq_client._active_key_index = 0


def _make_rate_limit_error():
    # groq.RateLimitError requires a response and a body
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    return RateLimitError("Rate limit reached", response=response, body={})


@patch("backend.orchestrator.groq_client.Groq")
def test_first_key_succeeds(mock_groq_class):
    """Scenario A: First key succeeds immediately."""
    mock_client = mock_groq_class.return_value
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Success"
    mock_client.chat.completions.create.return_value = mock_response

    res = chat([{"role": "user", "content": "hi"}])
    
    assert res["content"] == "Success"
    assert groq_client._active_key_index == 0
    assert mock_client.chat.completions.create.call_count == 1


@patch("backend.orchestrator.groq_client.Groq")
def test_first_key_429_second_succeeds(mock_groq_class):
    """Scenario B: First key 429s, second key succeeds."""
    mock_client = mock_groq_class.return_value
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Success"
    
    # Fail first call, succeed second call
    mock_client.chat.completions.create.side_effect = [
        _make_rate_limit_error(),
        mock_response
    ]

    res = chat([{"role": "user", "content": "hi"}])
    
    assert res["content"] == "Success"
    assert groq_client._active_key_index == 1
    assert mock_client.chat.completions.create.call_count == 2


@patch("time.sleep")
@patch("backend.orchestrator.groq_client.Groq")
def test_all_keys_429_falls_back(mock_groq_class, mock_sleep):
    """Scenario C: All 3 keys 429, falls back to exponential backoff and eventually fails."""
    mock_client = mock_groq_class.return_value
    
    # Always fail with 429
    mock_client.chat.completions.create.side_effect = _make_rate_limit_error()

    with pytest.raises(RateLimitError):
        chat([{"role": "user", "content": "hi"}])
    
    # 3 retries (max_retries) * 3 keys per retry = 9 calls
    assert mock_client.chat.completions.create.call_count == 9
    
    # Sleep should be called 2 times (for attempt 0 and 1, since attempt 2 raises without sleep)
    assert mock_sleep.call_count == 2
    # Check that sleep times were 2**0=1 and 2**1=2
    mock_sleep.assert_any_call(1)
    mock_sleep.assert_any_call(2)
    
    # active_key_index should have rotated 9 times, so 9 % 3 = 0.
    assert groq_client._active_key_index == 0
