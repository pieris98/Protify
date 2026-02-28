import pytest

try:
    from modal_utils import parse_modal_api_key
except ImportError:
    from ..modal_utils import parse_modal_api_key


def test_parse_modal_api_key_success() -> None:
    token_id, token_secret = parse_modal_api_key("abc123:def456")
    assert token_id == "abc123"
    assert token_secret == "def456"


def test_parse_modal_api_key_strips_whitespace() -> None:
    token_id, token_secret = parse_modal_api_key("  abc123  :  def456  ")
    assert token_id == "abc123"
    assert token_secret == "def456"


def test_parse_modal_api_key_requires_separator() -> None:
    with pytest.raises(AssertionError):
        parse_modal_api_key("missing_separator")


def test_parse_modal_api_key_rejects_empty_parts() -> None:
    with pytest.raises(AssertionError):
        parse_modal_api_key(":secret")
    with pytest.raises(AssertionError):
        parse_modal_api_key("token:")
