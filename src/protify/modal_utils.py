def parse_modal_api_key(modal_api_key: str) -> tuple:
    cleaned = modal_api_key.strip()
    assert cleaned != "", "modal_api_key cannot be empty."
    assert ":" in cleaned, "modal_api_key must be provided as '<modal_token_id>:<modal_token_secret>'."
    token_id, token_secret = cleaned.split(":", 1)
    token_id = token_id.strip()
    token_secret = token_secret.strip()
    assert token_id != "", "modal_token_id parsed from modal_api_key is empty."
    assert token_secret != "", "modal_token_secret parsed from modal_api_key is empty."
    return token_id, token_secret
