import pytest


@pytest.fixture(autouse=True)
def _zai_api_key(monkeypatch):
    """Provide a fake API key so model.key resolution doesn't fail."""
    monkeypatch.setenv("GLM_API_KEY", "test-key-...")
