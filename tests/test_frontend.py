from streamlit.testing.v1 import AppTest
import requests


def test_workspace_renders_when_api_is_offline(monkeypatch):
    def unavailable(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", unavailable)
    app = AppTest.from_file("../frontend/app.py").run()
    assert not app.exception
    assert any("API offline" in error.value for error in app.error)
    assert any("source" in heading.value.lower() for heading in app.header)
