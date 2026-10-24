from __future__ import annotations

import importlib

import pytest


def test_extraction_imports_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    import extraction

    importlib.reload(extraction)
