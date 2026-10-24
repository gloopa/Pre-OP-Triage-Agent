"""Single structured LLM extraction over all patient documents, with quote validation."""

from __future__ import annotations

from functools import cache
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel


@cache
def _client() -> OpenAI:
    """One reusable client per process; created lazily so imports work without an API key."""

    return OpenAI(timeout=60.0, max_retries=2)
