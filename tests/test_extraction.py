from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import openai
import pytest

import extraction
from core import PatientSubmission
from extraction import (
    AnticoagulationPlan,
    BloodPressure,
    Evidence,
    EXTRACTION_INSTRUCTIONS,
    PatientDocumentExtraction,
    empty_extraction,
    extract_documents,
)

NOTE_TEXT = "VITAL SIGNS: 09:05 BP 157/117, Temp 36.7 C. Hold Xarelto 2 days before surgery."
CONSULT_TEXT = "Plan: resume Xarelto 24-48 hours after surgery once hemostasis secure."


@pytest.fixture
def submission() -> PatientSubmission:
    return PatientSubmission.model_validate(
        {
            "procedure": {"procedure_type": "Elective carpal tunnel release"},
            "documents": [
                {"doc_id": "note-1", "type": "Nursing Note", "date": "2026-03-03", "text": NOTE_TEXT},
                {"doc_id": "consult-1", "type": "Cardiology", "date": "2026-02-20", "text": CONSULT_TEXT},
            ],
        }
    )


def _completed(parsed: PatientDocumentExtraction | None, output: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(status="completed", output_parsed=parsed, output=output or [])


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock()
    mock.responses.parse.return_value = _completed(empty_extraction())
    monkeypatch.setattr(extraction, "_client", lambda: mock)
    return mock


def _bp(doc_id: str, quote: str) -> BloodPressure:
    return BloodPressure(
        systolic=157, diastolic=117, measured_at="09:05", evidence=Evidence(doc_id=doc_id, quote=quote)
    )


def test_extraction_imports_without_api_key() -> None:
    # A fresh interpreter, so reloading doesn't swap out the model classes other tests use.
    env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
    subprocess.run(
        [sys.executable, "-c", "import extraction"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        check=True,
    )


def test_request_shape(client: Mock, submission: PatientSubmission) -> None:
    result = extract_documents(submission, model="test-model")

    assert not result.failed
    assert client.responses.parse.call_count == 1
    call = client.responses.parse.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["temperature"] == 0
    assert call["text_format"] is PatientDocumentExtraction
    assert call["instructions"] == EXTRACTION_INSTRUCTIONS
    assert "PLANNED" in call["instructions"] and "temporary perioperative hold" in call["instructions"]
    prompt = call["input"]
    assert "note-1" in prompt and "consult-1" in prompt
    assert "Elective carpal tunnel release" in prompt
    assert "Hold Xarelto 2 days before surgery." in prompt


def test_verbatim_quote_is_kept(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("note-1", "09:05 BP 157/117"))
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert len(result.facts.blood_pressures) == 1
    assert result.rejected == []


def test_altered_quote_is_rejected(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("note-1", "BP 157/110"))
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert result.facts.blood_pressures == []
    assert result.rejected[0]["kind"] == "blood_pressures"
    assert "verbatim" in result.rejected[0]["reason"]


def test_unknown_doc_id_is_rejected(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("made-up", "09:05 BP 157/117"))
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert result.facts.blood_pressures == []
    assert result.rejected[0]["reason"] == "unknown doc_id"


def test_quote_from_wrong_document_is_rejected(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("consult-1", "09:05 BP 157/117"))
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert result.facts.blood_pressures == []


def test_bad_plan_phase_quote_is_nulled(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(
            medication="Xarelto",
            before_surgery=Evidence(doc_id="note-1", quote="Hold Xarelto 3 days before surgery."),
            after_surgery=Evidence(doc_id="consult-1", quote="resume Xarelto 24-48 hours after surgery"),
            evidence=Evidence(doc_id="consult-1", quote="Plan: resume Xarelto"),
        )
    )
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    plan = result.facts.anticoagulation_plans[0]
    assert plan.before_surgery is None
    assert plan.after_surgery is not None
    assert result.rejected[0]["kind"] == "anticoagulation_plans.before_surgery"


def test_api_error_sets_failed(client: Mock, submission: PatientSubmission) -> None:
    client.responses.parse.side_effect = openai.APIConnectionError(request=Mock())

    result = extract_documents(submission, model="m")

    assert result.failed
    assert "APIConnectionError" in (result.error or "")
    assert result.facts == empty_extraction()


def test_refusal_sets_failed(client: Mock, submission: PatientSubmission) -> None:
    refusal = SimpleNamespace(
        type="message", content=[SimpleNamespace(type="refusal", refusal="I can't help with that.")]
    )
    client.responses.parse.return_value = _completed(None, output=[refusal])

    result = extract_documents(submission, model="m")

    assert result.failed
    assert "refusal" in (result.error or "")


def test_incomplete_response_sets_failed(client: Mock, submission: PatientSubmission) -> None:
    client.responses.parse.return_value = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output_parsed=None,
        output=[],
    )

    result = extract_documents(submission, model="m")

    assert result.failed
    assert "incomplete" in (result.error or "")


def test_no_documents_skips_the_call(client: Mock) -> None:
    result = extract_documents(PatientSubmission.model_validate({"documents": []}), model="m")

    assert not result.failed
    assert client.responses.parse.call_count == 0
