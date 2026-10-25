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
from core import Document, PatientSubmission
from extraction import (
    AnticoagulationPlan,
    PlanPhase,
    BloodPressure,
    Evidence,
    EXTRACTION_INSTRUCTIONS,
    PatientDocumentExtraction,
    SurgicalConsent,
    Temperature,
    empty_extraction,
    extract_documents,
    quote_pieces,
    short_doc_ids,
    validate_extraction,
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
    assert '"doc_id": "D1"' in prompt and '"doc_id": "D2"' in prompt
    assert "note-1" not in prompt and "consult-1" not in prompt  # Long ids never reach the model.
    assert "Elective carpal tunnel release" in prompt
    assert "Hold Xarelto 2 days before surgery." in prompt


def test_verbatim_quote_is_kept(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("D1", "09:05 BP 157/117"))
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert len(result.facts.blood_pressures) == 1
    assert result.rejected == []


def test_altered_quote_is_rejected(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("D1", "BP 157/110"))
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
    facts.blood_pressures.append(_bp("D2", "09:05 BP 157/117"))
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert result.facts.blood_pressures == []


def test_bad_plan_phase_quote_is_nulled(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(
            medication="Xarelto",
            before_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id="D1", quote="Hold Xarelto 3 days before surgery.")),
            after_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id="D2", quote="resume Xarelto 24-48 hours after surgery")),
            evidence=Evidence(doc_id="D2", quote="Plan: resume Xarelto"),
        )
    )
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    plan = result.facts.anticoagulation_plans[0]
    assert (plan.before_surgery.status, plan.before_surgery.evidence) == ("STATED", None)
    assert plan.after_surgery.evidence is not None
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


def test_short_ids_are_mapped_back_to_real_doc_ids(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("D1", "09:05 BP 157/117"))
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(
            medication="Xarelto",
            before_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id="D1", quote="Hold Xarelto 2 days before surgery.")),
            after_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id="D2", quote="resume Xarelto 24-48 hours after surgery")),
            evidence=Evidence(doc_id="D2", quote="Plan: resume Xarelto"),
        )
    )
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert result.rejected == []
    assert result.facts.blood_pressures[0].evidence.doc_id == "note-1"
    plan = result.facts.anticoagulation_plans[0]
    assert (plan.before_surgery.evidence.doc_id, plan.after_surgery.evidence.doc_id, plan.evidence.doc_id) == (
        "note-1", "consult-1", "consult-1",
    )


def test_unknown_short_id_is_rejected(client: Mock, submission: PatientSubmission) -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(_bp("D9", "09:05 BP 157/117"))
    client.responses.parse.return_value = _completed(facts)

    result = extract_documents(submission, model="m")

    assert result.facts.blood_pressures == []
    assert result.rejected[0]["reason"] == "unknown doc_id"
    assert result.rejected[0]["fact"]["evidence"]["doc_id"] == "D9"


def test_short_ids_follow_document_order(submission: PatientSubmission) -> None:
    assert short_doc_ids(submission) == {"D1": "note-1", "D2": "consult-1"}


# -------------------------
# Piecewise quote matching (patterns seen in the live run)
# -------------------------

CONSENT_DOC = Document(
    doc_id="consent-1",
    text=(
        "SURGICAL CONSENT\nProcedure: carpal tunnel release\n\nI, Mia Haddad, authorize the surgeon.\n\n"
        "Patient signature: /s/ Mia Haddad (electronically signed 03/08/2026 12:21)\n\nPhysician attestation: done."
    ),
)
VITALS_DOC = Document(
    doc_id="hp-1",
    text="PHYSICAL EXAM:\n  Vitals 10:05  BP 127/74, HR 68, Temperature 36.6 °C (oral), SpO2 95% RA\n",
)


def _consent_quote(quote: str) -> SurgicalConsent:
    return SurgicalConsent(status="SIGNED", signer="PATIENT", signed_at=None, procedure="carpal tunnel release",
                           matches_scheduled_procedure=True, evidence=Evidence(doc_id="consent-1", quote=quote))


def _temp_quote(quote: str) -> Temperature:
    return Temperature(value=36.6, unit="C", measured_at="10:05", evidence=Evidence(doc_id="hp-1", quote=quote))


def _kept(fact) -> bool:
    facts = empty_extraction()
    (facts.surgical_consents if isinstance(fact, SurgicalConsent) else facts.temperatures).append(fact)
    validated, rejected = validate_extraction(facts, [CONSENT_DOC, VITALS_DOC])
    return bool(validated.surgical_consents or validated.temperatures) and not rejected


def test_quote_pieces_split_on_line_breaks_and_ellipses() -> None:
    assert quote_pieces("a  b\n\nc ... d…e") == ["a b", "c", "d", "e"]
    assert quote_pieces(" \n ... ") == []


@pytest.mark.parametrize(
    "quote",
    [
        # Lines that are far apart in the document, joined by line breaks.
        "Patient signature: /s/ Mia Haddad (electronically signed 03/08/2026 12:21)\n\nProcedure: carpal tunnel release",
        # Parts joined by an ellipsis.
        "Procedure: carpal tunnel release ... Patient signature: /s/ Mia Haddad",
        # Extra or collapsed whitespace.
        "Patient   signature: /s/ Mia Haddad",
    ],
)
def test_consent_quote_pieces_each_found_are_accepted(quote: str) -> None:
    assert _kept(_consent_quote(quote))


@pytest.mark.parametrize(
    "quote",
    [
        "Temperature 36.6 °C (oral)",  # Real degree sign.
        "Temperature 36.6 \x176C (oral)",  # Garbled degree sign as the model emitted it.
        "Temperature 36.6 C (oral)",  # Degree sign dropped.
        "Temperature 36.6 �C (oral)",  # Replacement character.
    ],
)
def test_degree_sign_variants_are_accepted(quote: str) -> None:
    assert _kept(_temp_quote(quote))


@pytest.mark.parametrize(
    "quote",
    [
        "Vitals 10:05  Temperature 36.6 °C",  # Words skipped inside one piece.
        "Temperature 36.8 °C (oral)",  # Wrong value.
        "Procedure: carpal tunnel release\nPatient signature: /s/ Mia Okafor",  # One piece is not in the document.
        "\n...\n",  # No content at all.
    ],
)
def test_quote_with_any_unmatched_piece_is_rejected(quote: str) -> None:
    fact = _temp_quote(quote) if "Temperature" in quote or "Vitals" in quote else _consent_quote(quote)

    assert not _kept(fact)


def test_prompt_forbids_skipping_words() -> None:
    assert "Never skip" in EXTRACTION_INSTRUCTIONS
    assert '"..."' in EXTRACTION_INSTRUCTIONS


def test_prompt_plan_rules_require_a_status_per_phase() -> None:
    assert "choose a status for each phase, every time" in EXTRACTION_INSTRUCTIONS
    for status in ("STATED", "DEFERRED", "NOT_ADDRESSED"):
        assert f"- {status}:" in EXTRACTION_INSTRUCTIONS
    assert "The same sentence may be quoted" in EXTRACTION_INSTRUCTIONS
    assert "One entry per medication per document" in EXTRACTION_INSTRUCTIONS


def test_schema_forces_an_explicit_phase_status() -> None:
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(PatientDocumentExtraction)
    plan = schema["$defs"]["AnticoagulationPlan"]
    phase = schema["$defs"]["PlanPhase"]

    assert {"before_surgery", "after_surgery"} <= set(plan["required"])
    assert plan["properties"]["before_surgery"] == {"$ref": "#/$defs/PlanPhase"}  # Not nullable.
    assert phase["properties"]["status"]["enum"] == ["STATED", "DEFERRED", "NOT_ADDRESSED"]
