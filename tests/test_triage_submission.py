from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from core import (
    BASELINE_SYSTEM_PROMPT,
    EvidenceItem,
    Finding,
    PatientSubmission,
    TriageIssue,
    TriageOutput,
    triage_output_json_schema,
    triage_submission,
)
from run_evals import ScoredOutput


@pytest.fixture
def openai_client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    client = Mock()
    client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "decision": "READY",
                "issues": [],
                "explanation": "All required criteria are satisfied.",
            }
        ),
    )

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda: client))
    return client


@pytest.fixture
def submission_payload() -> dict[str, object]:
    return {
        "patient": {"id": "patient-1"},
        "procedure": {
            "case_id": "case-1",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {
                "type": "blood_pressure",
                "systolic": 120,
                "diastolic": 80,
                "date": "2026-01-25",
            }
        ],
        "labs": [
            {
                "code": "CBC",
                "display": "Complete blood count",
                "effective_at": "2026-01-20",
                "status": "final",
            }
        ],
        "medications": [],
        "conditions": [],
        "documents": [
            {
                "type": "history_and_physical",
                "date": "2026-01-20",
                "text": "History and physical completed.",
            },
            {
                "type": "surgical_consent",
                "date": "2026-01-22",
                "text": "Signed surgical consent.",
            },
        ],
    }


def test_triage_submission_returns_structured_output(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    output = triage_submission(submission_payload, model="test-model")

    assert isinstance(output, TriageOutput)
    assert output.decision == "READY"
    assert output.issues == []
    assert output.explanation == "All required criteria are satisfied."

    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    assert call["model"] == "test-model"
    assert call["instructions"] == BASELINE_SYSTEM_PROMPT

    text_config = call["text"]
    assert isinstance(text_config, dict)
    response_format = text_config["format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["name"] == "preop_triage_output"
    assert response_format["schema"] == triage_output_json_schema()
    assert response_format["strict"] is False


def test_triage_submission_builds_prompt_from_validated_submission(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    submission = PatientSubmission.model_validate(submission_payload)

    triage_submission(submission, model="test-model")

    assert openai_client.responses.create.call_count == 1
    call = openai_client.responses.create.call_args.kwargs
    message = call["input"][0]
    content = message["content"][0]
    prompt = content["text"]

    assert message["type"] == "message"
    assert message["role"] == "user"
    assert content["type"] == "input_text"
    assert json.loads(prompt.split("Submission JSON:\n", 1)[1]) == submission.model_dump()


def test_triage_submission_rejects_invalid_model_json(
    openai_client: Mock,
    submission_payload: dict[str, object],
) -> None:
    openai_client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps(
            {
                "decision": "MAYBE",
                "issues": [],
                "explanation": "Not a valid triage decision.",
            }
        )
    )

    with pytest.raises(ValidationError):
        triage_submission(submission_payload, model="test-model")

    assert openai_client.responses.create.call_count == 1


# -------------------------
# Output schema
# -------------------------


def _new_style_output() -> TriageOutput:
    cbc_evidence = EvidenceItem(source="structured", path="labs[0]", value="CBC 2026-01-01")
    temp_evidence = EvidenceItem(
        source="document",
        doc_id="doc-1",
        doc_type="Pre-op Nursing Intake",
        doc_date="2026-01-30",
        quote="Temp 38.6 C",
        value="38.6 C",
    )
    findings = [
        Finding(
            rule="TESTING",
            check="CBC",
            status="FAIL",
            category="REQUIRED_TESTING",
            summary="Latest CBC is 31 days before the procedure.",
            evidence=[cbc_evidence],
        ),
        Finding(
            rule="SAFETY",
            check="TEMPERATURE",
            status="FAIL",
            category="ACUTE_SAFETY_EXCLUSION",
            summary="Latest temperature 101.5 F exceeds 100.4 F.",
            evidence=[temp_evidence],
        ),
        Finding(rule="SAFETY", check="BP", status="PASS", summary="Latest BP 120/80."),
    ]
    issues = [
        TriageIssue(
            category=f.category,
            evidence=f.evidence,
            rule=f.rule,
            check=f.check,
            status=f.status,
            summary=f.summary,
        )
        for f in findings
        if f.status != "PASS" and f.category is not None
    ]
    return TriageOutput(
        decision="NOT_CLEARED",
        issues=issues,
        explanation="Fever exclusion; CBC out of window.",
        findings=findings,
    )


def test_new_style_output_validates_against_scorer() -> None:
    output = _new_style_output()

    scored = ScoredOutput.model_validate(output.model_dump())

    assert scored.decision == "NOT_CLEARED"
    assert sorted(issue.category for issue in scored.issues) == [
        "ACUTE_SAFETY_EXCLUSION",
        "REQUIRED_TESTING",
    ]


def test_new_style_output_round_trips_through_json() -> None:
    output = _new_style_output()

    restored = TriageOutput.model_validate_json(output.model_dump_json())

    assert restored == output
    assert isinstance(restored.issues[0].evidence, list)
    assert restored.issues[1].evidence[0].quote == "Temp 38.6 C"
    assert len(restored.findings) == 3


def test_baseline_style_output_still_validates() -> None:
    output = TriageOutput.model_validate(
        {
            "decision": "NEEDS_FOLLOW_UP",
            "issues": [{"category": "REQUIRED_DOCUMENTATION", "evidence": "No consent."}],
            "explanation": "Consent missing.",
        }
    )

    assert output.issues[0].evidence == "No consent."
    assert output.findings == []


def test_invalid_issue_category_is_rejected() -> None:
    with pytest.raises(ValidationError):
        TriageIssue.model_validate({"category": "SOMETHING_ELSE", "evidence": []})

    with pytest.raises(ValidationError):
        Finding.model_validate(
            {"rule": "SAFETY", "check": "BP", "status": "FAIL", "category": "BAD", "summary": "x"}
        )
