from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import openai
import pytest
from pydantic import ValidationError

import extraction
from core import EvidenceItem, Finding, PatientSubmission, TriageIssue, TriageOutput, triage_submission
from extraction import (
    AnticoagulationPlan,
    PlanPhase,
    BloodPressure,
    Evidence,
    HistoryPhysical,
    Medication,
    PatientDocumentExtraction,
    SurgicalConsent,
    Temperature,
    empty_extraction,
)
from run_evals import ScoredOutput

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "patients_sample_50.jsonl"


def _case(index: int) -> dict[str, object]:
    with SAMPLE.open(encoding="utf-8") as handle:
        return json.loads(handle.readlines()[index])["submission"]


def _doc_id(submission: dict[str, object], doc_type: str) -> str:
    """The short id ("D3") the model sees for this document."""

    (index,) = [i for i, d in enumerate(submission["documents"], start=1) if d["type"] == doc_type]
    return f"D{index}"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock = Mock()
    monkeypatch.setattr(extraction, "_client", lambda: mock)
    return mock


def _returns(client: Mock, facts: PatientDocumentExtraction) -> None:
    client.responses.parse.return_value = SimpleNamespace(status="completed", output_parsed=facts, output=[])


def _checks(output: TriageOutput) -> dict[str, str]:
    return {f.check: f.status for f in output.findings}


def _categories(output: TriageOutput) -> list[str]:
    return sorted({i.category for i in output.issues})


def _hp(doc_id: str, date: str, quote: str = "PREOPERATIVE HISTORY AND PHYSICAL") -> HistoryPhysical:
    return HistoryPhysical(status="COMPLETED", date=date, evidence=Evidence(doc_id=doc_id, quote=quote))


def _patient_consent(doc_id: str, quote: str, procedure: str) -> SurgicalConsent:
    return SurgicalConsent(
        status="SIGNED",
        signer="PATIENT",
        signed_at=None,
        procedure=procedure,
        matches_scheduled_procedure=True,
        evidence=Evidence(doc_id=doc_id, quote=quote),
    )


# -------------------------
# Sample-case integration (real submissions, hand-written extraction with real quotes)
# -------------------------


def _case0_facts(submission: dict[str, object]) -> PatientDocumentExtraction:
    """Xarelto appears only in a nursing note; the cardiology consult has the full plan."""

    nursing = _doc_id(submission, "Nursing Pre-procedure Assessment")
    cardiology = _doc_id(submission, "Cardiology Consultation")
    hp = _doc_id(submission, "History and Physical")
    facts = empty_extraction()
    facts.history_physicals.append(_hp(hp, "02/21/2026"))
    facts.surgical_consents.append(
        _patient_consent(
            _doc_id(submission, "Surgery Consent (scanned)"),
            "Patient signature: /s/ Grace Lopez (electronically signed 02/27/2026 08:05)",
            "carpal tunnel release",
        )
    )
    facts.medications.append(
        Medication(name="Xarelto 20 mg", status="CURRENT", as_of=None,
                   evidence=Evidence(doc_id=nursing, quote="Patient also reports taking Xarelto 20 mg"))
    )
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(
            medication="Xarelto",
            before_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id=cardiology, quote="Hold Xarelto for the 2 days before surgery")),
            after_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id=cardiology, quote="Resume Xarelto 24-48 hours after surgery")),
            evidence=Evidence(doc_id=cardiology, quote="maintained on Xarelto 20 mg PO daily with evening meal"),
        )
    )
    facts.blood_pressures.append(
        BloodPressure(systolic=127, diastolic=74, measured_at="10:05",
                      evidence=Evidence(doc_id=hp, quote="BP 127/74"))
    )
    facts.temperatures.append(
        Temperature(value=98.0, unit="F", measured_at="10:05", evidence=Evidence(doc_id=hp, quote="T 98.0 F"))
    )
    return facts


def test_anticoagulant_found_only_in_note_with_complete_plan_is_ready(client: Mock) -> None:
    submission = _case(0)
    _returns(client, _case0_facts(submission))

    output = triage_submission(submission, model="m")

    assert output.decision == "READY"
    assert output.issues == []
    assert _checks(output) == {
        "H&P": "PASS", "CONSENT": "PASS", "CBC": "PASS",
        "ANTICOAG:rivaroxaban": "PASS", "BP": "PASS", "TEMPERATURE": "PASS",
    }
    plan = next(f for f in output.findings if f.check == "ANTICOAG:rivaroxaban")
    assert "Hold Xarelto for the 2 days before surgery" in [e.quote for e in plan.evidence]
    # The model answered with short ids; the output carries the real document id.
    real_ids = {d["doc_id"] for d in submission["documents"]}
    assert {e.doc_id for e in plan.evidence if e.source == "document"} <= real_ids
    # The 02/24 structured vitals are newer than the 02/21 H&P reading.
    bp = next(f for f in output.findings if f.check == "BP")
    assert [e.path for e in bp.evidence] == ["vitals[1]"]


def test_bp_recheck_clears_high_arrival_reading(client: Mock) -> None:
    submission = _case(16)
    nursing = _doc_id(submission, "Nursing Pre-procedure Assessment")
    facts = empty_extraction()
    facts.history_physicals.append(_hp(_doc_id(submission, "H&P [PDF]"), "02/21/2026"))
    facts.history_physicals.append(_hp(_doc_id(submission, "History and Physical"), "06/22/2025", "HISTORY AND PHYSICAL"))
    facts.surgical_consents.append(
        _patient_consent(
            _doc_id(submission, "Procedure Consent Form"),
            "Patient signature on file - signed in clinic 03/13/2026 12:06",
            "inguinal hernia repair",
        )
    )
    facts.blood_pressures.append(
        BloodPressure(systolic=189, diastolic=112, measured_at="08:06",
                      evidence=Evidence(doc_id=nursing, quote="08:06  NIBP 189/112"))
    )
    facts.blood_pressures.append(
        BloodPressure(systolic=137, diastolic=78, measured_at="08:30",
                      evidence=Evidence(doc_id=nursing, quote="ADDENDUM 08:30: Re-measured 137/78 mmHg"))
    )
    facts.temperatures.append(
        Temperature(value=37.1, unit="C", measured_at="08:06", evidence=Evidence(doc_id=nursing, quote="T 37.1 C"))
    )
    _returns(client, facts)

    output = triage_submission(submission, model="m")

    assert output.decision == "READY"
    bp = next(f for f in output.findings if f.check == "BP")
    assert [e.quote for e in bp.evidence] == ["ADDENDUM 08:30: Re-measured 137/78 mmHg"]


def test_missing_cbc_needs_follow_up_for_testing_only(client: Mock) -> None:
    submission = _case(26)
    plan_doc = _doc_id(submission, "Perioperative Medication Plan")
    facts = empty_extraction()
    facts.history_physicals.append(
        _hp(_doc_id(submission, "Preoperative History & Physical (H&P)"), "03/17/2026",
            "pre-operative evaluation prior to tonsillectomy")
    )
    facts.surgical_consents.append(
        _patient_consent(
            _doc_id(submission, "Surgical Consent"),
            "Patient signature: /s/ Mateo Patel (electronically signed 03/16/2026 10:55)",
            "tonsillectomy",
        )
    )
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(
            medication="dabigatran",
            before_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id=plan_doc, quote="Hold dabigatran for the 3 days before surgery")),
            after_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id=plan_doc, quote="Restart dabigatran on postoperative day 1")),
            evidence=Evidence(doc_id=plan_doc, quote="maintained on dabigatran 150 mg PO BID"),
        )
    )
    _returns(client, facts)

    output = triage_submission(submission, model="m")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert _categories(output) == ["REQUIRED_TESTING"]
    assert [(i.check, i.status) for i in output.issues] == [("CBC", "FAIL")]
    assert _checks(output)["ANTICOAG:dabigatran"] == "PASS"


def test_one_sentence_plan_covering_both_phases_passes(client: Mock) -> None:
    """Case 26's plan is one sentence; the model may quote it for both phases."""

    submission = _case(26)
    plan_doc = _doc_id(submission, "Perioperative Medication Plan")
    sentence = (
        "Hold dabigatran for the 3 days before surgery (last dose 03/23/2026; no doses in the 72 hours prior). "
        "Restart dabigatran on postoperative day 1 at the full dose if there are no bleeding concerns."
    )
    facts = empty_extraction()
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(
            medication="dabigatran",
            before_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id=plan_doc, quote=sentence)),
            after_surgery=PlanPhase(status="STATED", evidence=Evidence(doc_id=plan_doc, quote=sentence)),
            evidence=Evidence(doc_id=plan_doc, quote=sentence),
        )
    )
    _returns(client, facts)

    output = triage_submission(submission, model="m")

    assert output.rejected_document_facts == []
    assert _checks(output)["ANTICOAG:dabigatran"] == "PASS"
    assert "ANTICOAGULATION_MANAGEMENT" not in _categories(output)


def test_null_procedure_risk_is_missing_required_data(client: Mock) -> None:
    submission = _case(29)
    facts = empty_extraction()
    facts.history_physicals.append(_hp(_doc_id(submission, "Surgical H&P"), "03/10/2026"))
    facts.surgical_consents.append(
        _patient_consent(
            _doc_id(submission, "Informed Consent - Surgical Procedure"),
            "Patient signature: /s/ Noah Patel (electronically signed 03/25/2026 15:26)",
            "abdominal aortic aneurysm repair",
        )
    )
    _returns(client, facts)

    output = triage_submission(submission, model="m")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert _categories(output) == ["MISSING_REQUIRED_DATA"]
    assert [(i.check, i.status) for i in output.issues] == [("REQUIRED_TESTS", "UNKNOWN")]


def test_extraction_failure_never_returns_ready(client: Mock) -> None:
    client.responses.parse.side_effect = openai.APIConnectionError(request=Mock())

    output = triage_submission(_case(0), model="m")

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert _categories(output) == ["MISSING_REQUIRED_DATA"]
    assert {i.check for i in output.issues} == {"H&P", "CONSENT", "ANTICOAGULANT", "BP", "TEMPERATURE"}
    assert _checks(output)["CBC"] == "PASS"  # Labs are structured, so they are still checked.
    bp = next(f for f in output.findings if f.check == "BP")
    assert bp.evidence[0].path == "vitals[1]"  # Structured evidence is kept.


def _with_rejected_bp(submission: dict[str, object], systolic: float, diastolic: float) -> PatientDocumentExtraction:
    """Case 0 facts plus a BP from the 02/26 nursing note whose quote is not in the document."""

    facts = _case0_facts(submission)
    nursing = _doc_id(submission, "Nursing Pre-procedure Assessment")
    facts.blood_pressures.append(
        BloodPressure(systolic=systolic, diastolic=diastolic, measured_at=None,
                      evidence=Evidence(doc_id=nursing, quote=f"BP {systolic}/{diastolic}"))
    )
    return facts


def test_rejected_reading_that_would_change_the_outcome_is_unknown(client: Mock) -> None:
    submission = _case(0)
    _returns(client, _with_rejected_bp(submission, 200, 120))

    output = triage_submission(submission, model="m")

    # Unverified, so not NOT_CLEARED; but it would be the latest BP, so not READY either.
    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [(i.check, i.status, i.category) for i in output.issues] == [("BP", "UNKNOWN", "MISSING_REQUIRED_DATA")]
    bp = output.issues[0]
    assert [e.rejected_reason is not None for e in bp.evidence] == [False, True]  # Structured, then rejected.
    assert "If the unverified document fact is correct" in bp.summary
    assert [r["kind"] for r in output.rejected_document_facts] == ["blood_pressures"]


def test_rejected_reading_that_agrees_does_not_block(client: Mock) -> None:
    submission = _case(0)
    _returns(client, _with_rejected_bp(submission, 124, 70))

    output = triage_submission(submission, model="m")

    assert output.decision == "READY"
    assert "rejected" in output.explanation


def test_rejected_consent_makes_consent_unknown_not_missing(client: Mock) -> None:
    submission = _case(0)
    facts = _case0_facts(submission)
    consent = facts.surgical_consents[0]
    consent.evidence = Evidence(doc_id=consent.evidence.doc_id, quote="Patient signature: /s/ Grace Lopex")
    _returns(client, facts)

    output = triage_submission(submission, model="m")

    assert output.decision == "NEEDS_FOLLOW_UP"
    (issue,) = output.issues
    assert (issue.check, issue.status, issue.category) == ("CONSENT", "UNKNOWN", "MISSING_REQUIRED_DATA")
    assert issue.evidence[-1].quote == "Patient signature: /s/ Grace Lopex"
    assert issue.evidence[-1].rejected_reason


def test_output_is_json_and_scorer_compatible(client: Mock) -> None:
    submission = _case(0)
    _returns(client, _case0_facts(submission))

    output = triage_submission(PatientSubmission.model_validate(submission), model="m")
    dumped = json.loads(json.dumps(output.model_dump()))

    assert ScoredOutput.model_validate(dumped).decision == "READY"
    assert TriageOutput.model_validate(dumped) == output


# -------------------------
# Output schema
# -------------------------


def test_invalid_issue_category_is_rejected() -> None:
    issue = {"category": "SOMETHING_ELSE", "evidence": [], "rule": "SAFETY", "check": "BP",
             "status": "FAIL", "summary": "x"}
    with pytest.raises(ValidationError):
        TriageIssue.model_validate(issue)

    with pytest.raises(ValidationError):
        Finding.model_validate(
            {"rule": "SAFETY", "check": "BP", "status": "FAIL", "category": "BAD", "summary": "x"}
        )


def test_issue_evidence_must_be_structured() -> None:
    with pytest.raises(ValidationError):
        TriageIssue.model_validate({"category": "REQUIRED_TESTING", "evidence": "free text", "rule": "TESTING",
                                    "check": "CBC", "status": "FAIL", "summary": "x"})

    issue = TriageIssue(category="REQUIRED_TESTING", evidence=[EvidenceItem(source="structured", path="labs[0]")],
                        rule="TESTING", check="CBC", status="FAIL", summary="x")
    assert TriageIssue.model_validate_json(issue.model_dump_json()) == issue
