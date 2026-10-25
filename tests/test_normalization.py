from __future__ import annotations

from datetime import date, datetime, time

import pytest

from core import PatientSubmission
from extraction import (
    AnticoagulationPlan,
    BloodPressure,
    Evidence,
    ExtractionResult,
    HistoryPhysical,
    Medication,
    PlanPhase,
    Temperature,
    empty_extraction,
)
from normalization import (
    build_evidence,
    canonical_anticoagulant,
    parse_date,
    parse_time_of_day,
    parse_when,
    to_f,
)

NOTE = "VITAL SIGNS: 09:05 BP 157/117, Temp 38.6 C. Currently taking Xarelto 20 mg."
HP = "HISTORY AND PHYSICAL: pre-op evaluation complete."


def _submission(**overrides: object) -> PatientSubmission:
    payload: dict[str, object] = {
        "procedure": {
            "procedure_type": "Elective hernia repair",
            "procedure_risk": "LOW",
            "procedure_date": "2026-03-10",
        },
        "vitals": [
            {"type": "blood_pressure", "systolic": 122, "diastolic": 76, "date": "2026-03-01T10:12:00Z"},
            {"type": "temperature", "value_f": 98.8, "date": "2026-03-01T10:15:00Z"},
        ],
        "labs": [
            {"id": "a", "code": "LAB-CBC", "effective_at": "2026-03-01T08:10:00Z", "status": "final"},
            {"id": "b", "code": "CBC", "effective_at": "2026-03-05T08:10:00Z", "status": "entered-in-error"},
            {"id": "c", "code": "BMP", "effective_at": "2026-03-02T08:10:00Z", "status": "final"},
            {"id": "d", "code": "CMP", "effective_at": "2026-03-02T08:10:00Z", "status": "final"},
            {"id": "e", "code": "CBC", "effective_at": "2026-03-06T08:10:00Z", "status": "preliminary"},
            {"id": "f", "code": "CMP", "effective_at": "2026-03-06T08:10:00Z", "status": None},
            {"id": "g", "code": "CMP", "effective_at": "2026-03-07T08:10:00Z", "status": "cancelled"},
        ],
        "medications": [
            {"name": "Aspirin EC 81 mg PO daily", "active": True},
            {"name": "Coumadin 5 mg PO daily", "active": None},
        ],
        "documents": [
            {"doc_id": "note-1", "type": "Nursing Intake", "date": "2026-03-03", "text": NOTE},
            {"doc_id": "hp-1", "type": "H&P", "date": "2026-02-20", "text": HP},
        ],
    }
    payload.update(overrides)
    return PatientSubmission.model_validate(payload)


def _ok(facts=None) -> ExtractionResult:
    return ExtractionResult(facts=facts or empty_extraction())


# -------------------------
# Normalization
# -------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("warfarin (Coumadin) 5 mg PO daily", "warfarin"),
        ("Coumadin 5 mg PO daily (INR goal 2-3)", "warfarin"),
        ("Xarelto 20 mg PO daily with evening meal", "rivaroxaban"),
        ("Eliquis 5 mg PO BID", "apixaban"),
        ("dabigatran (Pradaxa) 150 mg PO BID", "dabigatran"),
        ("edoxaban (Savaysa) 60 mg PO daily", "edoxaban"),
        ("Lovenox 40 mg SC daily", "enoxaparin"),
        ("ASA 81 mg PO daily", None),
        ("Aspirin EC 81 mg PO daily", None),
        ("clopidogrel 75 mg PO daily", None),
        (None, None),
    ],
)
def test_canonical_anticoagulant(name: str | None, expected: str | None) -> None:
    assert canonical_anticoagulant(name) == expected


def test_parse_dates_and_times() -> None:
    assert parse_when("2026-02-24T10:12:00Z") == (datetime(2026, 2, 24, 10, 12), True)
    assert parse_when("03/15/2026") == (datetime(2026, 3, 15), False)
    assert parse_date("2026-03-15") == date(2026, 3, 15)
    assert parse_date("not a date") is None
    assert parse_time_of_day("09:05") == time(9, 5)
    assert parse_time_of_day("8:30 PM") == time(20, 30)
    assert parse_time_of_day("see flowsheet") is None


def test_celsius_to_fahrenheit_is_unrounded() -> None:
    assert to_f(37.9, "C") == pytest.approx(100.22)
    assert to_f(38.0, "C") == pytest.approx(100.4)
    assert to_f(98.6, "F") == 98.6


# -------------------------
# Structured parsing
# -------------------------


def test_labs_use_aliases_and_drop_invalid() -> None:
    bundle = build_evidence(_submission(), _ok())

    assert [(lab.test, lab.at) for lab in bundle.labs] == [
        ("CBC", date(2026, 3, 1)),
        ("CMP", date(2026, 3, 2)),
    ]
    assert bundle.labs[0].evidence.path == "labs[0]"


def test_structured_vitals_keep_path_and_time() -> None:
    bundle = build_evidence(_submission(), _ok())

    bp, temp = bundle.vitals
    assert (bp.kind, bp.systolic, bp.diastolic) == ("BP", 122, 76)
    assert bp.at == datetime(2026, 3, 1, 10, 12) and bp.has_time
    assert bp.evidence.source == "structured" and bp.evidence.path == "vitals[0]"
    assert (temp.kind, temp.temp_f) == ("TEMP", 98.8)


def test_structured_meds_keep_only_anticoagulants() -> None:
    bundle = build_evidence(_submission(), _ok())

    assert len(bundle.meds) == 1
    med = bundle.meds[0]
    assert (med.canonical, med.status, med.as_of) == ("warfarin", "UNCERTAIN", None)
    assert med.evidence.path == "medications[1]"


def test_procedure_fields_parsed() -> None:
    bundle = build_evidence(_submission(), _ok())
    assert bundle.procedure_date == date(2026, 3, 10)
    assert bundle.procedure_risk == "LOW"

    blank = build_evidence(_submission(procedure={"procedure_type": "x"}), _ok())
    assert blank.procedure_date is None and blank.procedure_risk is None


# -------------------------
# Document facts
# -------------------------


def test_document_vital_time_combines_doc_date_and_time_of_day() -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(
        BloodPressure(systolic=157, diastolic=117, measured_at="09:05",
                      evidence=Evidence(doc_id="note-1", quote="09:05 BP 157/117"))
    )
    facts.temperatures.append(
        Temperature(value=38.6, unit="C", measured_at=None, evidence=Evidence(doc_id="note-1", quote="Temp 38.6 C"))
    )

    bundle = build_evidence(_submission(), _ok(facts))

    bp, temp = bundle.vitals[2:]
    assert bp.at == datetime(2026, 3, 3, 9, 5)
    assert bp.has_time and bp.time_source == "doc_date+text_time"
    assert bp.evidence.doc_type == "Nursing Intake" and bp.evidence.quote == "09:05 BP 157/117"
    assert temp.at == datetime(2026, 3, 3) and not temp.has_time and temp.time_source == "doc_date"
    assert temp.temp_f == pytest.approx(101.48)
    assert temp.evidence.value == "38.6 C (101.5 F)"


def test_document_vital_with_explicit_datetime() -> None:
    facts = empty_extraction()
    facts.blood_pressures.append(
        BloodPressure(systolic=157, diastolic=117, measured_at="03/02/2026 14:30",
                      evidence=Evidence(doc_id="note-1", quote="BP 157/117"))
    )

    bp = build_evidence(_submission(), _ok(facts)).vitals[-1]

    assert bp.at == datetime(2026, 3, 2, 14, 30) and bp.time_source == "text"


def test_document_meds_and_plans_are_canonicalized() -> None:
    facts = empty_extraction()
    facts.medications.append(
        Medication(name="Xarelto 20 mg", status="CURRENT", as_of=None,
                   evidence=Evidence(doc_id="note-1", quote="Currently taking Xarelto 20 mg."))
    )
    facts.medications.append(
        Medication(name="aspirin", status="CURRENT", as_of=None, evidence=Evidence(doc_id="note-1", quote="Temp"))
    )
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(medication="Xarelto", before_surgery=PlanPhase(status="NOT_ADDRESSED", evidence=None),
                            after_surgery=PlanPhase(status="NOT_ADDRESSED", evidence=None),
                            evidence=Evidence(doc_id="note-1", quote="Xarelto"))
    )

    bundle = build_evidence(_submission(), _ok(facts))

    doc_med = bundle.meds[-1]
    assert (doc_med.canonical, doc_med.status, doc_med.as_of) == ("rivaroxaban", "CURRENT", date(2026, 3, 3))
    assert len(bundle.meds) == 2  # structured Coumadin + document Xarelto; aspirin dropped.
    assert bundle.plans[0].canonical == "rivaroxaban"


def test_completed_hp_falls_back_to_document_date() -> None:
    facts = empty_extraction()
    facts.history_physicals.append(
        HistoryPhysical(status="COMPLETED", date=None, evidence=Evidence(doc_id="hp-1", quote="pre-op evaluation complete"))
    )

    hp = build_evidence(_submission(), _ok(facts)).history_physicals[0]

    assert hp.at == date(2026, 2, 20)


def test_extraction_failure_is_carried_into_bundle() -> None:
    failed = ExtractionResult(facts=empty_extraction(), failed=True, error="boom")

    bundle = build_evidence(_submission(), failed)

    assert bundle.extraction_failed and bundle.extraction_error == "boom"


def test_rejected_facts_are_carried_into_bundle() -> None:
    bp = {"systolic": 190, "diastolic": 100, "measured_at": "09:05",
          "evidence": {"doc_id": "note-1", "quote": "BP 190/100"}}
    rejected = [{"kind": "blood_pressures", "reason": "quote not found verbatim in document", "fact": bp}]

    bundle = build_evidence(_submission(), ExtractionResult(facts=empty_extraction(), rejected=rejected))

    assert bundle.rejected_facts == rejected
    assert all(v.evidence.source == "structured" for v in bundle.vitals)  # Not merged as valid evidence.
    (typed,) = bundle.rejected.vitals
    assert (typed.systolic, typed.at) == (190, datetime(2026, 3, 3, 9, 5))
    assert typed.evidence.rejected_reason == "blood_pressures: quote not found verbatim in document"


def test_rejected_plan_phase_brings_back_the_whole_plan() -> None:
    plan = {
        "medication": "Xarelto",
        "before_surgery": {"status": "STATED", "evidence": {"doc_id": "note-1", "quote": "Hold Xarelto 3 days"}},
        "after_surgery": {"status": "STATED",
                          "evidence": {"doc_id": "note-1", "quote": "Currently taking Xarelto 20 mg."}},
        "evidence": {"doc_id": "note-1", "quote": "Xarelto"},
    }
    rejected = [{"kind": "anticoagulation_plans.before_surgery", "reason": "quote not found verbatim in document",
                 "fact": plan["before_surgery"]["evidence"], "plan": plan}]

    bundle = build_evidence(_submission(), ExtractionResult(facts=empty_extraction(), rejected=rejected))

    (typed,) = bundle.rejected.plans
    assert typed.canonical == "rivaroxaban"
    assert typed.before_surgery is not None and typed.after_surgery is not None


def test_no_rejected_facts_means_no_rejected_bundle() -> None:
    assert build_evidence(_submission(), _ok()).rejected is None


def _plan_with(before: PlanPhase, after: PlanPhase):
    facts = empty_extraction()
    facts.anticoagulation_plans.append(
        AnticoagulationPlan(medication="Xarelto", before_surgery=before, after_surgery=after,
                            evidence=Evidence(doc_id="note-1", quote="Xarelto"))
    )
    return build_evidence(_submission(), _ok(facts)).plans[0]


def test_plan_phases_are_labelled_and_only_stated_counts() -> None:
    stated = PlanPhase(status="STATED", evidence=Evidence(doc_id="note-1", quote="Currently taking Xarelto 20 mg."))
    deferred = PlanPhase(status="DEFERRED", evidence=Evidence(doc_id="note-1", quote="Temp 38.6 C"))

    plan = _plan_with(stated, deferred)

    assert plan.evidence.value == "plan for Xarelto"
    assert plan.before_surgery.value == "before surgery: STATED"
    assert plan.after_surgery is None  # Deferred is not an instruction.
    assert [e.value for e in plan.other_evidence] == ["after surgery: DEFERRED"]


def test_stated_phase_without_a_quote_is_not_an_instruction() -> None:
    plan = _plan_with(PlanPhase(status="STATED", evidence=None), PlanPhase(status="NOT_ADDRESSED", evidence=None))

    assert (plan.before_surgery, plan.after_surgery, plan.other_evidence) == (None, None, [])
