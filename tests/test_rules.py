from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from core import EvidenceItem, Finding
from normalization import (
    ConsentFact,
    EvidenceBundle,
    HistoryPhysicalFact,
    Lab,
    MedStatement,
    PlanFact,
    Vital,
)
from rules import (
    evaluate_all,
    evaluate_anticoagulation,
    evaluate_documentation,
    evaluate_safety,
    evaluate_testing,
    route,
)

PROCEDURE = date(2026, 3, 13)


def _ev(label: str) -> EvidenceItem:
    return EvidenceItem(source="document", doc_id=label, quote=label)


def _bundle(**overrides: object) -> EvidenceBundle:
    values: dict[str, object] = {
        "procedure_date": PROCEDURE,
        "procedure_risk": "LOW",
        "procedure_type": "Carpal tunnel release",
        "procedure_evidence": EvidenceItem(source="structured", path="procedure"),
    }
    values.update(overrides)
    return EvidenceBundle(**values)


def _days_before(days: int) -> date:
    return PROCEDURE - timedelta(days=days)


def _by_check(findings) -> dict[str, tuple[str, str | None]]:
    return {f.check: (f.status, f.category) for f in findings}


# -------------------------
# Rule 1: Documentation
# -------------------------


def _hp(days: int, status: str = "COMPLETED") -> HistoryPhysicalFact:
    return HistoryPhysicalFact(status, _days_before(days), _ev(f"hp-{days}"))


def _consent(status: str = "SIGNED", signer: str = "PATIENT", matches: bool | None = True) -> ConsentFact:
    return ConsentFact(status, signer, matches, None, _ev(f"consent-{status}-{signer}-{matches}"))


@pytest.mark.parametrize(("days", "status"), [(0, "PASS"), (30, "PASS"), (31, "FAIL"), (-1, "FAIL")])
def test_hp_window_boundaries(days: int, status: str) -> None:
    hp = _by_check(evaluate_documentation(_bundle(history_physicals=[_hp(days)], consents=[_consent()])))["H&P"]

    assert hp == (status, None if status == "PASS" else "REQUIRED_DOCUMENTATION")


def test_old_hp_does_not_block_a_recent_one() -> None:
    findings = evaluate_documentation(_bundle(history_physicals=[_hp(200), _hp(20)], consents=[_consent()]))

    assert _by_check(findings)["H&P"] == ("PASS", None)


def test_missing_hp_fails_and_keeps_statements_as_evidence() -> None:
    finding = evaluate_documentation(_bundle(history_physicals=[_hp(5, "MISSING")]))[0]

    assert (finding.status, finding.category) == ("FAIL", "REQUIRED_DOCUMENTATION")
    assert finding.evidence[0].doc_id == "hp-5"


def test_uncertain_hp_is_unknown() -> None:
    finding = evaluate_documentation(_bundle(history_physicals=[_hp(5, "UNCERTAIN")]))[0]

    assert (finding.status, finding.category) == ("UNKNOWN", "MISSING_REQUIRED_DATA")


def test_null_procedure_date_makes_hp_unknown_but_consent_still_checked() -> None:
    findings = evaluate_documentation(_bundle(procedure_date=None, consents=[_consent()]))

    assert _by_check(findings) == {"H&P": ("UNKNOWN", "MISSING_REQUIRED_DATA"), "CONSENT": ("PASS", None)}


@pytest.mark.parametrize(
    ("consents", "expected"),
    [
        ([_consent()], "PASS"),
        ([_consent("PENDING", "UNKNOWN", None), _consent()], "PASS"),  # Later signed consent supersedes.
        ([_consent("SIGNED", "OTHER")], "FAIL"),  # Physician signature only.
        ([_consent("PENDING", "UNKNOWN", None)], "FAIL"),  # Sent to portal, not completed.
        ([_consent("UNSIGNED", "PATIENT")], "FAIL"),
        ([_consent(matches=False)], "FAIL"),  # Signed, but for a different procedure.
        ([], "FAIL"),
        ([_consent(matches=None)], "UNKNOWN"),
        ([_consent("SIGNED", "UNKNOWN")], "UNKNOWN"),
        ([_consent("UNCERTAIN", "UNKNOWN", None)], "UNKNOWN"),
    ],
)
def test_consent(consents: list[ConsentFact], expected: str) -> None:
    consent = _by_check(evaluate_documentation(_bundle(history_physicals=[_hp(5)], consents=consents)))["CONSENT"]

    assert consent[0] == expected


def test_extraction_failure_makes_documentation_unknown() -> None:
    findings = evaluate_documentation(_bundle(extraction_failed=True, extraction_error="timeout"))

    assert {f.status for f in findings} == {"UNKNOWN"}
    assert "timeout" in findings[0].summary


# -------------------------
# Rule 2: Testing
# -------------------------


def _lab(test: str, days: int | None) -> Lab:
    return Lab(test, None if days is None else _days_before(days), _ev(f"{test}-{days}"))


@pytest.mark.parametrize(("risk", "days", "status"), [
    ("LOW", 30, "PASS"), ("LOW", 31, "FAIL"), ("MODERATE", 30, "PASS"), ("MODERATE", 31, "FAIL"),
    ("HIGH", 14, "PASS"), ("HIGH", 15, "FAIL"),
])
def test_cbc_window_boundaries(risk: str, days: int, status: str) -> None:
    labs = [_lab("CBC", days), _lab("CMP", 1)]

    cbc = _by_check(evaluate_testing(_bundle(procedure_risk=risk, labs=labs)))["CBC"]

    assert cbc == (status, None if status == "PASS" else "REQUIRED_TESTING")


@pytest.mark.parametrize(("days", "status"), [(14, "PASS"), (15, "FAIL")])
def test_cmp_window_for_high_risk(days: int, status: str) -> None:
    labs = [_lab("CBC", 1), _lab("CMP", days)]

    assert _by_check(evaluate_testing(_bundle(procedure_risk="HIGH", labs=labs)))["CMP"][0] == status


def test_low_risk_does_not_require_cmp() -> None:
    assert set(_by_check(evaluate_testing(_bundle(labs=[_lab("CBC", 3)])))) == {"CBC"}


def test_high_risk_missing_cmp_fails() -> None:
    findings = _by_check(evaluate_testing(_bundle(procedure_risk="HIGH", labs=[_lab("CBC", 3)])))

    assert findings == {"CBC": ("PASS", None), "CMP": ("FAIL", "REQUIRED_TESTING")}


def test_only_most_recent_result_counts() -> None:
    assert _by_check(evaluate_testing(_bundle(labs=[_lab("CBC", 40), _lab("CBC", 10)])))["CBC"] == ("PASS", None)
    assert _by_check(evaluate_testing(_bundle(labs=[_lab("CBC", 10), _lab("CBC", 31)])))["CBC"] == ("PASS", None)


def test_lab_after_procedure_is_ignored() -> None:
    findings = _by_check(evaluate_testing(_bundle(labs=[_lab("CBC", -2), _lab("CBC", 45)])))

    assert findings["CBC"] == ("FAIL", "REQUIRED_TESTING")


def test_missing_cbc_fails() -> None:
    assert _by_check(evaluate_testing(_bundle(labs=[])))["CBC"] == ("FAIL", "REQUIRED_TESTING")


def test_undated_lab_makes_cbc_unknown() -> None:
    findings = _by_check(evaluate_testing(_bundle(labs=[_lab("CBC", 10), _lab("CBC", None)])))

    assert findings["CBC"] == ("UNKNOWN", "MISSING_REQUIRED_DATA")


@pytest.mark.parametrize("overrides", [{"procedure_date": None}, {"procedure_risk": None}])
def test_null_date_or_risk_makes_testing_unknown(overrides: dict[str, object]) -> None:
    findings = evaluate_testing(_bundle(labs=[_lab("CBC", 3)], **overrides))

    assert [(f.check, f.status, f.category) for f in findings] == [
        ("REQUIRED_TESTS", "UNKNOWN", "MISSING_REQUIRED_DATA")
    ]


# -------------------------
# Rule 3: Anticoagulation
# -------------------------


def _med(canonical: str, status: str, as_of: date | None = None) -> MedStatement:
    return MedStatement(canonical, status, as_of, _ev(f"med-{canonical}"))


def _plan(canonical: str, before: bool = True, after: bool = True) -> PlanFact:
    return PlanFact(
        canonical,
        _ev(f"before-{canonical}") if before else None,
        _ev(f"after-{canonical}") if after else None,
        _ev(f"plan-{canonical}"),
    )


def test_no_anticoagulant_passes() -> None:
    assert _by_check(evaluate_anticoagulation(_bundle())) == {"ANTICOAGULANT": ("PASS", None)}


def test_stopped_anticoagulant_needs_no_plan() -> None:
    findings = evaluate_anticoagulation(_bundle(meds=[_med("warfarin", "STOPPED")]))

    assert _by_check(findings) == {"ANTICOAGULANT": ("PASS", None)}
    assert findings[0].evidence[0].doc_id == "med-warfarin"


def test_current_anticoagulant_with_complete_plan_passes() -> None:
    findings = evaluate_anticoagulation(_bundle(meds=[_med("rivaroxaban", "CURRENT")], plans=[_plan("rivaroxaban")]))

    assert _by_check(findings) == {"ANTICOAG:rivaroxaban": ("PASS", None)}
    assert [e.doc_id for e in findings[0].evidence] == [
        "plan-rivaroxaban", "before-rivaroxaban", "after-rivaroxaban", "med-rivaroxaban",
    ]


def test_uncertain_status_with_complete_plan_passes() -> None:
    # Taking it: the plan covers it. Not taking it: no plan is needed. Rule 3 is met either way.
    findings = evaluate_anticoagulation(_bundle(meds=[_med("warfarin", "UNCERTAIN")], plans=[_plan("warfarin")]))

    assert _by_check(findings) == {"ANTICOAG:warfarin": ("PASS", None)}
    assert "unclear" in findings[0].summary
    assert "med-warfarin" in [e.doc_id for e in findings[0].evidence]


def test_uncertain_status_with_incomplete_plan_is_unknown() -> None:
    bundle = _bundle(meds=[_med("warfarin", "UNCERTAIN")], plans=[_plan("warfarin", after=False)])

    finding = evaluate_anticoagulation(bundle)[0]

    assert (finding.status, finding.category) == ("UNKNOWN", "MISSING_REQUIRED_DATA")
    assert [e.doc_id for e in finding.evidence] == ["med-warfarin", "plan-warfarin", "before-warfarin"]


def test_current_anticoagulant_without_plan_fails() -> None:
    finding = evaluate_anticoagulation(_bundle(meds=[_med("apixaban", "CURRENT")]))[0]

    assert (finding.status, finding.category) == ("FAIL", "ANTICOAGULATION_MANAGEMENT")


@pytest.mark.parametrize(("before", "after", "gap"), [(True, False, "after"), (False, True, "before")])
def test_incomplete_plan_fails_and_says_what_is_missing(before: bool, after: bool, gap: str) -> None:
    bundle = _bundle(meds=[_med("warfarin", "CURRENT")], plans=[_plan("warfarin", before, after)])

    finding = evaluate_anticoagulation(bundle)[0]

    assert (finding.status, finding.category) == ("FAIL", "ANTICOAGULATION_MANAGEMENT")
    assert f"{gap} surgery" in finding.summary


def test_plan_for_different_drug_fails() -> None:
    bundle = _bundle(meds=[_med("apixaban", "CURRENT")], plans=[_plan("warfarin")])

    finding = evaluate_anticoagulation(bundle)[0]

    assert (finding.check, finding.status) == ("ANTICOAG:apixaban", "FAIL")
    assert "no perioperative plan" in finding.summary


def test_two_anticoagulants_give_two_findings() -> None:
    bundle = _bundle(
        meds=[_med("warfarin", "CURRENT"), _med("apixaban", "CURRENT")],
        plans=[_plan("warfarin")],
    )

    assert _by_check(evaluate_anticoagulation(bundle)) == {
        "ANTICOAG:apixaban": ("FAIL", "ANTICOAGULATION_MANAGEMENT"),
        "ANTICOAG:warfarin": ("PASS", None),
    }


def test_uncertain_anticoagulant_status_is_unknown() -> None:
    findings = evaluate_anticoagulation(_bundle(meds=[_med("warfarin", "UNCERTAIN")]))

    assert _by_check(findings) == {"ANTICOAG:warfarin": ("UNKNOWN", "MISSING_REQUIRED_DATA")}


def test_planned_bridge_alone_needs_no_plan() -> None:
    findings = evaluate_anticoagulation(_bundle(meds=[_med("enoxaparin", "PLANNED")]))

    assert _by_check(findings) == {"ANTICOAGULANT": ("PASS", None)}


def test_extraction_failure_makes_anticoagulation_unknown() -> None:
    no_drug = evaluate_anticoagulation(_bundle(extraction_failed=True))
    current = evaluate_anticoagulation(_bundle(meds=[_med("warfarin", "CURRENT")], extraction_failed=True))

    assert _by_check(no_drug) == {"ANTICOAGULANT": ("UNKNOWN", "MISSING_REQUIRED_DATA")}
    assert _by_check(current) == {"ANTICOAG:warfarin": ("UNKNOWN", "MISSING_REQUIRED_DATA")}
    assert current[0].evidence[0].doc_id == "med-warfarin"


def test_extraction_failure_keeps_structured_med_evidence() -> None:
    findings = evaluate_anticoagulation(_bundle(meds=[_med("warfarin", "STOPPED")], extraction_failed=True))

    assert findings[0].status == "UNKNOWN"
    assert findings[0].evidence[0].doc_id == "med-warfarin"


# -------------------------
# Rule 4: Safety
# -------------------------


def _bp(sbp: float, dbp: float, at: datetime, has_time: bool = True) -> Vital:
    return Vital("BP", at, has_time, "text", EvidenceItem(source="document", value=f"{sbp}/{dbp}"),
                 systolic=sbp, diastolic=dbp)


def _temp(temp_f: float, at: datetime, has_time: bool = True) -> Vital:
    return Vital("TEMP", at, has_time, "text", EvidenceItem(source="document", value=f"{temp_f} F"), temp_f=temp_f)


T1 = datetime(2026, 3, 10, 8, 0)
T2 = datetime(2026, 3, 10, 8, 30)
NORMAL_TEMP = _temp(98.6, T1)
NORMAL_BP = _bp(120, 80, T1)


@pytest.mark.parametrize(("sbp", "dbp", "status"), [
    (179, 109, "PASS"), (180, 80, "FAIL"), (120, 110, "FAIL"), (179.9, 109.9, "PASS"),
])
def test_bp_boundaries(sbp: float, dbp: float, status: str) -> None:
    bp = _by_check(evaluate_safety(_bundle(vitals=[_bp(sbp, dbp, T1), NORMAL_TEMP])))["BP"]

    assert bp == (status, None if status == "PASS" else "ACUTE_SAFETY_EXCLUSION")


@pytest.mark.parametrize(("temp_f", "status"), [(100.4, "PASS"), (100.5, "FAIL")])
def test_temperature_boundaries(temp_f: float, status: str) -> None:
    temp = _by_check(evaluate_safety(_bundle(vitals=[NORMAL_BP, _temp(temp_f, T1)])))["TEMPERATURE"]

    assert temp[0] == status


def test_recheck_within_range_clears_high_reading() -> None:
    findings = _by_check(evaluate_safety(_bundle(vitals=[_bp(189, 112, T1), _bp(137, 78, T2), NORMAL_TEMP])))

    assert findings["BP"] == ("PASS", None)


def test_high_recheck_stays_excluded() -> None:
    findings = _by_check(evaluate_safety(_bundle(vitals=[_bp(196, 110, T1), _bp(181, 98, T2), NORMAL_TEMP])))

    assert findings["BP"] == ("FAIL", "ACUTE_SAFETY_EXCLUSION")


@pytest.mark.parametrize(("second", "status"), [
    ((125, 75), "PASS"),  # Both candidates normal: the unclear time does not matter.
    ((185, 75), "UNKNOWN"),  # Candidates disagree.
])
def test_unclear_timestamps_only_matter_when_candidates_disagree(second: tuple[int, int], status: str) -> None:
    untimed = _bp(*second, datetime(2026, 3, 10), has_time=False)

    bp = _by_check(evaluate_safety(_bundle(vitals=[_bp(120, 80, T2), untimed, NORMAL_TEMP])))["BP"]

    assert bp[0] == status


def test_both_candidates_high_is_fail() -> None:
    untimed = _bp(190, 100, datetime(2026, 3, 10), has_time=False)

    bp = _by_check(evaluate_safety(_bundle(vitals=[_bp(185, 90, T2), untimed, NORMAL_TEMP])))["BP"]

    assert bp == ("FAIL", "ACUTE_SAFETY_EXCLUSION")


def test_missing_vitals_are_unknown() -> None:
    findings = _by_check(evaluate_safety(_bundle(vitals=[NORMAL_BP])))

    assert findings == {"BP": ("PASS", None), "TEMPERATURE": ("UNKNOWN", "MISSING_REQUIRED_DATA")}


def test_extraction_failure_makes_safety_unknown() -> None:
    findings = _by_check(evaluate_safety(_bundle(vitals=[NORMAL_BP, NORMAL_TEMP], extraction_failed=True)))

    assert findings == {
        "BP": ("UNKNOWN", "MISSING_REQUIRED_DATA"),
        "TEMPERATURE": ("UNKNOWN", "MISSING_REQUIRED_DATA"),
    }


def test_extraction_failure_keeps_excluded_structured_reading_as_evidence() -> None:
    # A later recheck may be in the unread documents, so this is UNKNOWN rather than NOT_CLEARED,
    # but the high reading stays visible to the reviewer.
    bp = evaluate_safety(_bundle(vitals=[_bp(190, 100, T1), NORMAL_TEMP], extraction_failed=True))[0]

    assert (bp.status, bp.category) == ("UNKNOWN", "MISSING_REQUIRED_DATA")
    assert [e.value for e in bp.evidence] == ["190/100"]
    assert "190/100" in bp.summary


# -------------------------
# Routing
# -------------------------

def _f(rule: str, check: str, status: str, category: str | None = None) -> Finding:
    return Finding(rule=rule, check=check, status=status, category=category, summary=f"{check} {status}.")


ALL_PASS = [
    _f("DOCUMENTATION", "H&P", "PASS"),
    _f("DOCUMENTATION", "CONSENT", "PASS"),
    _f("TESTING", "CBC", "PASS"),
    _f("ANTICOAGULATION", "ANTICOAGULANT", "PASS"),
    _f("SAFETY", "BP", "PASS"),
    _f("SAFETY", "TEMPERATURE", "PASS"),
]


def _with(*replacements: Finding) -> list[Finding]:
    by_check = {f.check: f for f in replacements}
    return [by_check.get(f.check, f) for f in ALL_PASS]


def test_all_pass_is_ready() -> None:
    output = route(ALL_PASS, _bundle())

    assert (output.decision, output.issues) == ("READY", [])
    assert len(output.findings) == 6
    assert output.explanation.startswith("Ready")


def test_safety_fail_is_not_cleared_even_with_other_issues() -> None:
    findings = _with(
        _f("SAFETY", "BP", "FAIL", "ACUTE_SAFETY_EXCLUSION"),
        _f("DOCUMENTATION", "CONSENT", "FAIL", "REQUIRED_DOCUMENTATION"),
        _f("SAFETY", "TEMPERATURE", "UNKNOWN", "MISSING_REQUIRED_DATA"),
    )

    output = route(findings, _bundle())

    assert output.decision == "NOT_CLEARED"
    assert sorted(i.category for i in output.issues) == [
        "ACUTE_SAFETY_EXCLUSION", "MISSING_REQUIRED_DATA", "REQUIRED_DOCUMENTATION",
    ]
    assert "BP FAIL." in output.explanation and "Also needs follow-up" in output.explanation


@pytest.mark.parametrize("finding", [
    _f("TESTING", "CBC", "FAIL", "REQUIRED_TESTING"),
    _f("SAFETY", "TEMPERATURE", "UNKNOWN", "MISSING_REQUIRED_DATA"),
    _f("ANTICOAGULATION", "ANTICOAGULANT", "UNKNOWN", "MISSING_REQUIRED_DATA"),
])
def test_any_fail_or_unknown_needs_follow_up(finding: Finding) -> None:
    output = route(_with(finding), _bundle())

    assert output.decision == "NEEDS_FOLLOW_UP"
    assert [(i.check, i.status, i.category) for i in output.issues] == [
        (finding.check, finding.status, finding.category)
    ]


def test_issues_map_one_to_one_with_non_pass_findings() -> None:
    findings = _with(
        _f("DOCUMENTATION", "H&P", "FAIL", "REQUIRED_DOCUMENTATION"),
        _f("DOCUMENTATION", "CONSENT", "FAIL", "REQUIRED_DOCUMENTATION"),
    )

    output = route(findings, _bundle())

    assert [i.check for i in output.issues] == ["H&P", "CONSENT"]
    assert all(i.summary and i.rule == "DOCUMENTATION" for i in output.issues)


def test_rejected_facts_are_surfaced_in_output() -> None:
    rejected = [{"kind": "blood_pressures", "reason": "unknown doc_id", "fact": {}}]

    output = route(ALL_PASS, _bundle(rejected_facts=rejected))

    assert output.rejected_document_facts == rejected
    assert "1 extracted document fact(s) were rejected" in output.explanation


def test_explanation_is_deterministic() -> None:
    findings = _with(_f("TESTING", "CBC", "FAIL", "REQUIRED_TESTING"))

    assert route(findings, _bundle()).explanation == route(findings, _bundle()).explanation


# -------------------------
# Rejected (unverified) facts
# -------------------------


def _rej(label: str) -> EvidenceItem:
    return EvidenceItem(source="document", doc_id=label, quote=label, rejected_reason="quote not found")


def _all(bundle: EvidenceBundle) -> dict[str, tuple[str, str | None]]:
    return _by_check(evaluate_all(bundle))


VALID = {
    "history_physicals": [_hp(5)],
    "consents": [_consent()],
    "labs": [_lab("CBC", 3)],
    "vitals": [_bp(120, 80, T1), _temp(98.6, T1)],
}


def test_rejected_signed_consent_turns_consent_fail_into_unknown() -> None:
    signed = ConsentFact("SIGNED", "PATIENT", True, None, _rej("consent"))
    bundle = _bundle(**{**VALID, "consents": []}, rejected=_bundle(consents=[signed]))

    findings = evaluate_all(bundle)

    consent = next(f for f in findings if f.check == "CONSENT")
    assert (consent.status, consent.category) == ("UNKNOWN", "MISSING_REQUIRED_DATA")
    assert consent.evidence[-1].rejected_reason == "quote not found"


def test_rejected_fact_that_cannot_change_the_outcome_is_ignored() -> None:
    unsigned = ConsentFact("PENDING", "UNKNOWN", None, None, _rej("pending"))
    old_high_bp = _bp(190, 115, datetime(2026, 2, 1, 9, 0))
    bundle = _bundle(**VALID, rejected=_bundle(consents=[unsigned], vitals=[old_high_bp]))

    assert set(_all(bundle).values()) == {("PASS", None)}


def test_rejected_later_fever_makes_temperature_unknown() -> None:
    later_fever = _temp(101.7, datetime(2026, 3, 11, 8, 30))
    bundle = _bundle(**VALID, rejected=_bundle(vitals=[later_fever]))

    assert _all(bundle)["TEMPERATURE"] == ("UNKNOWN", "MISSING_REQUIRED_DATA")


def test_rejected_later_normal_reading_makes_exclusion_unknown() -> None:
    high = _bp(190, 100, T1)
    later_normal = _bp(130, 80, T2)
    bundle = _bundle(**{**VALID, "vitals": [high, _temp(98.6, T1)]}, rejected=_bundle(vitals=[later_normal]))

    assert _all(bundle)["BP"] == ("UNKNOWN", "MISSING_REQUIRED_DATA")


def test_rejected_anticoagulant_mention_adds_unknown_findings() -> None:
    taking = MedStatement("warfarin", "CURRENT", date(2026, 3, 1), _rej("med"))
    bundle = _bundle(**VALID, rejected=_bundle(meds=[taking]))

    findings = _all(bundle)

    assert findings["ANTICOAGULANT"] == ("UNKNOWN", "MISSING_REQUIRED_DATA")
    assert findings["ANTICOAG:warfarin"] == ("UNKNOWN", "MISSING_REQUIRED_DATA")


def test_rejected_plan_turns_plan_fail_into_unknown() -> None:
    rejected_plan = PlanFact("apixaban", _rej("before"), _rej("after"), _rej("plan"))
    bundle = _bundle(**VALID, meds=[_med("apixaban", "CURRENT")], rejected=_bundle(plans=[rejected_plan]))

    assert _all(bundle)["ANTICOAG:apixaban"] == ("UNKNOWN", "MISSING_REQUIRED_DATA")


def test_without_rejected_facts_evaluate_all_matches_evaluators() -> None:
    bundle = _bundle(**VALID)

    assert evaluate_all(bundle) == (
        evaluate_documentation(bundle) + evaluate_testing(bundle)
        + evaluate_anticoagulation(bundle) + evaluate_safety(bundle)
    )


def test_deferred_phase_quote_is_shown_on_plan_failure() -> None:
    deferred = EvidenceItem(source="document", doc_id="cardiology", value="after surgery: DEFERRED",
                            quote="will be addressed after surgery")
    plan = PlanFact("warfarin", _ev("before-warfarin"), None, _ev("plan-warfarin"), [deferred])

    finding = evaluate_anticoagulation(_bundle(meds=[_med("warfarin", "CURRENT")], plans=[plan]))[0]

    assert (finding.status, finding.category) == ("FAIL", "ANTICOAGULATION_MANAGEMENT")
    assert "after surgery" in finding.summary
    assert deferred in finding.evidence
