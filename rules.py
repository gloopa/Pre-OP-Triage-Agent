"""Apply the four policy rules and turn the results into a decision. No LLM calls here."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from core import (
    CheckStatus,
    Decision,
    EvidenceItem,
    Finding,
    IssueCategory,
    RuleName,
    TriageIssue,
    TriageOutput,
)
from normalization import EvidenceBundle, Lab, PlanFact, Vital
from reconciliation import anticoagulant_statuses, latest_lab_candidates, latest_vital_candidates

HP_WINDOW_DAYS = 30
LAB_WINDOW_DAYS = {"LOW": 30, "MODERATE": 30, "HIGH": 14}
SBP_LIMIT = 180  # Excluded at or above this value.
DBP_LIMIT = 110  # Excluded at or above this value.
TEMP_LIMIT_F = 100.4  # Excluded only above this value; 100.4 itself is fine.


def _finding(
    rule: RuleName,
    check: str,
    status: CheckStatus,
    summary: str,
    evidence: list[EvidenceItem],
    fail_category: IssueCategory,
) -> Finding:
    """Build a finding. FAIL uses the rule's category; UNKNOWN always becomes MISSING_REQUIRED_DATA."""

    category = {"PASS": None, "FAIL": fail_category, "UNKNOWN": "MISSING_REQUIRED_DATA"}[status]
    return Finding(rule=rule, check=check, status=status, category=category, summary=summary, evidence=evidence)


def _combine(outcomes: list[bool | None]) -> CheckStatus:
    """Combine the possible latest readings: all pass is PASS, all fail is FAIL, anything else UNKNOWN."""

    if outcomes and all(o is True for o in outcomes):
        return "PASS"
    if outcomes and all(o is False for o in outcomes):
        return "FAIL"
    return "UNKNOWN"


def _days_before(procedure_date: date, at: date) -> int:
    return (procedure_date - at).days


def _in_window(procedure_date: date, at: date | None, window: int) -> bool | None:
    """True if `at` is 0 to `window` days before the procedure; None if there is no date."""

    if at is None:
        return None
    return 0 <= _days_before(procedure_date, at) <= window


def _extraction_failed_summary(bundle: EvidenceBundle, what: str) -> str:
    return f"Documents could not be read ({bundle.extraction_error}), so {what} cannot be confirmed."


# Rule 1: required documentation (H&P within 30 days, consent signed by the patient)


def _evaluate_hp(bundle: EvidenceBundle) -> Finding:
    def finding(status: CheckStatus, summary: str, evidence: list[EvidenceItem]) -> Finding:
        return _finding("DOCUMENTATION", "H&P", status, summary, evidence, "REQUIRED_DOCUMENTATION")

    if bundle.procedure_date is None:
        return finding("UNKNOWN", "Procedure date is missing, so the H&P window cannot be checked.",
                       [bundle.procedure_evidence])

    completed = sorted(
        (hp for hp in bundle.history_physicals if hp.status == "COMPLETED" and hp.at),
        key=lambda hp: hp.at,
        reverse=True,
    )
    in_window = [hp for hp in completed if _in_window(bundle.procedure_date, hp.at, HP_WINDOW_DAYS)]
    if in_window:
        hp = in_window[0]
        days = _days_before(bundle.procedure_date, hp.at)
        return finding("PASS", f"H&P completed {hp.at}, {days} days before the procedure "
                               f"(limit {HP_WINDOW_DAYS}).", [hp.evidence])

    if bundle.extraction_failed:
        return finding("UNKNOWN", _extraction_failed_summary(bundle, "the H&P"), [])

    statements = [hp.evidence for hp in bundle.history_physicals]
    unclear = [
        hp for hp in bundle.history_physicals
        if hp.status == "UNCERTAIN" or (hp.status == "COMPLETED" and hp.at is None)
    ]
    if unclear:
        return finding("UNKNOWN", "H&P status or date is unclear in the documents.", statements)
    if completed:
        hp = completed[0]
        days = _days_before(bundle.procedure_date, hp.at)
        return finding("FAIL", f"Latest completed H&P is {hp.at}, {days} days before the procedure "
                               f"(must be 0-{HP_WINDOW_DAYS}).", statements)
    return finding("FAIL", "No completed H&P found.", statements)


def _evaluate_consent(bundle: EvidenceBundle) -> Finding:
    def finding(status: CheckStatus, summary: str, evidence: list[EvidenceItem]) -> Finding:
        return _finding("DOCUMENTATION", "CONSENT", status, summary, evidence, "REQUIRED_DOCUMENTATION")

    signed_by_patient = [c for c in bundle.consents if c.status == "SIGNED" and c.signer == "PATIENT"]
    passing = [c for c in signed_by_patient if c.matches_scheduled_procedure is True]
    if passing:
        return finding("PASS", f"Consent for {bundle.procedure_type} signed by the patient.",
                       [passing[0].evidence])

    if bundle.extraction_failed:
        return finding("UNKNOWN", _extraction_failed_summary(bundle, "a patient-signed consent"), [])

    statements = [c.evidence for c in bundle.consents]
    if any(c.matches_scheduled_procedure is None for c in signed_by_patient):
        return finding("UNKNOWN", "A patient-signed consent exists but it is unclear whether it is "
                                  "for the scheduled procedure.", statements)
    if any(c.status == "SIGNED" and c.signer == "UNKNOWN" for c in bundle.consents) or any(
        c.status == "UNCERTAIN" for c in bundle.consents
    ):
        return finding("UNKNOWN", "Consent signature or signer is unclear in the documents.", statements)
    if signed_by_patient:
        return finding("FAIL", f"The only patient-signed consent is for a different procedure than "
                               f"{bundle.procedure_type}.", statements)
    if bundle.consents:
        return finding("FAIL", "No consent signed by the patient (pending, unsigned, or not signed "
                               "by the patient).", statements)
    return finding("FAIL", "No surgical consent found.", statements)


def evaluate_documentation(bundle: EvidenceBundle) -> list[Finding]:
    return [_evaluate_hp(bundle), _evaluate_consent(bundle)]


# Rule 2: required tests (CBC, plus CMP for HIGH risk), using only the latest result of each


def _evaluate_lab(bundle: EvidenceBundle, test: str, window: int) -> Finding:
    procedure_date = bundle.procedure_date
    candidates: list[Lab] = latest_lab_candidates(bundle.labs, test, procedure_date)
    evidence = [lab.evidence for lab in candidates]
    if not candidates:
        status: CheckStatus = "FAIL"
        summary = f"No completed {test} on or before the procedure date."
    else:
        status = _combine([_in_window(procedure_date, lab.at, window) for lab in candidates])
        if status == "UNKNOWN":
            summary = f"Latest {test} cannot be determined (missing or unclear result date)."
        else:
            lab = candidates[0]
            days = _days_before(procedure_date, lab.at)
            relation = "within" if status == "PASS" else "outside"
            summary = f"Latest {test} {lab.at} is {days} days before the procedure, {relation} {window} days."
    return _finding("TESTING", test, status, summary, evidence, "REQUIRED_TESTING")


def evaluate_testing(bundle: EvidenceBundle) -> list[Finding]:
    missing = [
        name for name, value in (("procedure date", bundle.procedure_date), ("procedure risk", bundle.procedure_risk))
        if value is None
    ]
    if missing:
        return [_finding("TESTING", "REQUIRED_TESTS", "UNKNOWN",
                         f"{' and '.join(missing).capitalize()} missing, so required tests cannot be checked.",
                         [bundle.procedure_evidence], "REQUIRED_TESTING")]

    window = LAB_WINDOW_DAYS[bundle.procedure_risk]
    tests = ["CBC", "CMP"] if bundle.procedure_risk == "HIGH" else ["CBC"]
    return [_evaluate_lab(bundle, test, window) for test in tests]


# Rule 3: anyone on an anticoagulant needs a plan covering before and after surgery


def _plan_evidence(plans: list[PlanFact]) -> list[EvidenceItem]:
    """Every quote from these plans: the plan itself, each phase, and any deferred phases."""

    return [
        e for p in plans
        for e in (p.evidence, p.before_surgery, p.after_surgery, *p.other_evidence)
        if e
    ]


def evaluate_anticoagulation(bundle: EvidenceBundle) -> list[Finding]:
    def finding(check: str, status: CheckStatus, summary: str, evidence: list[EvidenceItem]) -> Finding:
        return _finding("ANTICOAGULATION", check, status, summary, evidence, "ANTICOAGULATION_MANAGEMENT")

    statuses = anticoagulant_statuses(bundle.meds)
    relevant = [s for s in statuses if s.status != "STOPPED"]
    if not relevant:
        evidence = [e for s in statuses for e in s.evidence]
        if bundle.extraction_failed:
            return [finding("ANTICOAGULANT", "UNKNOWN",
                            _extraction_failed_summary(bundle, "anticoagulant use"), evidence)]
        return [finding("ANTICOAGULANT", "PASS", "No current anticoagulant.", evidence)]

    findings: list[Finding] = []
    for drug in relevant:
        check = f"ANTICOAG:{drug.canonical}"
        plans = [p for p in bundle.plans if p.canonical == drug.canonical]
        complete = [p for p in plans if p.before_surgery and p.after_surgery]
        if complete:
            # A complete plan passes even if it's unclear whether the patient still takes the drug.
            plan = complete[0]
            unclear = " (current use is unclear, but the rule is met either way)" if drug.status == "UNKNOWN" else ""
            findings.append(finding(check, "PASS", f"Perioperative plan for {drug.canonical} covers before "
                                                   f"and after surgery{unclear}.",
                                    [plan.evidence, plan.before_surgery, plan.after_surgery] + drug.evidence))
            continue
        if drug.status == "UNKNOWN":
            plan_evidence = _plan_evidence(plans)
            findings.append(finding(check, "UNKNOWN", f"Unclear whether the patient is currently taking "
                                                      f"{drug.canonical}, and there is no complete plan for it.",
                                    drug.evidence + plan_evidence))
            continue
        if bundle.extraction_failed:
            findings.append(finding(check, "UNKNOWN",
                                    _extraction_failed_summary(bundle, f"a {drug.canonical} plan"),
                                    drug.evidence))
            continue
        plan_evidence = _plan_evidence(plans)
        if not plans:
            summary = f"Patient is taking {drug.canonical} but there is no perioperative plan for it."
        else:
            gaps = sorted({
                phase for p in plans
                for phase, value in (("before", p.before_surgery), ("after", p.after_surgery))
                if value is None
            })
            summary = (f"Perioperative plan for {drug.canonical} does not say how it is managed "
                       f"{' or '.join(gaps)} surgery.")
        findings.append(finding(check, "FAIL", summary, drug.evidence + plan_evidence))
    return findings


# Rule 4: acute safety exclusions, using only the latest blood pressure and temperature


def _bp_excluded(v: Vital) -> bool:
    return v.systolic >= SBP_LIMIT or v.diastolic >= DBP_LIMIT


def _temp_excluded(v: Vital) -> bool:
    return v.temp_f > TEMP_LIMIT_F


def _evaluate_vital(bundle: EvidenceBundle, kind: str) -> Finding:
    check, label = ("BP", "blood pressure") if kind == "BP" else ("TEMPERATURE", "temperature")
    excluded = _bp_excluded if kind == "BP" else _temp_excluded
    candidates = latest_vital_candidates(bundle.vitals, kind)
    evidence = [v.evidence for v in candidates]

    def finding(status: CheckStatus, summary: str) -> Finding:
        return _finding("SAFETY", check, status, summary, evidence, "ACUTE_SAFETY_EXCLUSION")

    values = ", ".join(v.evidence.value or "?" for v in candidates)
    if bundle.extraction_failed:
        # The documents might hold a later reading or a recheck, so the result is unknown.
        # The structured readings are still shown to the reviewer.
        latest_known = f" Latest structured {label}: {values}." if candidates else ""
        return finding("UNKNOWN", _extraction_failed_summary(bundle, f"the latest {label}") + latest_known)
    if not candidates:
        return finding("UNKNOWN", f"No {label} recorded.")

    status = _combine([not excluded(v) for v in candidates])
    if status == "UNKNOWN":
        return finding("UNKNOWN", f"Latest {label} is unclear and the possible readings disagree: {values}.")
    if status == "FAIL":
        return finding("FAIL", f"Latest {label} {values} meets the exclusion threshold.")
    return finding("PASS", f"Latest {label} {values} is within range.")


def evaluate_safety(bundle: EvidenceBundle) -> list[Finding]:
    return [_evaluate_vital(bundle, "BP"), _evaluate_vital(bundle, "TEMP")]


# Decision


EVALUATORS = (evaluate_documentation, evaluate_testing, evaluate_anticoagulation, evaluate_safety)


def _with_rejected(bundle: EvidenceBundle) -> EvidenceBundle:
    """The bundle as if every rejected fact were true."""

    rejected = bundle.rejected
    return replace(
        bundle,
        vitals=bundle.vitals + rejected.vitals,
        meds=bundle.meds + rejected.meds,
        history_physicals=bundle.history_physicals + rejected.history_physicals,
        consents=bundle.consents + rejected.consents,
        plans=bundle.plans + rejected.plans,
        rejected=None,
    )


def _unverified(finding: Finding, alternative: Finding | None) -> Finding:
    """Mark a check UNKNOWN because a rejected fact would change its result."""

    if alternative is None:
        summary = f"{finding.summary} An unverified document fact would change this check; review it."
        evidence = finding.evidence
    else:
        summary = (f"{finding.summary} If the unverified document fact is correct instead: "
                   f"{alternative.summary} Review it.")
        seen = {e.model_dump_json() for e in finding.evidence}
        evidence = finding.evidence + [e for e in alternative.evidence if e.model_dump_json() not in seen]
    return _finding(finding.rule, finding.check, "UNKNOWN", summary, evidence,
                    finding.category or "MISSING_REQUIRED_DATA")


def _compare(valid: list[Finding], alternative: list[Finding]) -> list[Finding]:
    """Keep checks the rejected facts can't change; mark the others UNKNOWN."""

    alt_by_check = {f.check: f for f in alternative}
    valid_checks = {f.check for f in valid}
    findings = []
    for f in valid:
        alt = alt_by_check.get(f.check)
        if alt is not None and alt.status == f.status:
            findings.append(f)
        else:
            findings.append(_unverified(f, alt))
    # Checks that exist only if the rejected facts are true, e.g. an anticoagulant only they mention.
    findings += [
        _finding(f.rule, f.check, "UNKNOWN", f"If the unverified document fact is correct: {f.summary} Review it.",
                 f.evidence, "MISSING_REQUIRED_DATA")
        for f in alternative
        if f.check not in valid_checks
    ]
    return findings


def evaluate_all(bundle: EvidenceBundle) -> list[Finding]:
    """Run all four rules, always, in a fixed order.

    Rejected facts never count as evidence. But if one would change a check's result, that check
    becomes UNKNOWN, so a nurse reviews it instead of trusting the result.
    """

    findings: list[Finding] = []
    alternative_bundle = _with_rejected(bundle) if bundle.rejected else None
    for evaluate in EVALUATORS:
        valid = evaluate(bundle)
        findings += _compare(valid, evaluate(alternative_bundle)) if alternative_bundle else valid
    return findings


def _explanation(decision: Decision, findings: list[Finding], bundle: EvidenceBundle) -> str:
    """A short summary built from the findings, so the same findings always give the same text."""

    exclusions = [f.summary for f in findings if f.rule == "SAFETY" and f.status == "FAIL"]
    follow_ups = [f.summary for f in findings if f.status != "PASS" and f.summary not in exclusions]
    if decision == "READY":
        text = "Ready: all checks passed (" + ", ".join(f.check for f in findings) + ")."
    elif decision == "NOT_CLEARED":
        text = "Not cleared, acute safety exclusion: " + " ".join(exclusions)
        if follow_ups:
            text += " Also needs follow-up: " + " ".join(follow_ups)
    else:
        text = "Needs follow-up: " + " ".join(follow_ups)
    if bundle.rejected_facts:
        text += (f" {len(bundle.rejected_facts)} extracted document fact(s) were rejected because their "
                 "quotes could not be verified; see rejected_document_facts.")
    return text


def route(findings: list[Finding], bundle: EvidenceBundle) -> TriageOutput:
    """NOT_CLEARED on any safety FAIL, NEEDS_FOLLOW_UP on any other FAIL or UNKNOWN, else READY."""

    if any(f.rule == "SAFETY" and f.status == "FAIL" for f in findings):
        decision: Decision = "NOT_CLEARED"
    elif any(f.status != "PASS" for f in findings):
        decision = "NEEDS_FOLLOW_UP"
    else:
        decision = "READY"

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
        if f.status != "PASS"
    ]
    return TriageOutput(
        decision=decision,
        issues=issues,
        explanation=_explanation(decision, findings, bundle),
        findings=findings,
        rejected_document_facts=bundle.rejected_facts,
    )
