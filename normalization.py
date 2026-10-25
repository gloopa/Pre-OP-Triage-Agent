"""Normalization and structured parsing: merge structured fields and document facts into one evidence bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
import re
from typing import Literal

from core import (
    BloodPressureVital,
    Document,
    EvidenceItem,
    PatientSubmission,
    ProcedureRisk,
    TemperatureVital,
)
from extraction import (
    AnticoagulationPlan,
    BloodPressure,
    Evidence,
    ExtractionResult,
    HistoryPhysical,
    Medication,
    PatientDocumentExtraction,
    SurgicalConsent,
    Temperature,
    empty_extraction,
)

# -------------------------
# Normalization
# -------------------------

# Assumption (APPENDIX.md names no drugs): these are the anticoagulants, by generic and brand
# name. Anything else, including antiplatelets such as aspirin and clopidogrel, is not one.
ANTICOAGULANTS: dict[str, tuple[str, ...]] = {
    "warfarin": ("warfarin", "coumadin"),
    "apixaban": ("apixaban", "eliquis"),
    "rivaroxaban": ("rivaroxaban", "xarelto"),
    "dabigatran": ("dabigatran", "pradaxa"),
    "edoxaban": ("edoxaban", "savaysa"),
    "enoxaparin": ("enoxaparin", "lovenox"),
    "heparin": ("heparin",),
}

LAB_ALIASES: dict[str, str] = {"CBC": "CBC", "LAB-CBC": "CBC", "CMP": "CMP", "LAB-CMP": "CMP"}
# Completed result statuses (FHIR Observation); anything else (preliminary, registered,
# cancelled, entered-in-error, unknown, missing) is not a valid result.
COMPLETED_LAB_STATUSES = frozenset({"final", "amended", "corrected"})

_DATETIME_FORMATS = ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M")
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y")
_TIME_OF_DAY = re.compile(r"^(\d{1,2}):(\d{2})\s*([AaPp][Mm])?$")


def parse_when(value: str | None) -> tuple[datetime, bool] | None:
    """Parse a date or date/time string; returns (naive datetime, has_time).

    Timezones are dropped: all times are treated as the clinic's local time.
    """

    if not value:
        return None
    text = value.strip().removesuffix("Z")
    text = re.sub(r"([+-]\d{2}:\d{2}|\.\d+)$", "", text)
    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt), True
        except ValueError:
            pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt), False
        except ValueError:
            pass
    return None


def parse_date(value: str | None) -> date | None:
    parsed = parse_when(value)
    return parsed[0].date() if parsed else None


def parse_time_of_day(value: str | None) -> time | None:
    match = _TIME_OF_DAY.match((value or "").strip())
    if not match:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2]), (match[3] or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def to_f(value: float, unit: Literal["C", "F"]) -> float:
    """Fahrenheit, unrounded, so threshold comparisons stay exact."""

    return value * 9 / 5 + 32 if unit == "C" else value


def canonical_anticoagulant(name: str | None) -> str | None:
    tokens = set(re.findall(r"[a-z]+", (name or "").lower()))
    for canonical, aliases in ANTICOAGULANTS.items():
        if tokens.intersection(aliases):
            return canonical
    return None


# -------------------------
# Internal evidence model
# -------------------------


@dataclass
class Vital:
    kind: Literal["BP", "TEMP"]
    at: datetime | None
    has_time: bool
    time_source: Literal["structured", "text", "text_date", "doc_date+text_time", "doc_date", "none"]
    evidence: EvidenceItem
    systolic: float | None = None
    diastolic: float | None = None
    temp_f: float | None = None


@dataclass
class Lab:
    test: Literal["CBC", "CMP"]
    at: date | None
    evidence: EvidenceItem


@dataclass
class MedStatement:
    """A status statement about an anticoagulant, from the med list or a document."""

    canonical: str
    status: Literal["CURRENT", "STOPPED", "HISTORICAL", "PLANNED", "UNCERTAIN"]
    as_of: date | None  # None for structured entries, which carry no date.
    evidence: EvidenceItem


@dataclass
class HistoryPhysicalFact:
    status: Literal["COMPLETED", "PENDING", "MISSING", "UNCERTAIN"]
    at: date | None
    evidence: EvidenceItem


@dataclass
class ConsentFact:
    status: Literal["SIGNED", "UNSIGNED", "PENDING", "MISSING", "UNCERTAIN"]
    signer: Literal["PATIENT", "OTHER", "UNKNOWN"]
    matches_scheduled_procedure: bool | None
    signed_at: date | None
    evidence: EvidenceItem


@dataclass
class PlanFact:
    canonical: str | None
    before_surgery: EvidenceItem | None  # Only a STATED phase with a verified quote.
    after_surgery: EvidenceItem | None
    evidence: EvidenceItem
    # Quotes for DEFERRED phases (e.g. "to be determined by the surgical team"), for the reviewer.
    other_evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    procedure_date: date | None
    procedure_risk: ProcedureRisk | None
    procedure_type: str | None
    procedure_evidence: EvidenceItem
    vitals: list[Vital] = field(default_factory=list)
    labs: list[Lab] = field(default_factory=list)
    meds: list[MedStatement] = field(default_factory=list)
    history_physicals: list[HistoryPhysicalFact] = field(default_factory=list)
    consents: list[ConsentFact] = field(default_factory=list)
    plans: list[PlanFact] = field(default_factory=list)
    extraction_failed: bool = False
    extraction_error: str | None = None
    # LLM facts dropped by quote validation, kept so they never silently disappear.
    rejected_facts: list[dict[str, object]] = field(default_factory=list)
    # The same rejected facts as typed evidence (marked rejected_reason), never used as valid
    # evidence; the rules only use them to tell whether they could change a check.
    rejected: EvidenceBundle | None = None


# -------------------------
# Structured parsing + merge
# -------------------------


def _fmt(number: float) -> str:
    return f"{number:g}"


def _structured(path: str, value: str) -> EvidenceItem:
    return EvidenceItem(source="structured", path=path, value=value)


def _document(
    evidence: Evidence, docs: dict[str, Document], value: str | None = None, rejected_reason: str | None = None
) -> EvidenceItem:
    doc = docs.get(evidence.doc_id)
    return EvidenceItem(
        source="document",
        doc_id=evidence.doc_id,
        doc_type=doc.type if doc else None,
        doc_date=doc.date if doc else None,
        quote=evidence.quote,
        value=value,
        rejected_reason=rejected_reason,
    )


def _document_time(
    measured_at: str | None, doc: Document | None
) -> tuple[datetime | None, bool, str]:
    """Explicit text date/time, else document date + text time of day, else document date."""

    explicit = parse_when(measured_at)
    if explicit:
        return explicit[0], explicit[1], "text" if explicit[1] else "text_date"
    doc_date = parse_date(doc.date if doc else None)
    time_of_day = parse_time_of_day(measured_at)
    if doc_date and time_of_day:
        return datetime.combine(doc_date, time_of_day), True, "doc_date+text_time"
    if doc_date:
        return datetime.combine(doc_date, time()), False, "doc_date"
    return None, False, "none"


def _structured_vitals(submission: PatientSubmission) -> list[Vital]:
    vitals: list[Vital] = []
    for i, vital in enumerate(submission.vitals):
        when = parse_when(vital.date)
        at, has_time = (when[0], when[1]) if when else (None, False)
        path = f"vitals[{i}]"
        if isinstance(vital, BloodPressureVital) and vital.systolic is not None and vital.diastolic is not None:
            value = f"{_fmt(vital.systolic)}/{_fmt(vital.diastolic)} mmHg at {vital.date}"
            vitals.append(
                Vital("BP", at, has_time, "structured", _structured(path, value),
                      systolic=float(vital.systolic), diastolic=float(vital.diastolic))
            )
        elif isinstance(vital, TemperatureVital) and vital.value_f is not None:
            value = f"{_fmt(vital.value_f)} F at {vital.date}"
            vitals.append(
                Vital("TEMP", at, has_time, "structured", _structured(path, value), temp_f=float(vital.value_f))
            )
    return vitals


def _structured_labs(submission: PatientSubmission) -> list[Lab]:
    labs: list[Lab] = []
    for i, lab in enumerate(submission.labs):
        test = LAB_ALIASES.get((lab.code or "").strip().upper())
        if test is None or (lab.status or "").strip().lower() not in COMPLETED_LAB_STATUSES:
            continue
        value = f"{lab.code} ({lab.display}) {lab.status} at {lab.effective_at}"
        labs.append(Lab(test, parse_date(lab.effective_at), _structured(f"labs[{i}]", value)))
    return labs


def _structured_meds(submission: PatientSubmission) -> list[MedStatement]:
    meds: list[MedStatement] = []
    for i, med in enumerate(submission.medications):
        canonical = canonical_anticoagulant(med.name)
        if canonical is None:
            continue
        status = {True: "CURRENT", False: "STOPPED", None: "UNCERTAIN"}[med.active]
        evidence = _structured(f"medications[{i}]", f"{med.name} (active={med.active})")
        meds.append(MedStatement(canonical, status, None, evidence))
    return meds


def _doc_date(docs: dict[str, Document], doc_id: str) -> date | None:
    doc = docs.get(doc_id)
    return parse_date(doc.date if doc else None)


def _add_document_facts(
    bundle: EvidenceBundle,
    facts: PatientDocumentExtraction,
    docs: dict[str, Document],
    rejected_reason: str | None = None,
) -> None:
    def ev(evidence: Evidence, value: str | None = None) -> EvidenceItem:
        return _document(evidence, docs, value, rejected_reason)

    for bp in facts.blood_pressures:
        at, has_time, source = _document_time(bp.measured_at, docs.get(bp.evidence.doc_id))
        value = f"{_fmt(bp.systolic)}/{_fmt(bp.diastolic)} mmHg"
        bundle.vitals.append(
            Vital("BP", at, has_time, source, ev(bp.evidence, value), systolic=bp.systolic, diastolic=bp.diastolic)
        )
    for temp in facts.temperatures:
        at, has_time, source = _document_time(temp.measured_at, docs.get(temp.evidence.doc_id))
        temp_f = to_f(temp.value, temp.unit)
        value = f"{_fmt(temp.value)} {temp.unit}" + (f" ({temp_f:.1f} F)" if temp.unit == "C" else "")
        bundle.vitals.append(Vital("TEMP", at, has_time, source, ev(temp.evidence, value), temp_f=temp_f))
    for hp in facts.history_physicals:
        at = parse_date(hp.date) or (_doc_date(docs, hp.evidence.doc_id) if hp.status == "COMPLETED" else None)
        bundle.history_physicals.append(HistoryPhysicalFact(hp.status, at, ev(hp.evidence, f"{hp.status} {hp.date}")))
    for consent in facts.surgical_consents:
        value = f"{consent.status} by {consent.signer} for {consent.procedure}"
        bundle.consents.append(
            ConsentFact(
                consent.status,
                consent.signer,
                consent.matches_scheduled_procedure,
                parse_date(consent.signed_at),
                ev(consent.evidence, value),
            )
        )
    for med in facts.medications:
        canonical = canonical_anticoagulant(med.name)
        if canonical is None:
            continue
        as_of = parse_date(med.as_of) or _doc_date(docs, med.evidence.doc_id)
        bundle.meds.append(MedStatement(canonical, med.status, as_of, ev(med.evidence, f"{med.name}: {med.status}")))
    for plan in facts.anticoagulation_plans:
        instructions: dict[str, EvidenceItem | None] = {}
        other: list[EvidenceItem] = []
        for phase, label in ((plan.before_surgery, "before surgery"), (plan.after_surgery, "after surgery")):
            quote = ev(phase.evidence, f"{label}: {phase.status}") if phase.evidence else None
            instructions[label] = quote if phase.status == "STATED" else None
            if quote and phase.status != "STATED":
                other.append(quote)
        bundle.plans.append(
            PlanFact(
                canonical_anticoagulant(plan.medication),
                instructions["before surgery"],
                instructions["after surgery"],
                ev(plan.evidence, f"plan for {plan.medication}"),
                other,
            )
        )


_FACT_MODELS = {
    "blood_pressures": BloodPressure,
    "temperatures": Temperature,
    "history_physicals": HistoryPhysical,
    "surgical_consents": SurgicalConsent,
    "medications": Medication,
    "anticoagulation_plans": AnticoagulationPlan,
}


def _rejected_as_facts(record: dict[str, object]) -> PatientDocumentExtraction:
    """One rejected record back as an extraction. A rejected plan phase brings back the whole plan."""

    facts = empty_extraction()
    kind = str(record["kind"])
    if kind.startswith("anticoagulation_plans."):
        facts.anticoagulation_plans.append(AnticoagulationPlan.model_validate(record["plan"]))
    else:
        getattr(facts, kind).append(_FACT_MODELS[kind].model_validate(record["fact"]))
    return facts


def build_evidence(submission: PatientSubmission, extraction: ExtractionResult) -> EvidenceBundle:
    """Merge structured fields and validated document facts, keeping provenance."""

    procedure = submission.procedure
    docs = {doc.doc_id: doc for doc in submission.documents if doc.doc_id}
    procedure_fields = dict(
        procedure_date=parse_date(procedure.procedure_date if procedure else None),
        procedure_risk=procedure.procedure_risk if procedure else None,
        procedure_type=procedure.procedure_type if procedure else None,
        procedure_evidence=_structured(
            "procedure",
            f"{procedure.procedure_type if procedure else None}; "
            f"date={procedure.procedure_date if procedure else None}; "
            f"risk={procedure.procedure_risk if procedure else None}",
        ),
    )
    bundle = EvidenceBundle(
        **procedure_fields,
        vitals=_structured_vitals(submission),
        labs=_structured_labs(submission),
        meds=_structured_meds(submission),
        extraction_failed=extraction.failed,
        extraction_error=extraction.error,
        rejected_facts=list(extraction.rejected),
    )
    _add_document_facts(bundle, extraction.facts, docs)

    if extraction.rejected:
        bundle.rejected = EvidenceBundle(**procedure_fields)
        for record in extraction.rejected:
            reason = f"{record['kind']}: {record['reason']}"
            _add_document_facts(bundle.rejected, _rejected_as_facts(record), docs, reason)
    return bundle
