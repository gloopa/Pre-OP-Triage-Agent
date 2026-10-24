"""Single structured LLM extraction over all patient documents, with quote validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import json
from typing import Literal

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from core import Document, PatientSubmission


@cache
def _client() -> OpenAI:
    """One reusable client per process; created lazily so imports work without an API key."""

    return OpenAI(timeout=60.0, max_retries=2)


# -------------------------
# LLM output models
# -------------------------


class Evidence(BaseModel):
    doc_id: str
    quote: str  # Exact substring of the identified document's text.


class BloodPressure(BaseModel):
    systolic: float
    diastolic: float
    measured_at: str | None  # As stated in text: full date/time, or time of day only ("09:05").
    evidence: Evidence


class Temperature(BaseModel):
    value: float
    unit: Literal["C", "F"]
    measured_at: str | None  # Same convention as BloodPressure.
    evidence: Evidence


class HistoryPhysical(BaseModel):
    status: Literal["COMPLETED", "PENDING", "MISSING", "UNCERTAIN"]
    date: str | None
    evidence: Evidence


class SurgicalConsent(BaseModel):
    status: Literal["SIGNED", "UNSIGNED", "PENDING", "MISSING", "UNCERTAIN"]
    signer: Literal["PATIENT", "OTHER", "UNKNOWN"]
    signed_at: str | None
    procedure: str | None  # As stated, for the reviewer.
    matches_scheduled_procedure: bool | None  # Against the scheduled procedure; null -> UNKNOWN.
    evidence: Evidence


class Medication(BaseModel):
    name: str  # Preserve the name stated in the text.
    # CURRENT includes a temporary perioperative hold; PLANNED = future/proposed only.
    status: Literal["CURRENT", "STOPPED", "HISTORICAL", "PLANNED", "UNCERTAIN"]
    as_of: str | None
    evidence: Evidence


class AnticoagulationPlan(BaseModel):
    medication: str | None  # Matched to current anticoagulants via the Python alias map.
    before_surgery: Evidence | None
    after_surgery: Evidence | None
    evidence: Evidence


class PatientDocumentExtraction(BaseModel):
    blood_pressures: list[BloodPressure]
    temperatures: list[Temperature]
    history_physicals: list[HistoryPhysical]
    surgical_consents: list[SurgicalConsent]
    medications: list[Medication]
    anticoagulation_plans: list[AnticoagulationPlan]


def empty_extraction() -> PatientDocumentExtraction:
    return PatientDocumentExtraction(
        blood_pressures=[],
        temperatures=[],
        history_physicals=[],
        surgical_consents=[],
        medications=[],
        anticoagulation_plans=[],
    )


# -------------------------
# Prompt
# -------------------------

EXTRACTION_INSTRUCTIONS = """
You extract facts from pre-operative clinical documents. Extract only; do not evaluate
readiness, apply policy, or make any clearance decision.

General rules:
- Every fact needs evidence: the doc_id of the document it came from and a quote copied
  verbatim (exact characters, including punctuation and spacing) from that document's text.
  Keep quotes short but sufficient to support the fact.
- Extract only what the text states. Leave a nullable field null when the text does not
  support it. Never treat silence as a negative.
- Do extract explicit statements that something is missing, pending, not done, or uncertain.
- Keep contradictory facts as separate entries; do not resolve conflicts.
- Dates: copy as written in the text (any format).

Vitals:
- One blood_pressures entry per measurement; systolic and diastolic must come from the same
  reading. An arrival reading and its recheck are separate entries.
- measured_at: the date/time stated for that reading, or only the time of day (e.g. "09:05")
  if that is all the text gives, or null. Do not copy the document's metadata date.
- Temperatures: keep the value and unit as written (C or F); do not convert.

History and physical (H&P):
- COMPLETED only for an H&P that was actually performed; date is the date it was performed.
- PENDING/MISSING when the text says it is not done or not on file; UNCERTAIN if unclear.

Surgical consent:
- signer is PATIENT only when the text says the patient (or patient e-signature) signed.
  A physician signature, a consent discussion, or a form that was sent, printed, or is
  awaiting signature does not establish a patient signature.
- matches_scheduled_procedure: true if the consent is for the scheduled procedure given
  below, false if it is for a different procedure, null if the text does not say.

Medications (any medication mentioned, with its status as of that document):
- CURRENT: the patient is taking it now. A temporary perioperative hold ("hold X for 2 days
  before surgery") is still CURRENT; put the hold/resume instructions in
  anticoagulation_plans, not in the medication status.
- PLANNED: mentioned only as a future or proposed medication (e.g. a planned bridge,
  "will start"). The patient is not taking it now.
- STOPPED or HISTORICAL: actually discontinued, or taken only in the past.
- UNCERTAIN: the text says it is unclear whether the patient is taking it.
- as_of: the date the status statement applies to, if stated.

Anticoagulation plans:
- One entry per documented perioperative plan for a specific medication.
- evidence: quote identifying the plan and the medication it covers.
- before_surgery / after_surgery: quote the actual instruction for that phase (e.g. when to
  stop, when to resume). Null if that phase is not addressed, deferred, or left to someone
  else. Never infer a missing instruction or combine instructions from different medications.
""".strip()


def build_prompt(submission: PatientSubmission) -> str:
    """User input: the scheduled procedure plus every document with its metadata."""

    procedure_type = submission.procedure.procedure_type if submission.procedure else None
    payload = {
        "scheduled_procedure": procedure_type,
        "documents": [
            {
                "doc_id": doc.doc_id,
                "type": doc.type,
                "date": doc.date,
                "author": doc.author,
                "text": doc.text,
            }
            for doc in submission.documents
        ],
    }
    return "Extract facts from these documents.\n" + json.dumps(payload, ensure_ascii=False)


# -------------------------
# Extraction + validation
# -------------------------


@dataclass
class ExtractionResult:
    """Validated facts. When failed is True, facts is empty and means 'unknown', not 'absent'."""

    facts: PatientDocumentExtraction
    rejected: list[dict[str, object]] = field(default_factory=list)
    failed: bool = False
    error: str | None = None


def _evidence_problem(evidence: Evidence, docs: dict[str, Document]) -> str | None:
    doc = docs.get(evidence.doc_id)
    if doc is None:
        return "unknown doc_id"
    if not evidence.quote or evidence.quote not in (doc.text or ""):
        return "quote not found verbatim in document"
    return None


def validate_extraction(
    facts: PatientDocumentExtraction, documents: list[Document]
) -> tuple[PatientDocumentExtraction, list[dict[str, object]]]:
    """Drop facts whose evidence is not a verbatim quote from a real document."""

    docs = {doc.doc_id: doc for doc in documents if doc.doc_id}
    rejected: list[dict[str, object]] = []
    kept: dict[str, list[BaseModel]] = {}

    for name in PatientDocumentExtraction.model_fields:
        kept[name] = []
        for fact in getattr(facts, name):
            problem = _evidence_problem(fact.evidence, docs)
            if problem:
                rejected.append({"kind": name, "reason": problem, "fact": fact.model_dump()})
                continue
            if isinstance(fact, AnticoagulationPlan):
                fact = fact.model_copy()
                for phase in ("before_surgery", "after_surgery"):
                    phase_evidence = getattr(fact, phase)
                    if phase_evidence is None:
                        continue
                    phase_problem = _evidence_problem(phase_evidence, docs)
                    if phase_problem:
                        rejected.append(
                            {
                                "kind": f"anticoagulation_plans.{phase}",
                                "reason": phase_problem,
                                "fact": phase_evidence.model_dump(),
                            }
                        )
                        setattr(fact, phase, None)
            kept[name].append(fact)

    return PatientDocumentExtraction(**kept), rejected


def _failure(error: str) -> ExtractionResult:
    return ExtractionResult(facts=empty_extraction(), failed=True, error=error)


def extract_documents(submission: PatientSubmission, *, model: str) -> ExtractionResult:
    """One structured LLM request over all documents; failures are reported, never hidden."""

    if not submission.documents:
        return ExtractionResult(facts=empty_extraction())

    try:
        response = _client().responses.parse(
            model=model,
            temperature=0,
            instructions=EXTRACTION_INSTRUCTIONS,
            input=build_prompt(submission),
            text_format=PatientDocumentExtraction,
        )
    except (openai.OpenAIError, ValidationError) as exc:
        return _failure(f"{type(exc).__name__}: {exc}")

    if response.status != "completed":
        details = getattr(response, "incomplete_details", None)
        return _failure(f"response status {response.status}: {details}")

    parsed = response.output_parsed
    if parsed is None:
        refusals = [
            content.refusal
            for item in response.output or []
            if item.type == "message"
            for content in item.content or []
            if content.type == "refusal"
        ]
        return _failure("refusal: " + " ".join(refusals) if refusals else "no parsed output")

    facts, rejected = validate_extraction(parsed, submission.documents)
    return ExtractionResult(facts=facts, rejected=rejected)
