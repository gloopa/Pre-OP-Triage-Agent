"""Ask the LLM to pull facts out of the documents, then check every quote against the source text."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import json
import re
from typing import Literal

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from core import Document, PatientSubmission


@cache
def _client() -> OpenAI:
    """Created on first use, so this module can be imported without an API key."""

    return OpenAI(timeout=60.0, max_retries=2)


# What the LLM returns. Every fact carries a quote so a person can check it against the document.


class Evidence(BaseModel):
    doc_id: str
    quote: str  # Must appear in that document's text; checked in validate_extraction.


class BloodPressure(BaseModel):
    systolic: float
    diastolic: float
    measured_at: str | None  # As written: a full date and time, just a time ("09:05"), or None.
    evidence: Evidence


class Temperature(BaseModel):
    value: float
    unit: Literal["C", "F"]
    measured_at: str | None  # Same format as BloodPressure.measured_at.
    evidence: Evidence


class HistoryPhysical(BaseModel):
    status: Literal["COMPLETED", "PENDING", "MISSING", "UNCERTAIN"]
    date: str | None
    evidence: Evidence


class SurgicalConsent(BaseModel):
    status: Literal["SIGNED", "UNSIGNED", "PENDING", "MISSING", "UNCERTAIN"]
    signer: Literal["PATIENT", "OTHER", "UNKNOWN"]
    signed_at: str | None
    procedure: str | None  # As written on the consent; shown to the reviewer.
    matches_scheduled_procedure: bool | None  # None means the text doesn't say; the rules treat it as unknown.
    evidence: Evidence


class Medication(BaseModel):
    name: str  # As written (often a brand name); matched to a generic name in normalization.
    # A temporary hold before surgery still counts as CURRENT. PLANNED means not started yet.
    status: Literal["CURRENT", "STOPPED", "HISTORICAL", "PLANNED", "UNCERTAIN"]
    as_of: str | None
    evidence: Evidence


class PlanPhase(BaseModel):
    """One phase (before or after surgery) of an anticoagulation plan."""

    # The model must pick a status, so it can't skip a phase by leaving it empty.
    # STATED: the instruction is given. DEFERRED: left to someone else or to later.
    # NOT_ADDRESSED: not mentioned, so there is no evidence.
    status: Literal["STATED", "DEFERRED", "NOT_ADDRESSED"]
    evidence: Evidence | None


class AnticoagulationPlan(BaseModel):
    medication: str | None  # Matched to a known anticoagulant in normalization.ANTICOAGULANTS.
    before_surgery: PlanPhase
    after_surgery: PlanPhase
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


# Prompt

EXTRACTION_INSTRUCTIONS = """
You extract facts from pre-operative clinical documents. Extract only; do not evaluate
readiness, apply policy, or make any clearance decision.

General rules:
- Every fact needs evidence: the doc_id of the document it came from and a quote copied
  verbatim (exact characters, including punctuation and spacing) from that document's text.
  Use the short doc_id exactly as given for each document (e.g. "D1").
  Keep quotes short but sufficient to support the fact.
- Never skip, reorder, or paraphrase words inside a quote. If the supporting text is in
  separate places, quote each part exactly and separate the parts with "...".
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
- One entry per medication per document: put all of that document's instructions for the
  medication in a single entry, never one entry per phase.
- evidence: quote identifying the plan and the medication it covers.
- before_surgery / after_surgery: choose a status for each phase, every time.
  - STATED: the text gives the instruction for that phase (e.g. when to stop, when to resume).
    evidence quotes it. The same sentence may be quoted for evidence, before_surgery and
    after_surgery when one sentence covers them.
  - DEFERRED: the text leaves that phase to someone else or to later (e.g. "to be determined
    by the surgical team", "will be addressed after surgery"). evidence quotes that statement.
  - NOT_ADDRESSED: the text says nothing about that phase. evidence is null.
  Never infer a missing instruction or combine instructions from different medications.
""".strip()


def short_doc_ids(submission: PatientSubmission) -> dict[str, str | None]:
    """Map short ids ("D1", "D2", ...) to real doc_ids. The model miscopies long ids, so the prompt uses these."""

    return {f"D{i}": doc.doc_id for i, doc in enumerate(submission.documents, start=1)}


def build_prompt(submission: PatientSubmission) -> str:
    """The scheduled procedure plus every document, each labelled with its short id."""

    procedure_type = submission.procedure.procedure_type if submission.procedure else None
    payload = {
        "scheduled_procedure": procedure_type,
        "documents": [
            {
                "doc_id": short_id,
                "type": doc.type,
                "date": doc.date,
                "author": doc.author,
                "text": doc.text,
            }
            for short_id, doc in zip(short_doc_ids(submission), submission.documents)
        ],
    }
    return "Extract facts from these documents.\n" + json.dumps(payload, ensure_ascii=False)


def _map_doc_ids(facts: PatientDocumentExtraction, ids: dict[str, str | None]) -> PatientDocumentExtraction:
    """Swap short ids back to real doc_ids. Unknown ids are left as they are, so validation rejects them."""

    def mapped(evidence: Evidence | None) -> Evidence | None:
        if evidence is None or ids.get(evidence.doc_id) is None:
            return evidence
        return evidence.model_copy(update={"doc_id": ids[evidence.doc_id]})

    kept: dict[str, list[BaseModel]] = {}
    for name in PatientDocumentExtraction.model_fields:
        kept[name] = []
        for fact in getattr(facts, name):
            update: dict[str, object] = {"evidence": mapped(fact.evidence)}
            if isinstance(fact, AnticoagulationPlan):
                for phase in ("before_surgery", "after_surgery"):
                    plan_phase = getattr(fact, phase)
                    update[phase] = plan_phase.model_copy(update={"evidence": mapped(plan_phase.evidence)})
            kept[name].append(fact.model_copy(update=update))
    return PatientDocumentExtraction(**kept)


# Running the extraction and checking quotes


@dataclass
class ExtractionResult:
    """Facts that passed the quote check, plus the ones that didn't.

    If failed is True the documents couldn't be read, so an empty fact list means "unknown",
    not "nothing found".
    """

    facts: PatientDocumentExtraction
    rejected: list[dict[str, object]] = field(default_factory=list)
    failed: bool = False
    error: str | None = None


# A degree sign before C or F, including the broken form the model sometimes writes instead
# (a control character followed by "6"). Removed before comparing so it can't cause a mismatch.
_DEGREE = re.compile(r"(?:[°º�]|[\x00-\x08\x0b\x0c\x0e-\x1f]6?)(?=\s*[CF]\b)")
# The model may join separate parts of a document with a line break or "...".
_PIECE_BREAK = re.compile(r"\n|\.\.\.|…")


def _clean(text: str) -> str:
    """Remove degree signs and collapse whitespace, so formatting alone can't fail a match."""

    return re.sub(r"\s+", " ", _DEGREE.sub("", text)).strip()


def quote_pieces(quote: str) -> list[str]:
    """Split a quote into the parts that must each appear in the document."""

    return [piece for piece in (_clean(p) for p in _PIECE_BREAK.split(quote)) if piece]


def _evidence_problem(evidence: Evidence, docs: dict[str, Document]) -> str | None:
    """Why this evidence can't be trusted, or None if it checks out."""

    doc = docs.get(evidence.doc_id)
    if doc is None:
        return "unknown doc_id"
    text = _clean(doc.text or "")
    pieces = quote_pieces(evidence.quote or "")
    if not pieces or any(piece not in text for piece in pieces):
        return "quote not found verbatim in document"
    return None


def validate_extraction(
    facts: PatientDocumentExtraction, documents: list[Document]
) -> tuple[PatientDocumentExtraction, list[dict[str, object]]]:
    """Keep facts whose quotes appear in the named document; return the rest as rejected."""

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
                original = fact
                fact = fact.model_copy()
                for phase in ("before_surgery", "after_surgery"):
                    phase_evidence = getattr(fact, phase).evidence
                    if phase_evidence is None:
                        continue
                    phase_problem = _evidence_problem(phase_evidence, docs)
                    if phase_problem:
                        rejected.append(
                            {
                                "kind": f"anticoagulation_plans.{phase}",
                                "reason": phase_problem,
                                "fact": phase_evidence.model_dump(),
                                "plan": original.model_dump(),  # The whole plan as extracted, for the reviewer.
                            }
                        )
                        # Keep the status but drop the unverified quote. A STATED phase with no
                        # quote doesn't count as an instruction.
                        setattr(fact, phase, PlanPhase(status=getattr(fact, phase).status, evidence=None))
            kept[name].append(fact)

    return PatientDocumentExtraction(**kept), rejected


def _failure(error: str) -> ExtractionResult:
    return ExtractionResult(facts=empty_extraction(), failed=True, error=error)


def extract_documents(submission: PatientSubmission, *, model: str) -> ExtractionResult:
    """Make one LLM call over all documents. Any failure comes back as a failed result, never hidden."""

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

    facts = _map_doc_ids(parsed, short_doc_ids(submission))
    facts, rejected = validate_extraction(facts, submission.documents)
    return ExtractionResult(facts=facts, rejected=rejected)
