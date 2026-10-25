"""Submission/output schemas and the triage_submission entry point."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: ProcedureRisk | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None

RuleName = Literal["DOCUMENTATION", "TESTING", "ANTICOAGULATION", "SAFETY"]
CheckStatus = Literal["PASS", "FAIL", "UNKNOWN"]

class EvidenceItem(BaseModel):
    """One verifiable piece of evidence: a structured field path or a document quote."""

    source: Literal["structured", "document"]
    path: str | None = None
    doc_id: str | None = None
    doc_type: str | None = None
    doc_date: str | None = None
    quote: str | None = None
    value: str | None = None
    # Set when this came from an extracted fact whose quote could not be verified.
    rejected_reason: str | None = None

class Finding(BaseModel):
    """Result of one policy subcheck (e.g. H&P, CBC, BP), including PASS results."""

    rule: RuleName
    check: str
    status: CheckStatus
    category: IssueCategory | None = None
    summary: str
    evidence: list[EvidenceItem] = Field(default_factory=list)

class TriageIssue(BaseModel):

    category: IssueCategory
    evidence: list[EvidenceItem]
    rule: RuleName
    check: str
    status: Literal["FAIL", "UNKNOWN"]
    summary: str

class TriageOutput(BaseModel):
    """Structured output contract for triage responses."""

    decision: Decision
    issues: list[TriageIssue] = Field(validation_alias=AliasChoices("issues"))
    explanation: str
    findings: list[Finding] = Field(default_factory=list)  # Every subcheck, PASS included.
    # Document facts the LLM returned whose quotes failed validation; never silently dropped.
    rejected_document_facts: list[dict[str, object]] = Field(default_factory=list)


# -------------------------
# Entry point
# -------------------------


def triage_submission(
    submission: dict[str, object] | PatientSubmission,
    *,
    model: str,
) -> TriageOutput:
    """Parse -> one LLM extraction over all documents -> merge -> evaluate four rules -> route."""

    # Imported here because these modules import the schemas above.
    from extraction import extract_documents
    from normalization import build_evidence
    from rules import evaluate_all, route

    if not isinstance(submission, PatientSubmission):
        submission = PatientSubmission.model_validate(submission)

    extraction = extract_documents(submission, model=model)
    bundle = build_evidence(submission, extraction)
    return route(evaluate_all(bundle), bundle)
