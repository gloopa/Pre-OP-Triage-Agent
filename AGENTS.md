# Pre-op triage implementation

Applies to the entire repository. This describes the target implementation.

## Source of truth

**`APPENDIX.md` is the ONLY SOURCE OF TRUTH for clinical policy and assignment requirements.** Read it before implementing rules. Do not add clinical criteria from general knowledge, the starter prompt, or sample labels. Labels are evaluation data, never inference inputs. The architecture and extraction contracts below record the agreed engineering design.

## Architecture

Submission JSON -> Python structured parsing + one LLM call over ALL patient documents -> validate extracted evidence -> unified evidence model -> reconciliation -> four Python policy evaluators -> deterministic router -> final JSON.

- Python parses structured fields and labs; the LLM extracts only text-supported document facts. No LLM lab extraction or final decision call.
- Procedure date and risk come only from the structured `procedure` fields. If either is null, every check that needs it is UNKNOWN.
- Preserve source paths/document IDs, exact quotes, original values/units, clinical event times, and document metadata. Python normalizes medication aliases, dates, and temperature units.
- Anticoagulants are identified by a hardcoded Python alias map (generic + brand): warfarin/Coumadin, apixaban/Eliquis, rivaroxaban/Xarelto, dabigatran/Pradaxa, edoxaban/Savaysa, enoxaparin/Lovenox, heparin. Anything else, including antiplatelets such as aspirin and clopidogrel, is not an anticoagulant. Record this as a documented assumption.
- Labs: exclude `entered-in-error`; map codes by alias (`CBC`/`LAB-CBC`, `CMP`/`LAB-CMP`). A BMP is not a CMP.
- Document vital times: use an explicit date/time from the text if one is stated; otherwise combine the document date with the time of day stated in the text. Record which source the time came from. A same-day comparison against a reading that has no time of day is UNKNOWN.
- Reconciliation uses simple per-fact rules instead of a generic alternatives framework:
  - Vitals: keep every reading that could be the latest (the latest timed reading plus any untimed reading on that day or later); usually there is only one. A recheck is simply the later reading. BP pairs stay together. The safety evaluator marks the vital UNKNOWN only when these candidates disagree about the outcome.
  - Consent: PASS if any patient-signed consent matches the scheduled procedure; earlier pending/unsigned forms don't negate it.
  - Medications: the latest dated statement wins. An explicit UNCERTAIN, or a null structured `active` with nothing more recent, gives UNKNOWN. An omitted medication is not evidence it was stopped. A temporary perioperative hold is still CURRENT (the instruction belongs in the plan), and a PLANNED (future or proposed) medication is not one the patient is currently taking.
- Each of the four evaluators returns one finding per subcheck (H&P, consent, CBC, CMP, each anticoagulant, BP, temperature). Every finding has `PASS`, `FAIL` or `UNKNOWN` and its supporting evidence, and all of them appear in the output so the reviewer can see each check. A missing required document/test, or one outside its window, is an ordinary FAIL. UNKNOWN is only for a required input that is missing or explicitly uncertain (procedure date/risk, a vital, anticoagulant status), or for a failed extraction.
- Each non-PASS finding becomes an issue. Every UNKNOWN maps to issue category `MISSING_REQUIRED_DATA`; ordinary failures use the applicable category from `APPENDIX.md`.
- A confirmed acute safety exclusion -> `NOT_CLEARED`; otherwise any FAIL/UNKNOWN -> `NEEDS_FOLLOW_UP`; all PASS -> `READY`. A safety PASS means no exclusion, and a safety FAIL means a confirmed exclusion.

## Layout

Flat modules, no package:

```
core.py              # Input schemas (existing), output schemas (TriageOutput/TriageIssue with structured evidence),
                     # triage_submission(): parse -> extract -> reconcile -> evaluate -> route.
extraction.py        # LLM output models, prompt, cached _client(), extract_documents(), quote validation.
                     # Returns validated facts, rejected facts, and an extraction-failed flag.
normalization.py     # Dates, temp units, med/lab aliases, structured parsing, and the merged
                     # EvidenceBundle (structured + validated document facts, with provenance).
reconciliation.py    # Latest-candidate vitals/labs and anticoagulant status; no policy.
rules.py             # Only policy evaluation (four evaluators) and decision routing + explanation.
tests/
  test_extraction.py        # Mocked client: request shape, quote/doc_id rejection, failure handling.
  test_normalization.py     # Parsing, aliases, unit conversion, bundle building.
  test_reconciliation.py    # Candidate selection and medication status precedence.
  test_rules.py             # Each evaluator and the router.
  test_triage_submission.py # Integration: triage_submission end to end with a mocked extraction.
```

## OpenAI usage

- Create one reusable, lazily created OpenAI client per process (a cached accessor) rather than one client per patient, so importing without `OPENAI_API_KEY` set still works and tests can monkeypatch it. Read the API key from `OPENAI_API_KEY` through the SDK.
- Set an explicit 60-second timeout and `max_retries=2` on the client. Use SDK retries for transient failures; do not add another retry loop.
- Set `temperature=0` on every extraction request. This reduces variation but does not guarantee identical outputs.
- Make one logical extraction request per patient containing all documents, using the configured model and structured output matching `PatientDocumentExtraction`. SDK transport retries may repeat that request.
- Treat exhausted retries, refusals, and incomplete/invalid responses as extraction failures and preserve uncertainty for required facts; never silently interpret failure as an empty successful extraction.

```python
from functools import cache
from openai import OpenAI


@cache
def _client() -> OpenAI:
    return OpenAI(timeout=60.0, max_retries=2)

# Within the extraction function, include temperature=0 in the request:
# _client().responses.parse(model=model, temperature=0, text_format=PatientDocumentExtraction, ...)
```

## LLM output models

One `PatientDocumentExtraction` per patient, produced from all documents in one call. Empty lists mean no facts extracted, not confirmed absence. Nullable fields remain null when unsupported. Do not introduce inheritance or a generic fact framework.

```python
from typing import Literal
from pydantic import BaseModel


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
    matches_scheduled_procedure: bool | None  # Against procedure_type given in the prompt; null -> UNKNOWN.
    evidence: Evidence


class Medication(BaseModel):
    name: str  # Preserve the name stated in the text.
    # CURRENT includes a temporary perioperative hold; PLANNED = future/proposed only.
    status: Literal["CURRENT", "STOPPED", "HISTORICAL", "PLANNED", "UNCERTAIN"]
    as_of: str | None
    evidence: Evidence


class PlanPhase(BaseModel):
    # Required for both phases so the model must decide; only STATED with a verified quote counts.
    status: Literal["STATED", "DEFERRED", "NOT_ADDRESSED"]
    evidence: Evidence | None  # The instruction (STATED) or the deferral statement (DEFERRED).


class AnticoagulationPlan(BaseModel):
    medication: str | None  # Matched to current anticoagulants via the Python alias map.
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
```

Extraction constraints:

- Every fact must have a real document ID and a verbatim supporting quote; validate both in Python. Substring validation establishes provenance, not semantic correctness.
- A BP pair must come from the same measurement. Keep distinct readings as separate entries.
- Consent evidence must establish who signed. A physician signature, consent discussion, or statement that a form was sent does not establish a patient signature. Accept patient signing only when `status == "SIGNED"`, `signer == "PATIENT"`, and the quote explicitly supports both. Missing signer support remains UNKNOWN.
- Plan-level evidence identifies the documented plan/context; before/after evidence quotes the actual instructions. Do not infer missing instructions or combine instructions for different medications/procedures.
- Extract explicit missing/pending/uncertain statements; never interpret silence as an explicit negative. Retain contradictory facts for reconciliation.

## Implementation steps

Implement one step at a time. Each step ships with its own tests, and the full suite (`make test`) must pass before the next step starts. Keep the baseline implementation and its tests working until the new pipeline replaces it in step 6.
Throughout, preserve the existing `triage_submission(submission, *, model)` entry point and output compatibility with the baseline runner, evaluator, and report viewer.

1. **Output schema** (`core.py`): `TriageIssue` with structured evidence, `TriageOutput`. Tests: output validates against `run_evals.ScoredOutput`; evidence serializes to JSON.
2. **Extraction** (`extraction.py`): models, prompt, one structured request containing all documents, quote/doc_id validation, failure handling. Instruct extraction only, and treat a failed extraction as uncertainty, never as absence. Unit tests with a mocked client: request carries `temperature=0`, the model, and all documents; a non-verbatim quote or unknown doc_id is rejected; a refusal, invalid response or exhausted retries sets the failed flag.
3. **Normalization + structured parsing** (`rules.py`): dates, C->F, medication/lab aliases, `entered-in-error` exclusion, document-vital time assembly. Keep the internal evidence representation simple and keep provenance. Unit tests: aliases, BMP != CMP, 37.9 C -> 100.2 F, doc date + time of day.
4. **Reconciliation** (`rules.py`), using the per-fact rules above. Unit tests: latest vital wins across sources, recheck supersedes, same-day reading without a time -> UNKNOWN, superseding signed consent, med status precedence (omitted != stopped, UNCERTAIN -> UNKNOWN).
5. **Evaluators** (`rules.py`), one per rule, directly from `APPENDIX.md`. Unit tests per rule: 30/31-day and 14/15-day boundaries, SBP 179/180, DBP 109/110, temp 100.4/100.5, patient vs physician signature, procedure mismatch, plan missing before/after, plan for a different drug, null procedure date/risk.
6. **Router + wiring** (`rules.py`, `core.py`): decision, categorized issues, verifiable evidence, and a deterministic explanation; `triage_submission` end to end. Unit tests for routing precedence (exclusion -> NOT_CLEARED, FAIL/UNKNOWN -> NEEDS_FOLLOW_UP, UNKNOWN -> MISSING_REQUIRED_DATA). Integration tests: run `triage_submission` on a few sample-data submissions with a hand-written mocked extraction.
7. **Live run**, only when API execution is authorized: `make baseline`, `make evals`, `make determinism`.
