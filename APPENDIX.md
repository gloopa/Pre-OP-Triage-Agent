# Input

You will receive a single patient submission package as JSON. The structure resembles what might be returned from a healthcare API and may include:

- Patient demographics
- Procedure information
- Vital signs
- Laboratory results
- Active medications
- Clinical notes or documents

Important information may appear in free text documents. You should make reasonable assumptions and document them.

# Output

Your system must return valid JSON containing at least:

- `decision`: one of `READY`, `NEEDS_FOLLOW_UP`, `NOT_CLEARED`
- `issues`: a list in which each issue has a category, one of:
  - `REQUIRED_DOCUMENTATION`
  - `REQUIRED_TESTING`
  - `ANTICOAGULATION_MANAGEMENT`
  - `ACUTE_SAFETY_EXCLUSION`
  - `MISSING_REQUIRED_DATA`

Beyond those two fields, the rest is up to you. A pre-op nurse reviews every non-READY case and a sampling of READY cases. You should design the evidence packet so the result can be easily verified by a human reviewer. A minimal example:

```json
{
  "decision": "NEEDS_FOLLOW_UP",
  "issues": [
    {
      "category": "MISSING_REQUIRED_DATA",
      "...": "your evidence design here"
    }
  ]
}
```

# Appendix

## Cadence Surgical Center Pre-Operative Scheduling Policy

Effective: January 1, 2026

### Purpose

To establish standardized clinical requirements for determining patient readiness for surgical scheduling at Cadence Surgical Center.

### Patient Clearance Status

Each patient must be assigned exactly one of the following pre-operative clearance statuses:

- `READY` – All required documentation, testing, and clinical criteria are complete and within defined timeframes.
- `NEEDS_FOLLOW_UP` – Required documentation, testing, or perioperative planning is missing, incomplete, or outside required timeframes.
- `NOT_CLEARED` – The patient has a safety condition that precludes proceeding with the surgery.

### Rule 1: Required Documentation

The following documentation must be present prior to scheduling:

1. History and Physical (H&P)
   - Must be completed within 30 days of the scheduled procedure date.
2. Signed Surgical Consent

If any required documentation is missing or outdated → `NEEDS_FOLLOW_UP`

### Rule 2: Required Pre-Operative Testing by Procedure Risk Level

Each procedure must be categorized as LOW, MODERATE, or HIGH risk.

#### 2.1 LOW or MODERATE Risk Procedures

- Complete Blood Count (CBC) within 30 days of the procedure date.

#### 2.2 HIGH Risk Procedures

- Complete Blood Count (CBC) within 14 days
- Comprehensive Metabolic Panel (CMP) within 14 days

#### Testing Review Standards

- Only the most recent result for each required test shall be considered.
- Results must fall within the specified timeframe relative to the procedure date.
- If any required test is missing or outside the permitted timeframe, classify the patient as `NEEDS_FOLLOW_UP`

### Rule 3: Anticoagulation Management

For patients currently taking anticoagulant medication:

- A documented perioperative anticoagulation management plan is required prior to scheduling.
- The plan must clearly describe how the medication will be managed before and after the procedure.

If there is no documented plan, or the plan is incomplete or ambiguous, classify the patient as `NEEDS_FOLLOW_UP`.

### Rule 4: Acute Safety Exclusion Criteria

The presence of any of the following conditions at the time of pre-operative review will result in `NOT_CLEARED` status:

- Systolic blood pressure ≥ 180 mmHg
- Diastolic blood pressure ≥ 110 mmHg
- Temperature > 100.4°F

Only the most recent result for each required test shall be considered. A rechecked vital within range is sufficient.

### Final Clearance Determination

A patient may be classified as `READY` only when:

- All required documentation is present and current
- All required testing is complete and within defined timeframes
- Anticoagulation management (if applicable) is clearly documented
- No acute safety exclusion criteria are present

If a required field needed to evaluate a rule is missing/unknown, the patient should be classified as `NEEDS_FOLLOW_UP`.

### Patient Submission Format

A trimmed example is below; see `data/patients_sample_50.jsonl` for full cases. Real documents are much longer, and many deciding facts appear only in their text.

#### Example Input

```json
{
  "patient": {
    "id": "14401b46-16cf-5c04-b43a-fec3e8485985",
    "mrn": "MRN-3000000",
    "name": {
      "given": "Danielle",
      "family": "Lopez"
    },
    "dob": "1946-01-01",
    "sex": "F"
  },
  "procedure": {
    "case_id": "case_00000",
    "procedure_type": "Elective total knee arthroplasty",
    "procedure_risk": "MODERATE",
    "procedure_date": null,
    "is_elective": true,
    "location": "Cadence Surgical Center - Main OR"
  },
  "vitals": [
    {
      "type": "blood_pressure",
      "systolic": 117,
      "diastolic": 72,
      "date": "2026-02-17T09:05:00Z",
      "source": "Pre-op clinic"
    },
    {
      "type": "blood_pressure",
      "systolic": 122,
      "diastolic": 76,
      "date": "2026-02-24T10:12:00Z",
      "source": "Primary care visit"
    },
    {
      "type": "temperature",
      "value_f": 98.8,
      "date": "2026-02-24T10:15:00Z",
      "source": "Pre-op clinic"
    }
  ],
  "labs": [
    {
      "id": "69d20291-f423-54f9-9d5c-14b6608351c4",
      "code": "CBC",
      "display": "Complete Blood Count",
      "effective_at": "2026-02-21T08:10:00Z",
      "status": "final",
      "source": "Internal Hospital Lab"
    },
    {
      "id": "58de471c-f6b4-5567-9b57-819f93e57e67",
      "code": "CBC",
      "display": "CBC (Complete Blood Count)",
      "effective_at": "2026-02-03T08:25:00Z",
      "status": "final",
      "source": "Internal Hospital Lab"
    },
    {
      "id": "682bf9a8-3325-57bf-a231-5f8ad1185692",
      "code": "HBA1C",
      "display": "Hemoglobin A1c",
      "effective_at": "2026-02-09T07:40:00Z",
      "status": "final",
      "source": "Regional Reference Lab"
    }
  ],
  "medications": [
    {
      "name": "lisinopril",
      "active": true
    },
    {
      "name": "apixaban",
      "active": true
    }
  ],
  "conditions": [
    {
      "name": "Atrial fibrillation",
      "active": true
    }
  ],
  "documents": [
    {
      "doc_id": "c92f23f5-bc1f-54cb-911f-c719bc7df37b",
      "type": "History and Physical",
      "date": "2026-02-19",
      "author": "Danielle Rivera, APRN",
      "text": "HISTORY AND PHYSICAL: pre-op evaluation complete for planned procedure. Procedure date target: 2026-03-01."
    },
    {
      "doc_id": "bdb55172-db02-5721-a48e-81b81b300312",
      "type": "Preop Pre-anesthesia Evaluation [PDF]",
      "date": "2026-01-30",
      "author": "Brian N. Murphy, MSN, NP-C",
      "text": "Prior pre-op H&P retained for longitudinal chart context."
    },
    {
      "doc_id": "ffd06a2d-6fc3-5243-9189-6054e87f1c86",
      "type": "Surgical Consent",
      "date": "2026-02-23",
      "author": "Brian O'Connor, DO",
      "text": "Electronic consent obtained and signed by patient for procedure."
    },
    {
      "doc_id": "43d8b80c-0050-5954-a60a-4964b07f0ae3",
      "type": "Pre-op Nursing Intake",
      "date": "2026-02-25",
      "author": "Sarah Murphy, DNP, FNP-BC",
      "text": "Nursing intake confirms home medications and fasting instructions were reviewed."
    },
    {
      "doc_id": "0e230e4e-469c-503f-9ad7-9ce6d086ab84",
      "type": "Perioperative Medicine Consult",
      "date": "2026-02-22",
      "author": "Stephanie Rodriguez, DO",
      "text": "CARDIOLOGY CONSULTATION - PERIOPERATIVE\n... maintained on Xarelto 20 mg PO daily ...\nRECOMMENDATIONS:\n  1. Hold Xarelto for the 2 days before surgery (last dose 02/26/2026) ..."
    }
  ],
  "metadata": {
    "submission_received_at": "2026-03-01T14:22:10Z",
    "source_system": "Cadence Intake API v2"
  }
}
```
