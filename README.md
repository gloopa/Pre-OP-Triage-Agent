# Pre-Op Scheduling Triage

**Ashwin Giritharan**

`triage_submission(submission, *, model)` in `core.py` returns a decision (`READY`, `NEEDS_FOLLOW_UP` or `NOT_CLEARED`), the issues blocking scheduling, and the evidence behind every check so a nurse can verify the result.

| | Baseline | My solution |
|---|---|---|
| Score (50 labeled cases) | 72.7% | 100% |
| Decisions correct | 70% | 100% |
| Issue categories correct | 52% | 100% |
| Valid output | 96% | 100% |
| Same decision across 10 replays | 80% | 100% |

Baseline figures are from running the starter implementation with the same model and harness.

## Running it

Requires [`uv`](https://docs.astral.sh/uv/) and `make`.

```bash
make test                          # 164 tests, no API key or API calls needed
export OPENAI_API_KEY="<your_api_key>"
make baseline                      # run this solution on all 50 sample cases
make evals                         # score the outputs against the labels
make determinism                   # replay one case 10 times
make report                        # browse per-case results
```

- On Windows PowerShell, set the key with `$env:OPENAI_API_KEY = "<your_api_key>"`.
- `make baseline` keeps the starter's target name but runs this solution. Let it finish all 50 cases before `make evals`, otherwise missing cases are scored as failures.
- The entry point, runner and evaluator are unchanged from the starter. `MODEL` defaults to `gpt-4.1-mini` and can be overridden, e.g. `make baseline MODEL=gpt-4.1`.

## Approach

Going through the sample data, the policy itself turned out to be simple. The hard part is that the facts it depends on are buried in long notes: a BP recheck in an addendum, an anticoagulant mentioned only by a nurse, a consent sent to the patient portal but never signed. The baseline reads everything and decides in one LLM call, making individual policy decisions harder to test and trace back to specific evidence.

So I split the work. The LLM only extracts facts from the documents, and plain Python applies the policy. Each patient goes through six steps:

1. **Read structured fields** (`normalization.py`): vitals, labs, medications and the procedure date and risk come straight from the JSON. The LLM never sees them.
2. **Extract document facts** (`extraction.py`): one LLM call per patient returns BPs, temperatures, H&Ps, consents, medications and anticoagulation plans, each with a document id and quote.
3. **Verify quotes** (`extraction.py`): a fact whose quote isn't in the document is rejected, but still shown in the output.
4. **Reconcile** (`reconciliation.py`): work out the latest BP, temperature and lab results, and which anticoagulants are current.
5. **Apply the four rules** (`rules.py`): documentation, testing, anticoagulation and safety, straight from `APPENDIX.md`. Each check returns PASS, FAIL or UNKNOWN with its evidence.
6. **Decide** (`rules.py`): a safety FAIL gives NOT_CLEARED, any other FAIL or UNKNOWN gives NEEDS_FOLLOW_UP, and otherwise the patient is READY.

## Key design decisions

- **The LLM never makes the decision.** All policy lives in Python, so every boundary (30 vs 31 days, 179 vs 180 mmHg) has a unit test instead of depending on a prompt.
- **One call per patient, with all documents.** Related facts often sit in different notes, like a medication in one and its plan in another. The API is also used properly now: one client with a timeout and retries, `temperature=0`, and strict structured output.
- **Every LLM-extracted fact needs a verified quote.** The first version used exact substring matching and rejected 59 correct facts, so quotes can now be split at line breaks or ellipses as long as each piece matches exactly. Documents are labelled `D1`, `D2` in the prompt, because the model kept miscopying the long ids.
- **Uncertainty never produces READY.** If the LLM call fails, every document-based check becomes UNKNOWN. If a rejected fact would have changed a check, that check becomes UNKNOWN and the fact is shown. This came out of v1, where a fever reading with a slightly wrong quote was dropped and the patient passed on an older, normal temperature.
- **Only uncertainty that changes the outcome is flagged.** If several readings could be the latest and they agree, the result stands. If they disagree, it's UNKNOWN. The same applies to a drug that may or may not be current: if a complete plan exists, Rule 3 passes either way.
- **The model has to choose rather than leave fields blank.** Plan phases are `STATED`, `DEFERRED` or `NOT_ADDRESSED`, medications always get a status (a temporary hold still counts as current), and consents record who signed. Plan phases became required after a prompt fix alone didn't stop the model leaving them empty.

## Output for the reviewer

The output is designed to support nurse review of non-READY cases and spot checks of READY cases:

- `findings`: every check (H&P, consent, CBC/CMP, each anticoagulant, BP, temperature), including the ones that passed, so a READY case can be verified without opening the chart.
- `issues`: one per check that didn't pass. UNKNOWN maps to `MISSING_REQUIRED_DATA`, and FAIL maps to the rule's category.
- Evidence: each item is either a quote with its document id, type and date, or a structured field path like `labs[2]`, labelled by its role (for example `after surgery: DEFERRED`).
- `rejected_document_facts`: anything the model claimed but couldn't back with a matching quote.
- `explanation`: one plain sentence summarizing the result.

An example issue from case 20 (abridged):

```json
{
  "category": "ANTICOAGULATION_MANAGEMENT", "check": "ANTICOAG:warfarin", "status": "FAIL",
  "summary": "Perioperative plan for warfarin does not say how it is managed after surgery.",
  "evidence": [
    {"doc_type": "Cardiology Progress Note", "doc_date": "2026-03-15",
     "quote": "Stop warfarin 5 days before surgery (last dose 03/15/2026)", "value": "before surgery: STATED"},
    {"doc_type": "Cardiology Progress Note", "doc_date": "2026-03-15",
     "quote": "Post-operative anticoagulation will be addressed after surgery", "value": "after surgery: DEFERRED"}
  ]
}
```

## Assumptions

- Policy comes only from `APPENDIX.md`. General knowledge is used only to read the record (brand names, °C to °F), and the labels were used only for scoring.
- Time windows are inclusive (30 days, or 14 for HIGH risk), and results dated after the procedure are ignored.
- Only completed labs count (`final`, `amended`, `corrected`). `CBC`/`LAB-CBC` and `CMP`/`LAB-CMP` are treated as the same test, and a BMP is not a CMP.
- Consent requires the patient's signature for the scheduled procedure. A physician signature or a form that was only sent doesn't count.
- Anticoagulants are warfarin, apixaban, rivaroxaban, dabigatran, edoxaban, enoxaparin and heparin, including brand names. Antiplatelets like aspirin are excluded.
- The most recently dated medication statement wins, so notes override the undated medication list. A medication missing from the list isn't assumed stopped.
- Procedure date and risk come only from structured fields. If either is missing, the checks that need it are UNKNOWN.
- If the documents can't be read, safety is UNKNOWN rather than NOT_CLEARED, since a later recheck could be in the notes.

## How I built it

I designed the architecture, extraction schema and policy rules up front, used Codex to refine them, and wrote the result into `AGENTS.md` as the plan. I then built it with Claude Code one step at a time, with tests at each step, keeping the baseline working until the new pipeline replaced it. The 164 tests cover policy boundaries, signatures, rechecks, medication status, quote checks, failed LLM calls and end-to-end runs on sample cases.

After each eval run, I grouped the misses by root cause and fixed the cause, not the individual case:

| Run | Score | What went wrong | What changed |
|---|---|---|---|
| v1 | 88.0% | Strict quote matching rejected 59 correct facts, so signed consents showed as missing | Match quotes in pieces; rejected facts make checks UNKNOWN |
| v2 | 100% | The model miscopied long document ids (no decision changed, but only by chance) | Short `D1`, `D2` ids |
| v3–v4 | 99.3% | Plan phases left empty, and a prompt fix didn't hold | Required status for each phase |
| v5 | 100% | No misses | None |


## Limitations and next steps

- 50 cases is a small sample, and the fixes were found on the same set used for scoring. Testing on more cases, especially unusual notes, would be the next step.
- A verified quote shows where a fact came from, not that the model interpreted it correctly. The nurse review covers that.
- Outputs vary slightly between runs, although the decisions stayed stable across replays.
- Some checks, like "no current anticoagulant", have no evidence to attach. These could point to the section that was searched.

