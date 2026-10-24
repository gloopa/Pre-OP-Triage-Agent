from __future__ import annotations

from datetime import date, datetime

from core import EvidenceItem
from normalization import Lab, MedStatement, Vital
from reconciliation import (
    anticoagulant_statuses,
    latest_lab_candidates,
    latest_vital_candidates,
)


def _ev(label: str) -> EvidenceItem:
    return EvidenceItem(source="document", doc_id=label)


def _bp(label: str, at: datetime | None, has_time: bool = True, sbp: float = 120) -> Vital:
    return Vital("BP", at, has_time, "text", _ev(label), systolic=sbp, diastolic=80)


def _labels(items: list) -> list[str]:
    return [item.evidence.doc_id or item.evidence.path for item in items]


def test_later_document_bp_beats_structured_bp() -> None:
    structured = Vital("BP", datetime(2026, 3, 1, 10, 12), True, "structured",
                       EvidenceItem(source="structured", path="vitals[0]"), systolic=122, diastolic=76)
    document = Vital("BP", datetime(2026, 3, 3, 9, 5), True, "doc_date+text_time",
                     _ev("note-1"), systolic=157, diastolic=117)

    candidates = latest_vital_candidates([structured, document], "BP")

    assert [(c.systolic, c.evidence.source) for c in candidates] == [(157, "document")]


def test_recheck_beats_arrival_reading() -> None:
    arrival = _bp("arrival", datetime(2026, 3, 15, 8, 6), sbp=189)
    recheck = _bp("recheck", datetime(2026, 3, 15, 8, 30), sbp=137)

    assert _labels(latest_vital_candidates([arrival, recheck], "BP")) == ["recheck"]


def test_untimed_reading_on_latest_day_keeps_both_candidates() -> None:
    timed = _bp("timed", datetime(2026, 3, 15, 8, 30))
    untimed = _bp("untimed", datetime(2026, 3, 15), has_time=False)
    older = _bp("older", datetime(2026, 3, 14, 23, 0))

    assert _labels(latest_vital_candidates([timed, untimed, older], "BP")) == ["timed", "untimed"]


def test_untimed_reading_on_later_day_wins_outright() -> None:
    timed = _bp("timed", datetime(2026, 3, 14, 8, 30))
    untimed = _bp("untimed", datetime(2026, 3, 15), has_time=False)

    assert _labels(latest_vital_candidates([timed, untimed], "BP")) == ["untimed"]


def test_vital_candidates_filter_by_kind_and_handle_empty() -> None:
    temp = Vital("TEMP", datetime(2026, 3, 15, 9, 0), True, "text", _ev("t"), temp_f=99.0)

    assert latest_vital_candidates([temp], "BP") == []
    assert _labels(latest_vital_candidates([temp], "TEMP")) == ["t"]


def _lab(label: str, at: date | None, test: str = "CBC") -> Lab:
    return Lab(test, at, _ev(label))


def test_latest_lab_on_or_before_procedure() -> None:
    labs = [
        _lab("old", date(2026, 2, 1)),
        _lab("new", date(2026, 3, 1)),
        _lab("after", date(2026, 3, 12)),
        _lab("cmp", date(2026, 3, 5), test="CMP"),
    ]

    assert _labels(latest_lab_candidates(labs, "CBC", date(2026, 3, 10))) == ["new"]
    assert _labels(latest_lab_candidates(labs, "CMP", date(2026, 3, 10))) == ["cmp"]
    assert latest_lab_candidates([], "CBC", date(2026, 3, 10)) == []


def test_undated_lab_is_kept_as_a_candidate() -> None:
    labs = [_lab("dated", date(2026, 3, 1)), _lab("undated", None)]

    assert _labels(latest_lab_candidates(labs, "CBC", date(2026, 3, 10))) == ["dated", "undated"]


def _med(label: str, status: str, as_of: date | None, canonical: str = "rivaroxaban") -> MedStatement:
    return MedStatement(canonical, status, as_of, _ev(label))


def _statuses(meds: list[MedStatement]) -> dict[str, str]:
    return {s.canonical: s.status for s in anticoagulant_statuses(meds)}


def test_document_mention_without_med_list_entry_is_current() -> None:
    assert _statuses([_med("note", "CURRENT", date(2026, 2, 26))]) == {"rivaroxaban": "CURRENT"}


def test_structured_inactive_and_document_stopped() -> None:
    meds = [_med("medications[0]", "STOPPED", None, "warfarin"), _med("hp", "HISTORICAL", date(2026, 2, 23), "warfarin")]

    assert _statuses(meds) == {"warfarin": "STOPPED"}


def test_dated_document_statement_overrides_structured_entry() -> None:
    meds = [_med("medications[0]", "CURRENT", None, "warfarin"), _med("note", "STOPPED", date(2026, 3, 17), "warfarin")]

    assert _statuses(meds) == {"warfarin": "STOPPED"}


def test_latest_document_statement_wins() -> None:
    meds = [_med("old", "STOPPED", date(2026, 1, 5)), _med("new", "CURRENT", date(2026, 3, 1))]

    result = anticoagulant_statuses(meds)[0]

    assert result.status == "CURRENT"
    assert _labels_of(result.evidence) == ["new", "old"]


def _labels_of(evidence: list[EvidenceItem]) -> list[str | None]:
    return [e.doc_id for e in evidence]


def test_uncertain_structured_and_document_is_unknown() -> None:
    meds = [_med("medications[0]", "UNCERTAIN", None, "warfarin"), _med("note", "UNCERTAIN", date(2026, 3, 31), "warfarin")]

    assert _statuses(meds) == {"warfarin": "UNKNOWN"}


def test_conflicting_statements_on_same_date_are_unknown() -> None:
    meds = [_med("a", "CURRENT", date(2026, 3, 1)), _med("b", "STOPPED", date(2026, 3, 1))]

    assert _statuses(meds) == {"rivaroxaban": "UNKNOWN"}


def test_hold_instruction_keeps_drug_current() -> None:
    # Extraction labels a perioperative hold as CURRENT; the hold itself lives in the plan.
    meds = [_med("medications[0]", "CURRENT", None), _med("consult", "CURRENT", date(2026, 2, 20))]

    assert _statuses(meds) == {"rivaroxaban": "CURRENT"}


def test_planned_bridge_is_not_a_current_anticoagulant() -> None:
    meds = [
        _med("medications[0]", "CURRENT", None, "warfarin"),
        _med("consult", "PLANNED", date(2026, 2, 24), "enoxaparin"),
    ]

    assert _statuses(meds) == {"warfarin": "CURRENT"}


def test_planned_mention_does_not_override_current_status() -> None:
    meds = [_med("note", "CURRENT", date(2026, 2, 20)), _med("plan", "PLANNED", date(2026, 3, 1))]

    assert _statuses(meds) == {"rivaroxaban": "CURRENT"}


def test_omitted_medication_is_not_stopped() -> None:
    # The med list omits the drug; only the document mentions it, and it stays CURRENT.
    assert _statuses([_med("note", "CURRENT", date(2026, 3, 25), "apixaban")]) == {"apixaban": "CURRENT"}
    assert _statuses([]) == {}
