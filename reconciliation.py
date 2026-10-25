"""Work out which readings are the latest and which drugs are current. No policy rules here."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Literal

from core import EvidenceItem
from normalization import Lab, MedStatement, Vital


def _interval(at: datetime | date | None, has_time: bool) -> tuple[datetime, datetime]:
    """The time range a record could fall in: an exact moment, a whole day, or any time if undated."""

    if at is None:
        return datetime.min, datetime.max
    if not isinstance(at, datetime):
        at = datetime.combine(at, time())
    if has_time:
        return at, at
    return at.replace(hour=0, minute=0), datetime.combine(at.date(), time.max)


def _possible_latest(items: list, interval) -> list:
    """Items that could be the latest. Usually one; several when the dates are too vague to order."""

    spans = [interval(item) for item in items]
    return [
        item
        for item, (_, latest) in zip(items, spans)
        if not any(earliest > latest for earliest, _ in spans)
    ]


def latest_vital_candidates(vitals: list[Vital], kind: Literal["BP", "TEMP"]) -> list[Vital]:
    """Readings of this kind that could be the latest."""

    readings = [v for v in vitals if v.kind == kind]
    return _possible_latest(readings, lambda v: _interval(v.at, v.has_time))


def latest_lab_candidates(
    labs: list[Lab], test: Literal["CBC", "CMP"], on_or_before: date | None
) -> list[Lab]:
    """Results that could be the latest one on or before the procedure date."""

    results = [
        lab for lab in labs
        if lab.test == test and not (on_or_before and lab.at and lab.at > on_or_before)
    ]
    return _possible_latest(results, lambda lab: _interval(lab.at, False))


@dataclass
class AnticoagulantStatus:
    canonical: str
    status: Literal["CURRENT", "STOPPED", "UNKNOWN"]
    evidence: list[EvidenceItem]  # The statements that decided the status first, then older ones.


# HISTORICAL counts as stopped; UNCERTAIN becomes UNKNOWN.
_MED_STATUS = {
    "CURRENT": "CURRENT",
    "STOPPED": "STOPPED",
    "HISTORICAL": "STOPPED",
    "UNCERTAIN": "UNKNOWN",
}


def anticoagulant_statuses(meds: list[MedStatement]) -> list[AnticoagulantStatus]:
    """Current status of each anticoagulant: the most recently dated statement wins.

    Medication-list entries have no date, so any dated document statement overrides them.
    PLANNED mentions are ignored. If the newest statements disagree, the status is UNKNOWN.
    """

    results: list[AnticoagulantStatus] = []
    for canonical in sorted({m.canonical for m in meds}):
        statements = [m for m in meds if m.canonical == canonical and m.status != "PLANNED"]
        if not statements:
            continue
        dated = [m for m in statements if m.as_of is not None]
        if dated:
            newest = max(m.as_of for m in dated)
            deciding = [m for m in dated if m.as_of == newest]
        else:
            deciding = statements
        outcomes = {_MED_STATUS[m.status] for m in deciding}
        status = outcomes.pop() if len(outcomes) == 1 else "UNKNOWN"
        older = sorted(
            (m for m in statements if m not in deciding),
            key=lambda m: m.as_of or date.min,
            reverse=True,
        )
        results.append(
            AnticoagulantStatus(canonical, status, [m.evidence for m in deciding + older])
        )
    return results
