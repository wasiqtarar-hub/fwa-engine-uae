"""The as-of join at service time (manuscript §3.6, §6.2 category 5).

    "Every join from a claim to coverage, contract, tariff, code, licence,
     network, edit or model reference data must be an AS-OF JOIN evaluated at
     SERVICE TIME, not submission time... Corrections must never overwrite a
     prior version."

This is the requirement a system can ignore while still appearing to work: it
simply applies today's policy to a two-year-old claim and answers confidently.
The tests are therefore mostly about what the service *refuses* to do — resolve
without a date, fall back to the latest version, or let a correction overwrite
the version that was actually in force.
"""

from __future__ import annotations

import datetime as _dt

import pandas as pd
import pytest

from fwa.reference.asof import AsOfReferenceService, ReferenceNotEffective

V1_FROM, V1_TO = _dt.date(2024, 1, 1), _dt.date(2025, 6, 30)
V2_FROM, V2_TO = _dt.date(2025, 7, 1), _dt.date(2099, 12, 31)


@pytest.fixture()
def service():
    """Two versions of one benefit rule, with a clean handover on 1 July 2025."""
    svc = AsOfReferenceService()
    svc.register("benefit_rule", "BR-001", {"co_pay": 0.10},
                 valid_from=V1_FROM, valid_to=V1_TO,
                 version_id="1.0.0", source="2024 policy schedule")
    svc.register("benefit_rule", "BR-001", {"co_pay": 0.20},
                 valid_from=V2_FROM, valid_to=V2_TO,
                 version_id="2.0.0", source="2025 policy schedule")
    return svc


# ------------------------------------------------ the §6.2 effective-date test


def test_a_claim_resolves_to_the_version_in_force_on_its_service_date(service):
    """Two policy versions, a claim between them: the earlier one applies."""
    assert service.resolve("benefit_rule", "BR-001", _dt.date(2025, 3, 1)).version_id == "1.0.0"
    assert service.resolve("benefit_rule", "BR-001", _dt.date(2025, 9, 1)).version_id == "2.0.0"


def test_the_resolved_payload_is_the_one_that_was_in_force(service):
    assert service.resolve("benefit_rule", "BR-001",
                           _dt.date(2025, 3, 1)).payload["co_pay"] == 0.10
    assert service.resolve("benefit_rule", "BR-001",
                           _dt.date(2025, 9, 1)).payload["co_pay"] == 0.20


@pytest.mark.parametrize("day,expected", [
    (V1_FROM, "1.0.0"),                                  # first day of v1
    (V1_TO, "1.0.0"),                                    # last day of v1
    (V2_FROM, "2.0.0"),                                  # first day of v2
])
def test_the_boundaries_are_inclusive_and_unambiguous(service, day, expected):
    assert service.resolve("benefit_rule", "BR-001", day).version_id == expected


def test_a_late_submission_still_resolves_at_service_time(service):
    """The whole point: a 2025 claim submitted in 2026 gets 2025's policy."""
    service_date = _dt.date(2025, 3, 1)
    resolved = service.resolve("benefit_rule", "BR-001", service_date)
    assert resolved.version_id == "1.0.0"
    assert resolved.version_id != service.resolve(
        "benefit_rule", "BR-001", _dt.date(2026, 3, 1)).version_id


# ------------------------------------------------------------- the refusals


def test_a_date_before_every_version_raises_rather_than_using_the_earliest(service):
    with pytest.raises(ReferenceNotEffective) as exc:
        service.resolve("benefit_rule", "BR-001", _dt.date(2020, 1, 1))
    assert "will not fall back" in str(exc.value)


def test_an_unknown_key_raises(service):
    with pytest.raises(ReferenceNotEffective):
        service.resolve("benefit_rule", "BR-999", _dt.date(2025, 3, 1))


def test_resolving_without_a_service_date_is_impossible(service):
    """There is no no-argument form, and a missing date is fatal, not today's."""
    with pytest.raises((ValueError, TypeError)):
        service.resolve("benefit_rule", "BR-001", None)


def test_the_error_message_lists_the_windows_that_do_exist(service):
    """A refusal has to be actionable, or it just moves the problem."""
    with pytest.raises(ReferenceNotEffective) as exc:
        service.resolve("benefit_rule", "BR-001", _dt.date(2020, 1, 1))
    message = str(exc.value)
    assert V1_FROM.isoformat() in message
    assert V2_FROM.isoformat() in message


def test_try_resolve_returns_none_instead_of_raising(service):
    assert service.try_resolve("benefit_rule", "BR-001", _dt.date(2020, 1, 1)) is None
    assert service.try_resolve("benefit_rule", "BR-001", _dt.date(2025, 3, 1)) is not None


# ------------------------------------------------------------ no overwriting


def test_a_correction_is_a_new_version_not_an_overwrite(service):
    """§3.6: "Corrections must never overwrite a prior version"."""
    service.register("benefit_rule", "BR-001", {"co_pay": 0.25},
                     valid_from=_dt.date(2026, 1, 1), version_id="2.1.0",
                     source="2026 correction")
    versions = service.versions("benefit_rule", "BR-001")
    assert [v.version_id for v in versions] == ["1.0.0", "2.0.0", "2.1.0"]
    # The old answer is still retrievable, which is the point.
    assert service.resolve("benefit_rule", "BR-001",
                           _dt.date(2025, 3, 1)).payload["co_pay"] == 0.10


def test_re_registering_the_same_version_and_date_is_refused(service):
    with pytest.raises(ValueError) as exc:
        service.register("benefit_rule", "BR-001", {"co_pay": 0.99},
                         valid_from=V1_FROM, valid_to=V1_TO,
                         version_id="1.0.0", source="an attempted overwrite")
    assert "never overwrite" in str(exc.value)


def test_every_record_carries_its_provenance(service):
    for record in service.versions("benefit_rule", "BR-001"):
        assert record.version_id
        assert record.source
        assert record.recorded_at is not None
        assert record.valid_from <= record.valid_to


# --------------------------------------------------------------- governance


def test_overlapping_windows_are_reported_as_a_defect():
    """Overlap is not resolved silently; it is surfaced for someone to fix."""
    svc = AsOfReferenceService()
    svc.register("tariff", "T-1", {"price": 100}, valid_from=_dt.date(2025, 1, 1),
                 valid_to=_dt.date(2025, 12, 31), version_id="1.0.0", source="s")
    svc.register("tariff", "T-1", {"price": 120}, valid_from=_dt.date(2025, 6, 1),
                 valid_to=_dt.date(2026, 12, 31), version_id="2.0.0", source="s")

    defects = svc.overlap_defects()
    assert defects
    assert defects[0]["domain"] == "tariff"


def test_no_overlap_means_no_defects(service):
    assert service.overlap_defects() == []


def test_the_version_stamp_is_what_a_signal_stores(service):
    assert service.version_stamp("benefit_rule", "BR-001",
                                 _dt.date(2025, 3, 1)) == "benefit_rule@1.0.0"


def test_an_unresolvable_stamp_says_so_rather_than_being_blank(service):
    """A blank version on a signal would be indistinguishable from an unversioned one."""
    assert service.version_stamp("benefit_rule", "BR-001",
                                 _dt.date(2020, 1, 1)) == "benefit_rule@NOT_EFFECTIVE"


def test_domains_and_length_report_what_is_loaded(service):
    assert service.domains() == ["benefit_rule"]
    assert len(service) == 2


# ---------------------------------------------------------------- bulk load


def test_a_frame_of_reference_rows_loads():
    svc = AsOfReferenceService()
    frame = pd.DataFrame([
        {"code": "J18.9", "chapter": "Respiratory", "valid_from": "2024-01-01",
         "valid_to": "2099-12-31", "version_id": "icd10-2024", "source": "WHO ICD-10"},
        {"code": "E11.9", "chapter": "Endocrine", "valid_from": "2024-01-01",
         "valid_to": "2099-12-31", "version_id": "icd10-2024", "source": "WHO ICD-10"},
    ])
    assert svc.register_frame("code", frame, key_columns=["code"]) == 2
    assert svc.resolve("code", "J18.9", _dt.date(2025, 1, 1)).payload["chapter"] == "Respiratory"
