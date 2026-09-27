"""The append-only, hash-chained audit log (§3.10 NFR, build brief §13.4).

    "Every disposition, override, rule change, parameter change, model
     promotion, PHI unmask and login is recorded in an APPEND-ONLY audit log
     with actor, timestamp, before/after and reason."

Append-only is a claim about what cannot happen, so most of these tests try to
do the forbidden thing and check that it fails: deleting an entry, editing one
in place, clearing the log, or writing an entry with no attributable actor. The
hash chain is what makes the last one detectable after the fact rather than
only at the moment of writing.
"""

from __future__ import annotations

import json

import pytest

from fwa.audit.log import GENESIS_HASH, AuditEventType, AuditLog, AuditLogViolation


@pytest.fixture()
def log():
    entries = AuditLog()
    entries.record(AuditEventType.LOGIN_SUCCESS, actor="rashid", actor_role="CLAIMS_REVIEWER",
                   tenant_id="T001", subject="session", reason="Signed in.")
    entries.record(AuditEventType.DISPOSITION_RECORDED, actor="rashid",
                   actor_role="CLAIMS_REVIEWER", tenant_id="T001", subject="CASE-1",
                   reason="Cleared: the repeat was a staged procedure.",
                   before={"disposition": "PREPAY_PEND"}, after={"disposition": "CLEARED"})
    return entries


# --------------------------------------------------------------- attribution


def test_every_entry_carries_actor_time_and_reason(log):
    for event in log:
        assert event.actor
        assert event.event_time
        assert event.event_type
        assert event.sequence >= 1


def test_an_entry_with_no_actor_is_rejected(log):
    """§3.10: attributably recorded. An unattributed entry is not a record."""
    with pytest.raises(AuditLogViolation) as exc:
        log.record(AuditEventType.LOGIN_SUCCESS, actor="")
    assert "no actor" in str(exc.value)

    with pytest.raises(AuditLogViolation):
        log.record(AuditEventType.LOGIN_SUCCESS, actor="   ")


@pytest.mark.parametrize("event_type", [
    AuditEventType.MANUAL_OVERRIDE,
    AuditEventType.KILL_SWITCH_ENGAGED,
    AuditEventType.RULE_ACTIVATED,
    AuditEventType.RULE_RETIRED,
    AuditEventType.PARAMETER_CHANGED,
    AuditEventType.PHI_UNMASK,
])
def test_the_mutating_actions_require_a_stated_reason(log, event_type):
    """An override with no reason is the entry an auditor most needs and least gets."""
    with pytest.raises(AuditLogViolation) as exc:
        log.record(event_type, actor="amina", subject="X", reason="  ")
    assert "reason" in str(exc.value).lower()

    log.record(event_type, actor="amina", subject="X", reason="A stated, specific reason.")


def test_a_non_mutating_event_does_not_require_a_reason(log):
    log.record(AuditEventType.LOGIN_FAILURE, actor="unknown")


def test_before_and_after_are_both_recorded(log):
    disposition = log.by_type(AuditEventType.DISPOSITION_RECORDED)[0]
    assert disposition.before == {"disposition": "PREPAY_PEND"}
    assert disposition.after == {"disposition": "CLEARED"}


# --------------------------------------------------------------- the chain


def test_the_chain_starts_at_genesis_and_links_forward(log):
    events = list(log)
    assert events[0].prev_hash == GENESIS_HASH
    for previous, current in zip(events, events[1:]):
        assert current.prev_hash == previous.entry_hash


def test_a_fresh_log_verifies(log):
    ok, message = log.verify_chain()
    assert ok
    assert "intact" in message


def test_an_empty_log_verifies():
    ok, _ = AuditLog().verify_chain()
    assert ok


def test_sequences_are_contiguous(log):
    assert [e.sequence for e in log] == list(range(1, len(log) + 1))


def test_tampering_with_an_entry_is_detected(log):
    """The chain's purpose: an alteration after the fact is visible.

    The mutation below goes around the append-only guard deliberately — that is
    what an attacker with process access would do — and the check still catches
    it, because the stored hash no longer matches the entry's content.
    """
    events = log._events  # deliberately reaching past the public interface
    object.__setattr__(events[0], "actor", "someone_else")

    ok, message = log.verify_chain()
    assert not ok
    assert "altered" in message


def test_removing_an_entry_is_detected(log):
    log.record(AuditEventType.LOGIN_SUCCESS, actor="qa")
    del log._events[1]
    ok, message = log.verify_chain()
    assert not ok
    assert "Chain broken" in message


# ------------------------------------------------------------- append-only


def test_an_entry_cannot_be_deleted_through_the_interface(log):
    with pytest.raises(AuditLogViolation):
        del log[0]


def test_an_entry_cannot_be_replaced_through_the_interface(log):
    """``__setitem__`` is refused. Indexed *reads* are not offered at all.

    The log is iterated and queried, never addressed by position, so there is
    no getter for an assignment to look like an ordinary update of.
    """
    entry = list(log)[0]
    with pytest.raises(AuditLogViolation):
        log[0] = entry
    with pytest.raises(TypeError):
        log[0]


def test_the_log_cannot_be_cleared(log):
    with pytest.raises(AuditLogViolation):
        log.clear()
    assert len(log) == 2


# ------------------------------------------------------------------- reads


def test_reads_by_type_actor_and_subject(log):
    assert len(log.by_type(AuditEventType.LOGIN_SUCCESS)) == 1
    assert len(log.by_actor("rashid")) == 2
    assert len(log.by_subject("CASE-1")) == 1
    assert log.by_actor("nobody") == []


def test_the_dataframe_renders_for_the_audit_page(log):
    frame = log.to_dataframe()
    assert len(frame) == 2
    assert {"sequence", "actor", "event_type", "reason"} <= set(frame.columns)


def test_the_log_exports_as_jsonl(log, tmp_path):
    path = tmp_path / "audit.jsonl"
    log.dump_jsonl(path)
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["actor"] == "rashid"
    assert first["entry_hash"]


def test_an_exported_log_still_verifies_as_a_chain(log, tmp_path):
    """The export has to be checkable by someone who was not there."""
    path = tmp_path / "audit.jsonl"
    log.dump_jsonl(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").strip().splitlines()]
    previous = GENESIS_HASH
    for row in rows:
        assert row["prev_hash"] == previous
        previous = row["entry_hash"]
