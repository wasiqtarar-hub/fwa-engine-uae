"""Unlocked NET controls: each fires on a minimal planted fixture, is silent on its
clean twin, survives empty tables, and is idempotent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fwa.canonical import CanonicalDataset
from fwa.engine.context import ControlContext
from fwa.engine.unlocked import net as N

ROOT = Path(__file__).resolve().parents[2]


def _control(registry, rule_id):
    c = registry.get(rule_id)
    return c.model_copy(update={"data_support": c.data_support.__class__("PARTIAL"),
                                "data_support_reason": "unit-test fixture"})


def _ctx(config, tables, claims):
    ds = CanonicalDataset.empty(source_system="fixture")
    ds.set("claim_header", claims, "POPULATED", "fixture")
    for name, frame in tables.items():
        ds.set(name, frame, "POPULATED", "fixture")
    return ControlContext(config=config, tenant_id="T001", claims=claims, features=None, dataset=ds,
                          run_date=pd.Timestamp("2026-09-20").date())


def _claim(sk, member, provider, date="2025-01-10", amount=1000.0, **kw):
    row = {"claim_sk": sk, "member_sk": member, "provider_sk": provider, "tenant_id": "T001",
           "service_date": pd.Timestamp(date), "gross_amount_aed": amount, "patient_share": 20.0,
           "agent_id": "AG0"}
    row.update(kw)
    return row


def _providers(n=12, per=None):
    rows = []
    for i in range(n):
        rows.append({"provider_sk": f"P{i}", "provider_type": "clinic", "specialty": "general",
                     "owner_entity_id": f"OWN{i}", "bank_account_token": f"BANK{i}",
                     "phone_token": f"PH{i}", "address_token": f"ADDR{i}", "emirate": "Dubai",
                     "regulator_id": f"REG{i}", "credentialing_date": pd.Timestamp("2020-01-01"),
                     "tenant_id": "T001"})
    df = pd.DataFrame(rows).set_index("provider_sk")
    for (p, col), v in (per or {}).items():
        df.loc[p, col] = v
    return df.reset_index()


# ------------------------------------------------------------------ NET-01-R02

def _loop(planted: bool):
    rng = np.random.default_rng(1)
    rows = []
    k = 0
    for _ in range(300):  # background: referrals to the "hospitals" P8..P11
        a = f"P{rng.integers(0, 8)}"
        b = f"P{rng.integers(8, 12)}"
        rows.append((a, b))
    if planted:
        rows += [("P1", "P2")] * 12 + [("P2", "P1")] * 10
    ref = pd.DataFrame([{"referral_sk": f"R{i}", "referrer_provider_sk": a, "recipient_provider_sk": b,
                         "member_sk": f"M{i}", "referral_date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i % 200),
                         "tenant_id": "T001", "resulting_claim_sk": None} for i, (a, b) in enumerate(rows)])
    claims = pd.DataFrame([_claim("C0", "M0", "P1")])
    return {"referral": ref, "provider": _providers()}, claims


def test_net_01_r02_fires_on_loop_and_is_silent_without(config, registry):
    ctrl = _control(registry, "NET-01-R02")
    tables, claims = _loop(True)
    sigs = N.net_01_r02_reciprocal_referral_loop(_ctx(config, tables, claims), ctrl)
    assert len(sigs) == 1 and sigs[0].subject_id == "P1" and sigs[0].evidence["plain_language"]
    tables, claims = _loop(False)
    assert N.net_01_r02_reciprocal_referral_loop(_ctx(config, tables, claims), ctrl) == []


def test_net_01_r02_same_owner_is_a_team(config, registry):
    ctrl = _control(registry, "NET-01-R02")
    tables, claims = _loop(True)
    tables["provider"].loc[tables["provider"]["provider_sk"].isin(["P1", "P2"]), "owner_entity_id"] = "OWNX"
    assert N.net_01_r02_reciprocal_referral_loop(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ NET-01-R03

def _conversion(planted: bool):
    rng = np.random.default_rng(2)
    refs, claims, lines = [], [], []
    for i in range(400):
        dst = f"P{8 + i % 4}"
        c = f"C{i}"
        high = rng.random() < 0.08
        if planted and dst == "P8":
            high = rng.random() < 0.7
        refs.append({"referral_sk": f"R{i}", "referrer_provider_sk": f"P{i % 8}", "recipient_provider_sk": dst,
                     "member_sk": f"M{i}", "referral_date": pd.Timestamp("2025-01-01"),
                     "reason_code": "knee pain", "resulting_claim_sk": c, "tenant_id": "T001"})
        claims.append(_claim(c, f"M{i}", dst, amount=20000.0 if high else 800.0))
        lines.append({"line_sk": f"L{i}", "claim_sk": c, "activity_code": "X", "tenant_id": "T001",
                      "net_amount": 20000.0 if high else 800.0})
    prov = _providers()
    return {"referral": pd.DataFrame(refs), "claim_line": pd.DataFrame(lines), "provider": prov}, pd.DataFrame(claims)


def test_net_01_r03_fires_on_converting_recipient_and_is_silent_otherwise(config, registry):
    ctrl = _control(registry, "NET-01-R03")
    tables, claims = _conversion(True)
    sigs = N.net_01_r03_high_cost_conversion(_ctx(config, tables, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P8"] and sigs[0].evidence["plain_language"]
    assert sigs[0].exposure_aed > 0
    tables, claims = _conversion(False)
    assert N.net_01_r03_high_cost_conversion(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ NET-01-R04

def _chain(planted: bool, declared_owner: bool = False):
    prov = _providers()
    prov.loc[prov["provider_sk"].isin(["P5", "P6"]), "provider_type"] = "laboratory"
    prov.loc[prov["provider_sk"] == "P7", "provider_type"] = "imaging centre"
    if planted:
        prov.loc[prov["provider_sk"] == "P5", "bank_account_token"] = "BANK1"
        prov.loc[prov["provider_sk"] == "P7", "phone_token"] = "PH1"
    if declared_owner:
        prov.loc[prov["provider_sk"].isin(["P1", "P5", "P7"]), "owner_entity_id"] = "OWN1"
    refs = []
    for i in range(20):
        dst = ["P5", "P7"][i % 2] if i < 16 else "P6"
        refs.append({"referral_sk": f"R{i}", "referrer_provider_sk": "P1", "recipient_provider_sk": dst,
                     "member_sk": f"M{i}", "referral_date": pd.Timestamp("2025-02-01"),
                     "resulting_claim_sk": f"C{i}", "tenant_id": "T001"})
    claims = pd.DataFrame([_claim(f"C{i}", f"M{i}", r["recipient_provider_sk"]) for i, r in enumerate(refs)])
    return {"referral": pd.DataFrame(refs), "provider": prov}, claims


def test_net_01_r04_fires_on_hidden_chain_and_respects_declared_owner(config, registry):
    ctrl = _control(registry, "NET-01-R04")
    tables, claims = _chain(True)
    sigs = N.net_01_r04_closed_downstream_chain(_ctx(config, tables, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P1"] and sigs[0].evidence["plain_language"]
    tables, claims = _chain(False)
    assert N.net_01_r04_closed_downstream_chain(_ctx(config, tables, claims), ctrl) == []
    tables, claims = _chain(True, declared_owner=True)
    assert N.net_01_r04_closed_downstream_chain(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ NET-02-R01

def test_net_02_r01_fires_on_shared_bank_and_excludes_known_group(config, registry):
    ctrl = _control(registry, "NET-02-R01")
    claims = pd.DataFrame([_claim("C0", "M0", "P3")])
    prov = _providers()
    prov.loc[prov["provider_sk"].isin(["P3", "P4"]), "bank_account_token"] = "BANKX"
    sigs = N.net_02_r01_shared_admin_identity(_ctx(config, {"provider": prov}, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P3"] and sigs[0].evidence["plain_language"]
    assert N.net_02_r01_shared_admin_identity(_ctx(config, {"provider": _providers()}, claims), ctrl) == []
    prov.loc[prov["provider_sk"].isin(["P3", "P4"]), "owner_entity_id"] = "OWNG"
    assert N.net_02_r01_shared_admin_identity(_ctx(config, {"provider": prov}, claims), ctrl) == []


# ------------------------------------------------------------------ NET-02-R04

def _structural(planted: bool, contract: bool = False):
    rng = np.random.default_rng(3)
    rows = []
    k = 0
    months = pd.date_range("2024-07-01", "2025-12-01", freq="MS")
    for mi, m in enumerate(months):
        for p in range(12):
            n = 3 + p  # stable ranks: P0 smallest, P11 largest
            if planted and p == 1 and mi >= 9:
                n = 40
            for j in range(n):
                rows.append(_claim(f"C{k}", f"M{p}-{j}-{mi % 2}", f"P{p}", date=str(m.date()), amount=3000.0))
                k += 1
    claims = pd.DataFrame(rows)
    prov = _providers(per={("P1", "ownership_changed_on"): pd.Timestamp("2025-02-15")})
    con = pd.DataFrame([{"contract_sk": "K0", "provider_sk": "P0", "valid_from": pd.Timestamp("2020-01-01"),
                         "tenant_id": "T001"}])
    if contract:
        con = pd.concat([con, pd.DataFrame([{"contract_sk": "K1", "provider_sk": "P1",
                                              "valid_from": pd.Timestamp("2025-03-01"), "tenant_id": "T001"}])])
    return {"provider": prov, "contract": con}, claims


def test_net_02_r04_fires_on_sudden_rise_and_excludes_new_contract(config, registry):
    ctrl = _control(registry, "NET-02-R04")
    tables, claims = _structural(True)
    sigs = N.net_02_r04_structural_change(_ctx(config, tables, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P1"] and sigs[0].evidence["plain_language"]
    assert sigs[0].evidence["ownership_changed_on"] == "2025-02-15"
    tables, claims = _structural(False)
    assert N.net_02_r04_structural_change(_ctx(config, tables, claims), ctrl) == []
    tables, claims = _structural(True, contract=True)
    assert N.net_02_r04_structural_change(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ NET-03-R02

def _closed(planted: bool, onsite: bool = False):
    rows, members = [], []
    k = 0
    for i in range(200):
        emp = f"E{i % 20}"
        members.append({"member_sk": f"M{i}", "employer_id": emp, "emirate": "Dubai", "tenant_id": "T001"})
        for j in range(2):
            rows.append(_claim(f"C{k}", f"M{i}", f"P{(i + j) % 10}"))
            k += 1
    if planted:
        for i in range(12):
            members.append({"member_sk": f"G{i}", "employer_id": "EMPX", "emirate": "Dubai", "tenant_id": "T001"})
            for j in range(3):
                rows.append(_claim(f"C{k}", f"G{i}", "P11" if (j < 2 or onsite) else "P10"))
                k += 1
    prov = _providers()
    if onsite:
        prov.loc[prov["provider_sk"] == "P11", "facility_type"] = "onsite clinic"
    return {"member": pd.DataFrame(members), "provider": prov}, pd.DataFrame(rows)


def test_net_03_r02_fires_on_closed_group_and_excludes_onsite(config, registry):
    ctrl = _control(registry, "NET-03-R02")
    tables, claims = _closed(True)
    sigs = N.net_03_r02_closed_member_group(_ctx(config, tables, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P11"] and sigs[0].evidence["plain_language"]
    tables, claims = _closed(False)
    assert N.net_03_r02_closed_member_group(_ctx(config, tables, claims), ctrl) == []
    tables, claims = _closed(True, onsite=True)
    assert N.net_03_r02_closed_member_group(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ NET-03-R03

def _inducement(planted: bool):
    rng = np.random.default_rng(4)
    rows, lines = [], []
    k = 0
    for p in range(10):
        for i in range(40):
            member = f"M{p}-{i % 20}"
            high = rng.random() < 0.08
            share = 0.0 if rng.random() < 0.05 else 30.0
            if planted and p == 3:
                member = f"M{p}-{i % 8}"
                high = rng.random() < 0.8
                share = 0.0 if rng.random() < 0.8 else 30.0
            amount = 15000.0 if high else 500.0
            rows.append(_claim(f"C{k}", member, f"P{p}", amount=amount, patient_share=share))
            lines.append({"line_sk": f"L{k}", "claim_sk": f"C{k}", "activity_code": "X", "net_amount": amount,
                          "tenant_id": "T001"})
            k += 1
    return {"claim_line": pd.DataFrame(lines)}, pd.DataFrame(rows)


def test_net_03_r03_fires_on_inducement_and_is_silent_otherwise(config, registry):
    ctrl = _control(registry, "NET-03-R03")
    tables, claims = _inducement(True)
    sigs = N.net_03_r03_inducement_signature(_ctx(config, tables, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P3"] and sigs[0].evidence["plain_language"]
    tables, claims = _inducement(False)
    assert N.net_03_r03_inducement_signature(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ NET-03-R04

def _complaints(planted: bool, unreliable: bool = False):
    claims = pd.DataFrame([_claim(f"C{i}", f"M{i}", "P1" if i < 3 else "P2") for i in range(6)])
    conf = [{"confirmation_sk": "F0", "member_sk": "M0", "claim_sk": "C0", "service_confirmed": not planted,
             "response": "I never had this scan", "response_date": pd.Timestamp("2025-02-01"),
             "channel": "unverified" if unreliable else "app", "tenant_id": "T001"},
            {"confirmation_sk": "F1", "member_sk": "M3", "claim_sk": "C3", "service_confirmed": True,
             "response": "yes", "response_date": pd.Timestamp("2025-02-01"), "channel": "app", "tenant_id": "T001"}]
    comp = [{"complaint_sk": "K0", "member_sk": "M1", "provider_sk": "P1", "claim_sk": "C1",
             "complaint_type": "billing" if planted else "waiting time",
             "complaint_date": pd.Timestamp("2025-02-03"), "text": "Charged for items not provided",
             "tenant_id": "T001"}]
    if not planted:
        comp[0]["text"] = "Long wait at reception"
    return {"member_confirmation": pd.DataFrame(conf), "complaint": pd.DataFrame(comp)}, claims


def test_net_03_r04_fires_on_corroboration_and_checks_reliability(config, registry):
    ctrl = _control(registry, "NET-03-R04")
    tables, claims = _complaints(True)
    sigs = N.net_03_r04_complaint_corroboration(_ctx(config, tables, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P1"] and sigs[0].evidence["plain_language"]
    tables, claims = _complaints(False)
    assert N.net_03_r04_complaint_corroboration(_ctx(config, tables, claims), ctrl) == []
    tables, claims = _complaints(True, unreliable=True)
    assert N.net_03_r04_complaint_corroboration(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ NET-04-R04

def _actors(planted: bool):
    prov = _providers()
    if planted:
        prov.loc[prov["provider_sk"] == "P2", "owner_entity_id"] = "AGT6"
    cover = pd.DataFrame([{"coverage_id": f"CV{i}", "member_sk": f"M{i}", "agent_id": f"AGT{i % 10}",
                           "product": "basic", "tenant_id": "T001"} for i in range(30)])
    members = pd.DataFrame([{"member_sk": f"M{i}", "employer_id": f"EMP{i % 5}", "tenant_id": "T001"}
                            for i in range(30)])
    claims = pd.DataFrame([_claim(f"C{i}", f"M{i}", f"P{i % 4}", agent_id=f"AGT{i % 10}") for i in range(30)])
    return {"provider": prov, "coverage_period": cover, "member": members}, claims


def test_net_04_r04_fires_on_cross_actor_identifier_and_is_silent_otherwise(config, registry):
    ctrl = _control(registry, "NET-04-R04")
    tables, claims = _actors(True)
    sigs = N.net_04_r04_shared_identifiers_actors(_ctx(config, tables, claims), ctrl)
    assert [s.subject_id for s in sigs] == ["P2"] and sigs[0].evidence["plain_language"]
    assert sigs[0].claim_ids  # the claims P2 billed for AGT6 patients
    tables, claims = _actors(False)
    assert N.net_04_r04_shared_identifiers_actors(_ctx(config, tables, claims), ctrl) == []


# ------------------------------------------------------------------ generic properties

CASES = {
    "NET-01-R02": (N.net_01_r02_reciprocal_referral_loop, lambda: _loop(True)),
    "NET-01-R03": (N.net_01_r03_high_cost_conversion, lambda: _conversion(True)),
    "NET-01-R04": (N.net_01_r04_closed_downstream_chain, lambda: _chain(True)),
    "NET-02-R01": (N.net_02_r01_shared_admin_identity,
                   lambda: ({"provider": _providers(per={("P3", "phone_token"): "PH4"})},
                            pd.DataFrame([_claim("C0", "M0", "P3")]))),
    "NET-02-R04": (N.net_02_r04_structural_change, lambda: _structural(True)),
    "NET-03-R02": (N.net_03_r02_closed_member_group, lambda: _closed(True)),
    "NET-03-R03": (N.net_03_r03_inducement_signature, lambda: _inducement(True)),
    "NET-03-R04": (N.net_03_r04_complaint_corroboration, lambda: _complaints(True)),
    "NET-04-R04": (N.net_04_r04_shared_identifiers_actors, lambda: _actors(True)),
}


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_empty_tables_return_nothing(config, registry, rule_id):
    fn, _ = CASES[rule_id]
    ctx = _ctx(config, {}, pd.DataFrame(columns=["claim_sk", "member_sk", "provider_sk"]))
    assert fn(ctx, _control(registry, rule_id)) == []
    ctx2 = ControlContext(config=config, tenant_id="T001", claims=pd.DataFrame(), features=None, dataset=None)
    assert fn(ctx2, _control(registry, rule_id)) == []


@pytest.mark.parametrize("rule_id", sorted(CASES))
def test_idempotent(config, registry, rule_id):
    fn, build = CASES[rule_id]
    tables, claims = build()
    ctrl = _control(registry, rule_id)
    a = [s.signal_id for s in fn(_ctx(config, tables, claims), ctrl)]
    b = [s.signal_id for s in fn(_ctx(config, tables, claims), ctrl)]
    assert a and a == b


def test_unlock_yaml_matches_implementations():
    raw = yaml.safe_load((ROOT / "rules" / "unlocks" / "NET.yaml").read_text(encoding="utf-8"))
    declared = {u["implementation"] for u in raw["unlocks"]}
    assert declared == set(N.IMPLEMENTATIONS)
    assert "NET-04-R01" not in {u["rule_id"] for u in raw["unlocks"]}  # needs real reviewer decisions
