"""The CLEAN base population of the SYNTHETIC UAE demo dataset, and the claim factory.

``build(world, n_claims=24000)`` creates providers, clinicians, employers, agents,
members, cover, benefits and ~``n_claims`` internally consistent claims across
every claim table, then registers :func:`make_claim` as ``world.claim_factory``.

The base is meant to be *clean*: as far as the generator can arrange it, none of
the catalogue's patterns occurs except at a low natural rate. Injectors
(``tools/uae_demo/inject_*.py``) plant the patterns afterwards.

What "clean" means here, concretely
-----------------------------------
* every claim falls inside the member's active cover and the provider's
  licence and network periods; no exclusions, no deaths;
* every rendering clinician is on the provider's roster, licensed, privileged
  for the code family and not on leave on the service date, and never booked
  for two overlapping timed services;
* lines are priced exactly at ``tariff.allowed_price × (1 − contract.discount_pct)``;
  patient share is ``gross × benefit_rule_version.patient_share_pct`` (see
  "benefit family" below); ``net = gross − patient_share``;
* services needing authorisation have an APPROVED authorisation, valid on the
  service date, for the same provider, code and enough units/value;
* labs/imaging respect ``repeat_interval_policy``; MRI/arthroplasty/PCI respect
  ``care_pathway_policy``; step-therapy products follow their first-line
  product; refills never start before 80% of the previous supply has elapsed;
  code pairs in ``bundling_edit_table`` are never billed together; line codes
  agree with the principal diagnosis (``indication_policy``,
  ``dx_proc_prohibition_table``); unit maxima are respected;
* inpatient claims bill one case-rate code (severity from ``drg_grouper``),
  package components at zero price, implants and high-cost drugs at price;
  secondary diagnoses come from the member's own chronic conditions plus acute
  complications drawn from the morbidity model, which is then recalibrated to
  the population so observed ≈ expected;
* every lab/imaging line has an observation, every claim with lab or imaging
  lines has a report document, inpatient claims have a SYNTHETIC EN/AR
  discharge summary (``fwa.nlp.synthetic.render_discharge_summary``), and
  services named in ``medical_necessity_policy`` have a clinical note
  containing the required terms.

Benefit family
    The ``service_family`` of the line's code in ``activity_code_reference``
    (oral products ``RX####`` → ``PHARMACY``), except that on an INPATIENT
    claim every line takes the case-rate code's family (INPATIENT or
    DAY_SURGERY). Authorisation is required for a line when the benefit rule
    for (product, benefit family) says so, or the claim is INPATIENT.

Status vocabularies (for injectors)
    ``provider_status_period.status_type``: LICENCE (ACTIVE/SUSPENDED/EXPIRED),
    NETWORK (IN_NETWORK/OUT_OF_NETWORK), EXCLUSION (EXCLUDED — the adapter's
    blacklist flag). ``claim_header.claim_type``: OUTPATIENT, PHARMACY, LAB,
    RADIOLOGY, INPATIENT. ``encounter.encounter_type``: OUTPATIENT, EMERGENCY,
    TELEHEALTH, PHARMACY, DIAGNOSTIC, INPATIENT, DAY_CASE. ``remittance.decision``
    (per line): PAID, PARTIAL, DENIED. ``claim_version.relationship``:
    ORIGINAL, RESUBMISSION, CORRECTION, CANCELLATION.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from . import reference as ref
from .world import World

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

__all__ = ["build", "make_claim", "DEFAULT_CLAIMS"]

D = _dt.date
TD = _dt.timedelta
DT = _dt.datetime
TENANT = "T001"
DEFAULT_CLAIMS = 24000
OPEN_END = D(2099, 12, 31)

# =============================================================================
# fixed vocabularies
# =============================================================================

#: (emirate, short code, population weight, regulator)
EMIRATES = [
    ("Abu Dhabi", "AUH", 0.30, "DOH"), ("Dubai", "DXB", 0.38, "DHA"), ("Sharjah", "SHJ", 0.14, "MOHAP"),
    ("Ajman", "AJM", 0.06, "MOHAP"), ("Ras Al Khaimah", "RAK", 0.05, "MOHAP"),
    ("Fujairah", "FUJ", 0.04, "MOHAP"), ("Umm Al Quwain", "UAQ", 0.03, "MOHAP"),
]
EMIRATE_NAMES = [e[0] for e in EMIRATES]
REGULATOR = {e[0]: e[3] for e in EMIRATES}
NEIGHBOURS = {
    "Abu Dhabi": ["Dubai"], "Dubai": ["Sharjah", "Abu Dhabi"], "Sharjah": ["Dubai", "Ajman"],
    "Ajman": ["Sharjah", "Umm Al Quwain"], "Umm Al Quwain": ["Ajman", "Sharjah"],
    "Ras Al Khaimah": ["Umm Al Quwain", "Sharjah"], "Fujairah": ["Sharjah", "Ras Al Khaimah"],
}

PAYERS = {
    "SYN-PAYER-A": {"name": "Synthetic Payer A", "factor": 1.00},
    "SYN-PAYER-B": {"name": "Synthetic Payer B", "factor": 0.97},
    "SYN-PAYER-C": {"name": "Synthetic Payer C", "factor": 1.03},
}
TIERS = ("TIER_1", "TIER_2", "TIER_3")
TIER_FACTOR = {"TIER_1": 1.15, "TIER_2": 1.00, "TIER_3": 0.88}

#: product → network, accessible provider tiers, tier label, policy-type prefix, payer weights, TPA
PRODUCTS = {
    "GROUP_ESSENTIAL": dict(network="NW-BRONZE", tiers=("TIER_3",), tier="ESSENTIAL",
                            payers={"SYN-PAYER-A": 0.5, "SYN-PAYER-B": 0.3, "SYN-PAYER-C": 0.2}),
    "GROUP_ENHANCED": dict(network="NW-SILVER", tiers=("TIER_2", "TIER_3"), tier="ENHANCED",
                           payers={"SYN-PAYER-A": 0.3, "SYN-PAYER-B": 0.4, "SYN-PAYER-C": 0.3}),
    "FAMILY_PREMIER": dict(network="NW-GOLD", tiers=TIERS, tier="PREMIER",
                           payers={"SYN-PAYER-A": 0.2, "SYN-PAYER-B": 0.3, "SYN-PAYER-C": 0.5}),
    "INDIVIDUAL_PLUS": dict(network="NW-SILVER", tiers=("TIER_2", "TIER_3"), tier="PLUS",
                            payers={"SYN-PAYER-A": 0.4, "SYN-PAYER-B": 0.2, "SYN-PAYER-C": 0.4}),
}
TPA_OF = {  # (payer, product) → TPA
    ("SYN-PAYER-A", "GROUP_ESSENTIAL"): "SYN-TPA-1", ("SYN-PAYER-A", "GROUP_ENHANCED"): "SYN-TPA-1",
    ("SYN-PAYER-A", "FAMILY_PREMIER"): "SYN-TPA-2", ("SYN-PAYER-A", "INDIVIDUAL_PLUS"): "SYN-TPA-3",
    ("SYN-PAYER-B", "GROUP_ESSENTIAL"): "SYN-TPA-2", ("SYN-PAYER-B", "GROUP_ENHANCED"): "SYN-TPA-2",
    ("SYN-PAYER-B", "FAMILY_PREMIER"): "SYN-TPA-2", ("SYN-PAYER-B", "INDIVIDUAL_PLUS"): "SYN-TPA-3",
    ("SYN-PAYER-C", "GROUP_ESSENTIAL"): "SYN-TPA-3", ("SYN-PAYER-C", "GROUP_ENHANCED"): "SYN-TPA-1",
    ("SYN-PAYER-C", "FAMILY_PREMIER"): "SYN-TPA-1", ("SYN-PAYER-C", "INDIVIDUAL_PLUS"): "SYN-TPA-3",
}

FAMILIES = ["CONSULTATION", "TELEHEALTH", "EMERGENCY", "INPATIENT", "DAY_SURGERY", "PROCEDURE", "LAB",
            "IMAGING", "ADVANCED_IMAGING", "PHARMACY", "SPECIALTY_DRUG", "DEVICE", "PHYSIOTHERAPY",
            "MENTAL_HEALTH", "MATERNITY", "COSMETIC"]
AUTH_FAMILIES = {"INPATIENT", "DAY_SURGERY", "ADVANCED_IMAGING", "SPECIALTY_DRUG"}
#: patient-share % per tier label and family (2024); 2025 changes in _SHARE_2025.
_SHARE = {
    "ESSENTIAL": dict(CONSULTATION=.20, TELEHEALTH=.10, EMERGENCY=.10, INPATIENT=.05, DAY_SURGERY=.05,
                      PROCEDURE=.20, LAB=.10, IMAGING=.10, ADVANCED_IMAGING=.10, PHARMACY=.30,
                      SPECIALTY_DRUG=.10, DEVICE=.10, PHYSIOTHERAPY=.20, MENTAL_HEALTH=.20, MATERNITY=.10),
    "ENHANCED": dict(CONSULTATION=.10, TELEHEALTH=.05, EMERGENCY=.05, INPATIENT=.0, DAY_SURGERY=.0,
                     PROCEDURE=.10, LAB=.05, IMAGING=.05, ADVANCED_IMAGING=.10, PHARMACY=.20,
                     SPECIALTY_DRUG=.10, DEVICE=.10, PHYSIOTHERAPY=.10, MENTAL_HEALTH=.10, MATERNITY=.05),
    "PREMIER": dict(CONSULTATION=.05, TELEHEALTH=.0, EMERGENCY=.0, INPATIENT=.0, DAY_SURGERY=.0,
                    PROCEDURE=.0, LAB=.0, IMAGING=.0, ADVANCED_IMAGING=.05, PHARMACY=.10,
                    SPECIALTY_DRUG=.05, DEVICE=.0, PHYSIOTHERAPY=.05, MENTAL_HEALTH=.05, MATERNITY=.0),
    "PLUS": dict(CONSULTATION=.15, TELEHEALTH=.10, EMERGENCY=.10, INPATIENT=.10, DAY_SURGERY=.10,
                 PROCEDURE=.15, LAB=.10, IMAGING=.10, ADVANCED_IMAGING=.15, PHARMACY=.25,
                 SPECIALTY_DRUG=.15, DEVICE=.15, PHYSIOTHERAPY=.15, MENTAL_HEALTH=.15, MATERNITY=.10),
}
_SHARE_2025 = {("ESSENTIAL", "PHARMACY"): .25, ("ENHANCED", "CONSULTATION"): .15,
               ("PLUS", "LAB"): .15, ("PREMIER", "PHARMACY"): .15}
_LIMIT = dict(CONSULTATION=15000, TELEHEALTH=3000, EMERGENCY=30000, INPATIENT=500000, DAY_SURGERY=150000,
              PROCEDURE=30000, LAB=15000, IMAGING=15000, ADVANCED_IMAGING=40000, PHARMACY=30000,
              SPECIALTY_DRUG=400000, DEVICE=60000, PHYSIOTHERAPY=12000, MENTAL_HEALTH=15000,
              MATERNITY=60000, COSMETIC=0)

#: specialty → privileged code families
_EM_OFFICE = ["EM_OFFICE_NEW", "EM_OFFICE_EST"]
_EM_IP = ["EM_INPATIENT_INIT", "EM_INPATIENT_SUBSEQ", "EM_DISCHARGE", "CASE_RATE", "ROOM_DAY"]
PRIVILEGES: dict[str, list[str]] = {
    "GENERAL_PRACTICE": _EM_OFFICE + ["TELEHEALTH", "PROC_MINOR", "DIAG_OFFICE", "LAB_POC", "INJECTION",
                                      "VACCINE", "DRUG_INJECTABLE", "SPECIMEN", "DME"],
    "FAMILY_MEDICINE": _EM_OFFICE + ["TELEHEALTH", "PROC_MINOR", "DIAG_OFFICE", "LAB_POC", "INJECTION",
                                     "VACCINE", "DRUG_INJECTABLE", "SPECIMEN", "DME"],
    "PAEDIATRICS": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "TELEHEALTH", "PROC_MINOR", "DIAG_OFFICE", "LAB_POC",
                                          "INJECTION", "VACCINE", "DRUG_INJECTABLE", "SPECIMEN"],
    "INTERNAL_MEDICINE": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "TELEHEALTH", "DIAG_OFFICE", "LAB_POC", "INJECTION",
                                                "INFUSION", "DRUG_INJECTABLE", "SPECIMEN", "CRITICAL_CARE", "DME"],
    "EMERGENCY_MEDICINE": ["EM_ED", "PROC_MINOR", "DIAG_OFFICE", "LAB_POC", "INJECTION", "INFUSION",
                           "DRUG_INJECTABLE", "SPECIMEN", "CRITICAL_CARE", "DME"],
    "GENERAL_SURGERY": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "SURG_ABDOMINAL", "SURG_BREAST", "PROC_MINOR",
                                              "IMPLANT", "ENDOSCOPY", "DRUG_INJECTABLE"],
    "ORTHOPAEDICS": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "SURG_ORTHO", "PROC_ORTHO", "IMPLANT", "DME",
                                           "INJECTION", "DRUG_INJECTABLE"],
    "OBSTETRICS_GYNAECOLOGY": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "OBSTETRIC", "IMAGING_US", "DRUG_INJECTABLE"],
    "CARDIOLOGY": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "DIAG_OFFICE", "ECHO", "CARDIAC_INVASIVE",
                                         "CRITICAL_CARE", "IMPLANT", "DRUG_INJECTABLE"],
    "OPHTHALMOLOGY": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "SURG_EYE", "IMPLANT"],
    "ANAESTHESIA": ["ANAESTHESIA", "SEDATION", "CRITICAL_CARE"],
    "RADIOLOGY": ["IMAGING_XR", "IMAGING_US", "IMAGING_CT", "IMAGING_MRI", "IMAGING_NM", "IMAGING_MAMMO",
                  "IMAGING_DEXA"],
    "PATHOLOGY": ["LAB_PANEL", "LAB_HAEM", "LAB_URINE", "LAB_POC", "LAB_ENDO", "LAB_CHEM", "LAB_MICRO",
                  "LAB_PATH", "SPECIMEN"],
    "PHARMACY": ["DRUG_ORAL", "DRUG_INJECTABLE", "SUPPLY", "DME"],
    "PHYSIOTHERAPY": ["PHYSIO"],
    "PSYCHIATRY": _EM_OFFICE + ["EM_CONSULT", "PSYCHOTHERAPY", "TELEHEALTH"],
    "CLINICAL_PSYCHOLOGY": ["PSYCHOTHERAPY"],
    "DERMATOLOGY": _EM_OFFICE + ["EM_CONSULT", "PROC_MINOR", "TELEHEALTH"],
    "ENT": _EM_OFFICE + ["EM_CONSULT", "PROC_MINOR", "SURG_ENT"],
    "ENDOCRINOLOGY": _EM_OFFICE + ["EM_CONSULT", "TELEHEALTH", "DIAG_OFFICE", "LAB_POC", "INJECTION",
                                   "DRUG_INJECTABLE"],
    "RHEUMATOLOGY": _EM_OFFICE + ["EM_CONSULT", "PROC_ORTHO", "INJECTION", "INFUSION", "DRUG_INJECTABLE"],
    "NEUROLOGY": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "NEURODIAG"],
    "GASTROENTEROLOGY": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "ENDOSCOPY", "SEDATION", "INFUSION",
                                               "DRUG_INJECTABLE"],
    "ONCOLOGY": _EM_OFFICE + ["EM_CONSULT", "CHEMO", "INFUSION", "DRUG_INJECTABLE", "INJECTION"],
    "PULMONOLOGY": _EM_OFFICE + _EM_IP + ["EM_CONSULT", "DIAG_OFFICE"],
}
ROLE_OF = {"PATHOLOGY": "PATHOLOGIST", "RADIOLOGY": "RADIOLOGIST", "PHARMACY": "PHARMACIST",
           "PHYSIOTHERAPY": "PHYSIOTHERAPIST", "CLINICAL_PSYCHOLOGY": "PSYCHOLOGIST",
           "ANAESTHESIA": "CONSULTANT", "GENERAL_PRACTICE": "GP", "FAMILY_MEDICINE": "GP",
           "EMERGENCY_MEDICINE": "SPECIALIST"}
GP_SPECS = ("GENERAL_PRACTICE", "FAMILY_MEDICINE")

#: clinic kinds: (kind, share, facility_type, {specialty: (min, max)})
CLINIC_KINDS = [
    ("GP", 0.36, "MEDICAL_CENTRE", {"GENERAL_PRACTICE": (1, 3), "FAMILY_MEDICINE": (1, 2)}),
    ("POLY", 0.20, "POLYCLINIC", {"GENERAL_PRACTICE": (1, 2), "FAMILY_MEDICINE": (0, 1)}),
    ("PAEDIATRICS", 0.06, "SPECIALIST_CLINIC", {"PAEDIATRICS": (2, 4)}),
    ("OBSTETRICS_GYNAECOLOGY", 0.05, "SPECIALIST_CLINIC", {"OBSTETRICS_GYNAECOLOGY": (2, 4)}),
    ("PHYSIOTHERAPY", 0.06, "PHYSIOTHERAPY_CENTRE", {"PHYSIOTHERAPY": (3, 5)}),
    ("DERMATOLOGY", 0.04, "SPECIALIST_CLINIC", {"DERMATOLOGY": (2, 4)}),
    ("ORTHOPAEDICS", 0.04, "SPECIALIST_CLINIC", {"ORTHOPAEDICS": (2, 3), "PHYSIOTHERAPY": (1, 2)}),
    ("PSYCHIATRY", 0.035, "SPECIALIST_CLINIC", {"PSYCHIATRY": (1, 2), "CLINICAL_PSYCHOLOGY": (1, 3)}),
    ("ENDOCRINOLOGY", 0.03, "SPECIALIST_CLINIC", {"ENDOCRINOLOGY": (2, 3), "GENERAL_PRACTICE": (0, 1)}),
    ("CARDIOLOGY", 0.025, "SPECIALIST_CLINIC", {"CARDIOLOGY": (2, 4)}),
    ("ENT", 0.02, "SPECIALIST_CLINIC", {"ENT": (2, 3)}),
    ("OPHTHALMOLOGY", 0.02, "SPECIALIST_CLINIC", {"OPHTHALMOLOGY": (2, 3)}),
    ("RHEUMATOLOGY", 0.015, "SPECIALIST_CLINIC", {"RHEUMATOLOGY": (2, 3)}),
    ("NEUROLOGY", 0.015, "SPECIALIST_CLINIC", {"NEUROLOGY": (2, 3)}),
    ("GASTROENTEROLOGY", 0.015, "SPECIALIST_CLINIC", {"GASTROENTEROLOGY": (2, 3)}),
]
POLY_SPECIALISTS = ["PAEDIATRICS", "DERMATOLOGY", "OBSTETRICS_GYNAECOLOGY", "INTERNAL_MEDICINE", "ENT",
                    "ORTHOPAEDICS", "ENDOCRINOLOGY", "CARDIOLOGY", "PSYCHIATRY"]
HOSPITAL_REQUIRED = {"EMERGENCY_MEDICINE": (2, 4), "INTERNAL_MEDICINE": (2, 5), "GENERAL_SURGERY": (1, 3),
                     "ANAESTHESIA": (2, 4), "RADIOLOGY": (1, 3), "PATHOLOGY": (1, 2), "PHARMACY": (1, 3),
                     "OBSTETRICS_GYNAECOLOGY": (1, 3), "ORTHOPAEDICS": (1, 3), "PAEDIATRICS": (1, 2),
                     "PHYSIOTHERAPY": (1, 2)}
HOSPITAL_OPTIONAL = ["CARDIOLOGY", "OPHTHALMOLOGY", "GASTROENTEROLOGY", "NEUROLOGY", "ONCOLOGY",
                     "PULMONOLOGY", "ENDOCRINOLOGY", "RHEUMATOLOGY", "ENT", "DERMATOLOGY", "PSYCHIATRY",
                     "CARDIOLOGY", "INTERNAL_MEDICINE", "GENERAL_SURGERY"]

EQUIPMENT = {
    "XRAY": (["71045", "71046", "73030", "73562", "73610", "72100", "73502"], 10, 16),
    "ULTRASOUND": (["76700", "76705", "76856", "76801", "76805", "76536"], 20, 12),
    "CT": (["70450", "71250", "71260", "74176", "74177"], 20, 16),
    "MRI": (["70551", "70552", "70553", "72148", "73721", "73221"], 45, 14),
    "PET_CT": (["78815"], 90, 10),
    "MAMMOGRAPHY": (["77067"], 20, 10),
    "DEXA": (["77080"], 15, 10),
    "CATH_LAB": (["93458", "92920"], 90, 12),
}
EQUIP_OF_CODE = {c: e for e, (codes, _, _) in EQUIPMENT.items() for c in codes}

DENIAL_CODES = ["MNEC-003", "MNEC-004", "DOC-001", "DOC-003", "CLAI-012"]
RESUBMITTABLE = {"DOC-001", "DOC-003", "CLAI-012"}

# =============================================================================
# clinical scenarios
# =============================================================================

#: Acute outpatient scenarios. rx: list of (probability, [products]); labs/imaging: (probability, [codes]).
ACUTE: list[dict[str, Any]] = [
    dict(key="URTI", dx="J06.9", w=12, levels={2: .35, 3: .5, 4: .12, 1: .03}, rx=[(.5, ["RX1033"]), (.35, ["RX1041", "RX1042"])], tele=.08, season=1.8),
    dict(key="PHARYNGITIS", dx="J02.9", w=5, levels={2: .3, 3: .55, 4: .15}, office=[(.5, "87880")], rx=[(.7, ["RX1001", "RX1002"]), (.4, ["RX1035"])], season=1.5),
    dict(key="TONSILLITIS", dx="J03.90", w=3, levels={2: .2, 3: .6, 4: .2}, rx=[(.8, ["RX1004", "RX1005", "RX1001"]), (.5, ["RX1035"])], season=1.4),
    dict(key="BRONCHITIS", dx="J20.9", w=3, levels={3: .6, 4: .4}, imaging=[(.6, ["71046"])], rx=[(.3, ["RX1038"]), (.3, ["RX1006"])], season=1.6),
    dict(key="COUGH", dx="R05.9", w=2, levels={2: .4, 3: .6}, imaging=[(.5, ["71046"])], rx=[(.2, ["RX1041"])], season=1.5),
    dict(key="OTITIS", dx="H66.90", w=3, age=(0, 12), levels={2: .3, 3: .6, 4: .1}, rx=[(.8, ["RX1001"]), (.5, ["RX1033"])], season=1.3),
    dict(key="CONJUNCTIVITIS", dx="H10.9", w=2, levels={2: .6, 3: .4}),
    dict(key="RHINITIS", dx="J30.9", w=3, levels={2: .4, 3: .5, 4: .1}, rx=[(.7, ["RX1041", "RX1042"]), (.2, ["RX1040"])], tele=.08),
    dict(key="GASTRO", dx="A09", w=4, levels={2: .2, 3: .6, 4: .2}, labs=[(.1, ["85025"])], rx=[(.6, ["RX1059"]), (.3, ["RX1057"])], ed=.12),
    dict(key="GASTRITIS", dx="K29.70", w=2, age=(16, 90), levels={3: .6, 4: .4}, rx=[(.9, ["RX1029", "RX1030", "RX1032"])], refer=("GASTROENTEROLOGY", .05)),
    dict(key="CONSTIPATION", dx="K59.00", w=1, levels={2: .5, 3: .5}),
    dict(key="UTI", dx="N39.0", w=3, sexw={"F": 4, "M": 1}, age=(12, 90), levels={3: .7, 4: .3}, office=[(.6, "81002")], labs=[(.35, ["81001", "87086"])], rx=[(.9, ["RX1008", "RX1009"])]),
    dict(key="BACKPAIN", dx="M54.50", w=5, age=(18, 80), levels={3: .6, 4: .4}, imaging=[(.6, ["72100"])], rx=[(.8, ["RX1035", "RX1037"])], physio=.15),
    dict(key="KNEEPAIN", dx="M25.561", w=2, age=(20, 80), levels={3: .6, 4: .4}, imaging=[(.8, ["73562"])], rx=[(.7, ["RX1035", "RX1037"])], physio=.12, refer=("ORTHOPAEDICS", .12)),
    dict(key="SPRAIN", dx="S93.401A", w=2.5, age=(8, 70), levels={3: .6, 4: .4}, imaging=[(.8, ["73610"])], rx=[(.7, ["RX1035"])], dme=[(.15, "E0114")], ed=.3, injury=True),
    dict(key="LACERATION", dx="S61.411A", w=1, age=(3, 80), levels={3: .6, 4: .4}, office=[(1.0, "12001")], rx=[(.5, ["RX1056"])], ed=.7, injury=True),
    dict(key="ABSCESS", dx="L02.91", w=1, age=(5, 85), levels={3: .6, 4: .4}, office=[(1.0, "10060")], rx=[(.8, ["RX1004", "RX1005"])]),
    dict(key="DERMATITIS", dx="L30.9", w=2, levels={2: .4, 3: .6}, rx=[(.8, ["RX1055"]), (.3, ["RX1041"])], spec="DERMATOLOGY", specp=.3, tele=.05),
    dict(key="ACNE", dx="L70.0", w=1, age=(12, 30), levels={3: .7, 4: .3}, rx=[(.25, ["RX1062"])], spec="DERMATOLOGY", specp=.8),
    dict(key="WART", dx="B07.9", w=1, age=(5, 60), levels={2: .5, 3: .5}, office=[(1.0, "17110")], spec="DERMATOLOGY", specp=.5),
    dict(key="SKINTAG", dx="L91.8", w=.3, age=(25, 80), levels={2: .5, 3: .5}, office=[(1.0, "11200")], spec="DERMATOLOGY", specp=.8),
    dict(key="CERUMEN", dx="H61.23", w=1, levels={2: .6, 3: .4}, office=[(1.0, "69210")], spec="ENT", specp=.2),
    dict(key="HEADACHE", dx="R51.9", w=2, age=(10, 90), levels={3: .6, 4: .4}, rx=[(.8, ["RX1033", "RX1035"])], imaging=[(.05, ["70450"])]),
    dict(key="MIGRAINE", dx="G43.909", w=1, age=(12, 70), levels={3: .5, 4: .5}, rx=[(.8, ["RX1035"])], imaging=[(.04, ["70551"])], refer=("NEUROLOGY", .08)),
    dict(key="FEVER", dx="R50.9", w=2, levels={2: .3, 3: .5, 4: .2}, labs=[(.3, ["85025", "86140"])], rx=[(.8, ["RX1033"])], season=1.4),
    dict(key="COVID", dx="U07.1", w=1, levels={3: .7, 4: .3}, labs=[(.8, ["87635"])], rx=[(.7, ["RX1033"])], season=1.5),
    dict(key="CHESTPAIN", dx="R07.9", w=1, age=(30, 90), levels={3: .3, 4: .5, 5: .2}, office=[(.9, "93000")], labs=[(.3, ["84484"])], imaging=[(.5, ["71046"])], ed=.4, refer=("CARDIOLOGY", .2)),
    dict(key="VITD", dx="E55.9", w=2, age=(12, 90), levels={2: .4, 3: .6}, labs=[(.9, ["82306"])], rx=[(.9, ["RX1044"])]),
    dict(key="IRONDEF", dx="D50.9", w=1, sexw={"F": 5, "M": 1}, age=(12, 90), levels={3: .7, 4: .3}, labs=[(.9, ["85025", "83540", "82728"])], rx=[(.9, ["RX1045"])]),
    dict(key="CHECKUP", dx="Z00.00", w=3, age=(18, 90), levels={3: .6, 4: .4}, labs=[(.8, ["85025", "80061"])], imaging=[(.6, ["71046"])]),
    dict(key="VACCINE", dx="Z23", w=1, levels={}, vaccine=True, season=2.5),
    dict(key="ANXIETY", dx="F41.1", w=1, age=(16, 80), levels={3: .5, 4: .5}, rx=[(.6, ["RX1052", "RX1053"])], spec="PSYCHIATRY", specp=.7, psych=True),
    dict(key="INSOMNIA", dx="G47.00", w=.4, age=(20, 90), levels={3: .7, 4: .3}, rx=[(.5, ["RX1050"])]),
    dict(key="EPILEPSY", dx="G40.909", w=.3, age=(5, 80), levels={4: .7, 5: .3}, spec="NEUROLOGY", specp=.9, office=[(.3, "95816")], imaging=[(.1, ["70551"])]),
    dict(key="RADICULOPATHY", dx="M54.16", w=1, age=(25, 80), levels={3: .5, 4: .5}, rx=[(.3, ["RX1048"]), (.6, ["RX1035"])], physio=.3, refer=("ORTHOPAEDICS", .15), chain="LUMBAR"),
    dict(key="ROTATOR", dx="M75.101", w=.6, age=(30, 80), levels={3: .5, 4: .5}, rx=[(.6, ["RX1037"])], physio=.5, refer=("ORTHOPAEDICS", .3), chain="SHOULDER"),
    dict(key="MENORRHAGIA", dx="N92.0", w=1, sexw={"F": 1, "M": 0}, age=(15, 50), levels={3: .6, 4: .4}, imaging=[(.6, ["76856"])], labs=[(.5, ["85025"])], rx=[(.5, ["RX1035"])], spec="OBSTETRICS_GYNAECOLOGY", specp=.6),
    dict(key="ABDPAIN", dx="R10.9", w=4, age=(8, 90), levels={3: .5, 4: .5}, imaging=[(.8, ["76700"])], labs=[(.3, ["85025", "80053"])], rx=[(.4, ["RX1058"])]),
    dict(key="KIDNEYSTONE", dx="N20.0", w=.8, age=(20, 80), levels={3: .4, 4: .6}, imaging=[(.5, ["74176"]), (.4, ["76705"])], labs=[(.5, ["81001"])], rx=[(.8, ["RX1035"])], ed=.35),
    dict(key="THYROIDNODULE", dx="E04.1", w=.5, age=(20, 80), sexw={"F": 3, "M": 1}, levels={3: .6, 4: .4}, imaging=[(.9, ["76536"])], labs=[(.8, ["84443"])], spec="ENDOCRINOLOGY", specp=.4),
    dict(key="PNEUMONIA_OP", dx="J18.9", w=.8, age=(5, 80), levels={3: .3, 4: .6, 5: .1}, imaging=[(1.0, ["71046"])], labs=[(.3, ["85025", "86140"])], rx=[(.9, ["RX1004", "RX1005", "RX1006"])], season=1.8),
    dict(key="SCREENMAMMO", dx="Z12.31", w=6, sexw={"F": 1, "M": 0}, age=(40, 75), levels={2: .5, 3: .5}, imaging=[(1.0, ["77067"])]),
    dict(key="SEPTUM", dx="J34.2", w=.2, age=(16, 70), levels={3: .6, 4: .4}, spec="ENT", specp=.9),
]
ACUTE_BY_KEY = {s["key"]: s for s in ACUTE}

#: Chronic conditions: prevalence by age band, follow-up interval (days), clinic specialties,
#: labs/office at visit, drug slots [(probability, [(product, weight)])].
CHRONIC: dict[str, dict[str, Any]] = {
    "I10": dict(prev=[(30, .01), (45, .08), (60, .25), (200, .45)], interval=120, spec=GP_SPECS + ("INTERNAL_MEDICINE",),
                labs=[(.35, ["80048"])], office=[(.15, "93000")],
                drugs=[(1.0, [("RX1020", 5), ("RX1021", 2), ("RX1022", 3)]), (.3, [("RX1024", 1)])]),
    "E11.9": dict(prev=[(30, .005), (45, .06), (60, .18), (200, .25)], interval=90, spec=("ENDOCRINOLOGY",) + GP_SPECS,
                  imaging_at_radiology=[(.15, ["76705"])],
                  labs=[(.95, ["83036"]), (.3, ["80053"]), (.4, ["80061"]), (.3, ["81001"])],
                  drugs=[(1.0, [("RX1010", 4), ("RX1011", 4), ("RX1012", 2)]), (.3, [("RX1013", 1)]),
                         (.2, [("RX1014", 1)]), (.15, [("RX1015", 1)]), (.3, [("A4253", 1)])]),
    "E78.5": dict(prev=[(30, .03), (45, .15), (60, .3), (200, .4)], interval=180, spec=GP_SPECS + ("INTERNAL_MEDICINE",),
                  labs=[(.9, ["80061"])], drugs=[(1.0, [("RX1017", 5), ("RX1018", 2), ("RX1019", 2)])]),
    "E03.9": dict(prev=[(200, .035)], sexw={"F": 1.6, "M": .4}, min_age=12, interval=180, spec=("ENDOCRINOLOGY",) + GP_SPECS,
                  labs=[(.9, ["84443"])], imaging_at_radiology=[(.25, ["76536"])], drugs=[(1.0, [("RX1043", 1)])]),
    "J45.909": dict(prev=[(18, .08), (200, .05)], min_age=4, interval=180, spec=GP_SPECS + ("PULMONOLOGY", "PAEDIATRICS"),
                    office=[(.15, "94010")],
                    drugs=[(1.0, [("RX1038", 1)]), (.4, [("RX1039", 1)]), (.3, [("RX1040", 1)])]),
    "K21.9": dict(prev=[(18, .005), (200, .05)], interval=180, spec=GP_SPECS, imaging_at_radiology=[(.2, ["76705"])],
                  drugs=[(1.0, [("RX1029", 4), ("RX1030", 3), ("RX1031", 1), ("RX1032", 2)])]),
    "F32.9": dict(prev=[(16, 0), (200, .03)], interval=60, spec=("PSYCHIATRY",), psych=True,
                  drugs=[(1.0, [("RX1052", 1), ("RX1053", 1)])]),
    "M05.79": dict(prev=[(30, 0), (200, .006)], sexw={"F": 1.5, "M": .5}, interval=90, spec=("RHEUMATOLOGY",),
                   labs=[(.8, ["86140", "85652"])], drugs=[(1.0, [("RX1060", 1)]), (.4, [("J0135", 1)])]),
    "K50.90": dict(prev=[(16, 0), (200, .003)], interval=56, spec=("GASTROENTEROLOGY",), infusion="J1745"),
    "C50.911": dict(prev=[(35, 0), (200, .004)], sexw={"F": 1.0, "M": 0.0}, interval=21, spec=("ONCOLOGY",), chemo=True),
    "N40.0": dict(prev=[(50, 0), (65, .08), (200, .18)], sexw={"F": 0.0, "M": 1.0}, interval=180, spec=GP_SPECS,
                  labs=[(.5, ["84153"])], imaging_at_radiology=[(.35, ["76705"])], drugs=[(1.0, [("RX1061", 1)])]),
    "M81.0": dict(prev=[(55, 0), (200, .10)], sexw={"F": 1.0, "M": 0.0}, interval=180, spec=GP_SPECS + ("RHEUMATOLOGY",),
                  imaging_at_radiology=[(.6, ["77080"])], drugs=[(.8, [("RX1044", 1)])], denosumab=.3),
    "N18.30": dict(prev=[(55, 0), (200, .05)], interval=120, spec=("INTERNAL_MEDICINE",) + GP_SPECS,
                   labs=[(.8, ["80048"]), (.5, ["81001"])], imaging_at_radiology=[(.4, ["76705"])],
                   drugs=[(1.0, [("RX1022", 1)])]),
    "I25.10": dict(prev=[(45, 0), (200, .06)], interval=120, spec=("CARDIOLOGY",), office=[(.4, "93000")],
                   echo=.15, drugs=[(1.0, [("RX1025", 1)]), (1.0, [("RX1017", 3), ("RX1018", 1)]),
                                    (.3, [("RX1026", 3), ("RX1027", 1)]), (.5, [("RX1024", 1)]), (.1, [("RX1063", 1)])]),
    "J44.9": dict(prev=[(55, 0), (200, .03)], interval=120, spec=GP_SPECS + ("PULMONOLOGY",), office=[(.2, "94010")],
                  imaging_at_radiology=[(.35, ["71046"])],
                  drugs=[(1.0, [("RX1039", 1)]), (.8, [("RX1038", 1)])]),
    "F90.0": dict(prev=[(6, 0), (17, .03), (200, 0)], interval=90, spec=("PSYCHIATRY", "PAEDIATRICS"),
                  drugs=[(1.0, [("RX1051", 1)])]),
    "E66.9": dict(prev=[(18, .05), (200, .2)], interval=0, spec=()),  # comorbidity only, never visited for
}
CHRONIC_SECONDARY_OK = {"I10", "E11.9", "E78.5", "E03.9", "J45.909", "K21.9", "N18.30", "I25.10",
                        "J44.9", "E66.9", "F32.9", "N40.0", "M81.0"}

#: Emergency inpatient case rates: rate per member-year by age band (before the global multiplier).
EMERGENCY_IP = {
    "CR101": [(5, 0), (40, .004), (200, .002)], "CR108": [(65, 0), (200, .006)],
    "CR109": [(40, 0), (60, .003), (200, .008)], "CR110": [(5, .004), (60, .003), (200, .012)],
    "CR111": [(50, 0), (200, .006)], "CR112": [(50, 0), (200, .005)], "CR113": [(30, 0), (200, .003)],
    "CR114": [(15, 0), (200, .003)], "CR115": [(10, .006), (60, .002), (200, .005)],
    "CR116": [(5, 0), (18, .004), (200, .002)], "CR117": [(15, 0), (200, .003)],
    "CR118": [(30, 0), (200, .006)],
}
ELECTIVE_IP = {"CR102": [(25, 0), (200, .004)], "CR103": [(20, 0), (200, .003)],
               "CR106": [(55, 0), (200, .012)], "CR107": [(55, 0), (200, .005)],
               "CR119": [(18, 0), (60, .003), (200, 0)]}

#: lab reference values: code → (mean, sd, unit)
LAB_VALUES = {
    "82947": (5.4, .8, "mmol/L"), "82565": (80, 15, "umol/L"), "84132": (4.2, .3, "mmol/L"),
    "84295": (139, 2.5, "mmol/L"), "82435": (102, 2.5, "mmol/L"), "82374": (25, 2, "mmol/L"),
    "84520": (5.0, 1.2, "mmol/L"), "82310": (2.35, .1, "mmol/L"), "82040": (42, 3, "g/L"),
    "84155": (70, 4, "g/L"), "82247": (10, 4, "umol/L"), "84075": (80, 20, "U/L"), "84460": (28, 10, "U/L"),
    "84450": (26, 8, "U/L"), "82465": (5.0, .9, "mmol/L"), "84478": (1.5, .6, "mmol/L"),
    "83718": (1.3, .3, "mmol/L"), "85048": (7.0, 1.8, "10^9/L"), "85018": (135, 14, "g/L"),
    "85027": (135, 14, "g/L"), "85652": (15, 8, "mm/h"), "85610": (12, 1, "s"), "81001": (0, 0, "negative"),
    "81002": (0, 0, "negative"), "84443": (2.2, 1.0, "mIU/L"), "84439": (15, 3, "pmol/L"),
    "83036": (6.8, 1.1, "%"), "86140": (8, 10, "mg/L"), "82306": (55, 20, "nmol/L"),
    "82607": (350, 100, "pmol/L"), "83540": (14, 6, "umol/L"), "82728": (60, 40, "ug/L"),
    "84153": (1.5, 1.0, "ng/mL"), "84484": (8, 5, "ng/L"), "83880": (150, 120, "pg/mL"),
    "84702": (20000, 15000, "IU/L"), "87086": (0, 0, "no growth"), "87040": (0, 0, "no growth"),
    "87880": (0, 0, "negative"), "87635": (0, 0, "detected"), "88305": (0, 0, "see report"),
}


# =============================================================================
# small helpers
# =============================================================================


def _band(bands: list[tuple[int, float]], age: float) -> float:
    for upper, value in bands:
        if age < upper:
            return value
    return bands[-1][1]


def _age(dob: D, on: D) -> float:
    return (on - dob).days / 365.25


def _token(rng: np.random.Generator, prefix: str, n: int = 16) -> str:
    return prefix + "".join(rng.choice(list("0123456789abcdef"), size=n))


def _year_key(d: D) -> int:
    return 2024 if d < D(2025, 1, 1) else 2025


def _weekday_shift(d: D) -> D:
    while d.weekday() >= 5:  # Saturday/Sunday
        d += TD(days=1)
    return d


def _dt_at(d: D, minutes: int) -> DT:
    return DT(d.year, d.month, d.day) + TD(minutes=int(minutes))


def _pick(rng: np.random.Generator, options: list, weights: Iterable[float] | None = None):
    if not options:
        return None
    if weights is None:
        return options[int(rng.integers(len(options)))]
    w = np.asarray(list(weights), dtype=float)
    if w.sum() <= 0:
        return options[int(rng.integers(len(options)))]
    return options[int(rng.choice(len(options), p=w / w.sum()))]


def _level_from(rng, levels: dict[int, float]) -> int:
    keys = sorted(levels)
    return int(_pick(rng, keys, [levels[k] for k in keys]))


# =============================================================================
# build context
# =============================================================================


class _Ctx:
    """Everything the claim composer needs, built once by :func:`build`."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.rng = world.rng
        self.codes: dict[str, dict] = world.context["codes"]
        self.drugs: dict[str, dict] = world.context["drugs"]
        self.icd: dict[str, tuple[str, str]] = world.context["icd"]
        self.case_rates = world.context["case_rates"]
        self.providers: dict[str, dict] = {}
        self.clinicians: dict[str, list[dict]] = defaultdict(list)
        self.clin_by_id: dict[str, dict] = {}
        self.members: dict[str, dict] = {}
        self.contracts: dict[tuple[str, str], dict] = {}
        self.benefit: dict[tuple[str, str, int], dict] = {}
        self.bookings: dict[tuple[str, D], list[tuple[int, int]]] = defaultdict(list)
        self.equip_use: dict[tuple[str, str, D], int] = defaultdict(int)
        self.beds: dict[str, dict[str, D]] = defaultdict(dict)
        self.seen_mp: set[tuple[str, str]] = set()
        self.claim_ids: set[str] = set()
        self.auth_ids: set[str] = set()
        self.pay_refs: set[str] = set()
        self.serials: set[str] = set()
        self.rows: dict[str, list[dict]] | None = None
        self.adjudicators = [f"ADJ{i:02d}" for i in range(1, 21)]

    # ---------------------------------------------------------------- ids

    def unique(self, pool: set[str], prefix: str, digits: int) -> str:
        while True:
            value = f"{prefix}{int(self.rng.integers(10 ** (digits - 1), 10 ** digits))}"
            if value not in pool:
                pool.add(value)
                return value

    # ---------------------------------------------------------------- codes

    def family_of(self, code: str) -> str:
        if code.startswith("RX"):
            return "DRUG_ORAL"
        c = self.codes.get(code)
        return c["code_family"] if c else "UNKNOWN"

    def service_family(self, code: str) -> str:
        if code.startswith("RX"):
            return "PHARMACY"
        c = self.codes.get(code)
        return c["service_family"] if c else "PROCEDURE"

    def activity_type(self, code: str) -> str:
        if code.startswith("RX"):
            return "DRUG"
        c = self.codes.get(code)
        return c["activity_type"] if c else "CPT"

    def description(self, code: str) -> str:
        if code in self.drugs:
            return self.drugs[code]["description"]
        c = self.codes.get(code)
        return c["description"] if c else code

    def is_drug(self, code: str) -> bool:
        return code in self.drugs

    # ---------------------------------------------------------------- prices

    def allowed_price(self, code: str, tier: str, payer: str, on: D) -> float:
        if code in self.drugs:
            return round(self.drugs[code]["unit_price"], 2)
        base = self.codes[code]["unit_price_reference"]
        year = 1.0 if on < D(2025, 1, 1) else 1.03
        return round(base * TIER_FACTOR[tier] * PAYERS[payer]["factor"] * year, 2)

    # ---------------------------------------------------------------- clinicians

    def available(self, clin: dict, on: D) -> bool:
        if not (clin["lic_from"] <= on <= clin["lic_to"]):
            return False
        for a, b in clin["leave"]:
            if a <= on <= b:
                return False
        return True

    def pick_clinician(self, provider: str, family: str, on: D, specialty: str | None = None,
                       prefer: str | None = None) -> str | None:
        if prefer:
            c = self.clin_by_id.get(prefer)
            if c and c["provider"] == provider and family in c["privileges"] and self.available(c, on):
                return prefer
        pool = [c for c in self.clinicians.get(provider, [])
                if family in c["privileges"] and (specialty is None or c["specialty"] == specialty)
                and self.available(c, on)]
        if not pool:
            return None
        return pool[int(self.rng.integers(len(pool)))]["id"]

    def book(self, clin: str, on: D, start: int, minutes: int) -> bool:
        slots = self.bookings[(clin, on)]
        end = start + max(int(minutes), 1)
        for s, e in slots:
            if start < e and s < end:
                return False
        slots.append((start, end))
        return True

    def find_slot(self, clin: str, on: D, minutes: int, lo: int, hi: int, tries: int = 40) -> int | None:
        for _ in range(tries):
            start = int(self.rng.integers(lo // 5, max(hi // 5, lo // 5 + 1))) * 5
            if self.book(clin, on, start, minutes):
                return start
        return None

    # ---------------------------------------------------------------- cover

    def coverage_at(self, member: str, on: D) -> dict | None:
        m = self.members.get(member)
        if not m:
            return None
        for cov in m["coverage"]:
            if cov["valid_from"] <= on <= cov["valid_to"]:
                return cov
        return None

    def share_pct(self, product: str, family: str, on: D) -> float:
        rule = self.benefit.get((product, family, _year_key(on)))
        return 0.0 if rule is None else rule["patient_share_pct"]

    def needs_auth(self, product: str, family: str, on: D) -> bool:
        rule = self.benefit.get((product, family, _year_key(on)))
        return bool(rule and rule["authorization_required"])


# =============================================================================
# entities
# =============================================================================


def _allocate(total: int, shares: list[float], minimum: int = 0) -> list[int]:
    raw = np.array(shares) / sum(shares) * total
    out = np.floor(raw).astype(int)
    out = np.maximum(out, minimum if total >= minimum * len(shares) else 0)
    rem = total - out.sum()
    order = np.argsort(-(raw - np.floor(raw)))
    i = 0
    while rem > 0:
        out[order[i % len(order)]] += 1
        rem -= 1
        i += 1
    while rem < 0:
        j = int(np.argmax(out))
        out[j] -= 1
        rem += 1
    return out.tolist()


def _build_providers(ctx: _Ctx, scale: float) -> None:
    rng = ctx.rng
    n = {
        "HOSPITAL": max(6, round(30 * scale)), "CLINIC": max(14, round(110 * scale)),
        "PHARMACY": max(7, round(50 * scale)), "DIAGNOSTIC_LAB": max(4, round(25 * scale)),
        "RADIOLOGY_CENTRE": max(4, round(25 * scale)),
    }
    weights = [e[2] for e in EMIRATES]
    rows, status, contracts, equipment, clin_rows = [], [], [], [], []
    for ptype, count in n.items():
        per_emirate = _allocate(count, weights, minimum=1)
        # clinic kinds for this type
        kinds: list[tuple] = []
        if ptype == "CLINIC":
            kind_counts = _allocate(count, [k[1] for k in CLINIC_KINDS], minimum=0)
            for kind, k in zip(CLINIC_KINDS, kind_counts):
                kinds += [kind] * k
            order = rng.permutation(len(kinds))
            # GP clinics first in each emirate so every emirate has primary care
            gp = [kinds[i] for i in order if kinds[i][0] in ("GP", "POLY")]
            other = [kinds[i] for i in order if kinds[i][0] not in ("GP", "POLY")]
            kinds = []
            gi = oi = 0
            for e_count in per_emirate:
                for j in range(e_count):
                    if (j == 0 or oi >= len(other)) and gi < len(gp):
                        kinds.append(gp[gi]); gi += 1
                    elif oi < len(other):
                        kinds.append(other[oi]); oi += 1
                    else:
                        kinds.append(gp[gi]); gi += 1
        idx = 0
        for emirate, e_count in zip(EMIRATE_NAMES, per_emirate):
            for j in range(e_count):
                sk = ctx.world.new_id("PRV", 4)
                reg = REGULATOR[emirate]
                tier = "TIER_3" if j == 0 else _pick(rng, list(TIERS), [0.22, 0.35, 0.43])
                cred = D(2008, 1, 1) + TD(days=int(rng.integers(0, 365 * 15)))
                beds = None
                facility, specialty = {
                    "HOSPITAL": ("GENERAL_HOSPITAL", "MULTISPECIALTY"),
                    "PHARMACY": ("RETAIL_PHARMACY", "PHARMACY"),
                    "DIAGNOSTIC_LAB": ("CLINICAL_LABORATORY", "PATHOLOGY"),
                    "RADIOLOGY_CENTRE": ("IMAGING_CENTRE", "RADIOLOGY"),
                }.get(ptype, (None, None))
                kind = None
                if ptype == "CLINIC":
                    kind = kinds[idx]
                    facility = kind[2]
                    specialty = {"GP": "GENERAL_PRACTICE", "POLY": "MULTISPECIALTY"}.get(kind[0], kind[0])
                if ptype == "HOSPITAL":
                    beds = int(rng.choice([60, 80, 100, 120, 150, 200, 250, 300, 400, 550]))
                    if beds >= 250 and rng.random() < 0.5:
                        facility = "TERTIARY_HOSPITAL"
                regulator_id = {"DHA": f"DHA-F-{int(rng.integers(10 ** 6, 10 ** 7)):07d}",
                                "DOH": f"MF{int(rng.integers(1000, 9999))}",
                                "MOHAP": f"MOH-F-{int(rng.integers(10 ** 4, 10 ** 5)):05d}"}[reg]
                info = dict(sk=sk, type=ptype, emirate=emirate, tier=tier, facility=facility,
                            specialty=specialty, kind=kind, weight=float(rng.lognormal(0, 0.45)),
                            equipment={}, specialties=set(), beds=beds)
                ctx.providers[sk] = info
                rows.append({
                    "provider_sk": sk, "source_provider_id": f"SRC-{sk}", "regulator_id": regulator_id,
                    "provider_type": ptype, "specialty": specialty, "facility_type": facility,
                    "owner_entity_id": _token(rng, "OWN-", 12), "bank_account_token": _token(rng, "IBANTOK-", 16),
                    "phone_token": _token(rng, "TELTOK-", 12), "address_token": _token(rng, "ADDRTOK-", 14),
                    "emirate": emirate, "tenant_id": TENANT,
                    "credentialing_date": cred, "bed_count": beds,
                    "operational_status": "OPERATIONAL", "ownership_changed_on": None,
                    "licence_no": f"{reg}-LIC-{cred.year}-{int(rng.integers(10000, 99999))}",
                })
                status.append({"provider_sk": sk, "status_type": "LICENCE", "status_value": "ACTIVE",
                               "valid_from": cred, "valid_to": D(2026, 12, 31) + TD(days=int(rng.integers(0, 730))),
                               "source": f"Synthetic {reg} facility licence register", "match_type": "EXACT",
                               "version_id": "SYN-REG-2024.1"})
                status.append({"provider_sk": sk, "status_type": "NETWORK", "status_value": "IN_NETWORK",
                               "valid_from": D(2023, 1, 1), "valid_to": D(2026, 12, 31),
                               "source": "Synthetic payer network file", "match_type": "EXACT",
                               "version_id": "SYN-NET-2024.1"})
                disc_range = {"HOSPITAL": (0.05, 0.12), "CLINIC": (0.03, 0.10), "PHARMACY": (0.0, 0.02),
                              "DIAGNOSTIC_LAB": (0.06, 0.15), "RADIOLOGY_CENTRE": (0.05, 0.12)}[ptype]
                for payer in PAYERS:
                    disc = round(float(rng.uniform(*disc_range)), 3)
                    ctx.contracts[(sk, payer)] = {"discount": disc, "tier": tier}
                    contracts.append({"contract_sk": f"CTR-{sk}-{payer[-1]}", "provider_sk": sk, "payer_id": payer,
                                      "valid_from": D(2023, 1, 1), "valid_to": D(2026, 12, 31),
                                      "tariff_basis": "SYN-TARIFF-2024/2025", "discount_pct": disc,
                                      "network_tier": tier, "tenant_id": TENANT})
                idx += 1
    ctx.world.append("provider", rows)
    ctx.world.append("provider_status_period", status)
    ctx.world.append("contract", contracts)

    # --- clinicians ------------------------------------------------------------
    for sk, info in ctx.providers.items():
        plan: dict[str, int] = {}
        t = info["type"]
        if t == "HOSPITAL":
            for spec, (lo, hi) in HOSPITAL_REQUIRED.items():
                plan[spec] = int(rng.integers(lo, hi + 1))
            target = int(rng.integers(15, 41))
            while sum(plan.values()) < target:
                s = _pick(rng, HOSPITAL_OPTIONAL)
                plan[s] = plan.get(s, 0) + 1
        elif t == "CLINIC":
            for spec, (lo, hi) in info["kind"][3].items():
                k = int(rng.integers(lo, hi + 1))
                if k:
                    plan[spec] = k
            if info["kind"][0] == "POLY":
                for s in rng.choice(POLY_SPECIALISTS, size=int(rng.integers(2, 4)), replace=False):
                    plan[str(s)] = plan.get(str(s), 0) + 1
            total = sum(plan.values())
            if total < 3:
                first = next(iter(plan))
                plan[first] += 3 - total
            while sum(plan.values()) > 6:
                big = max(plan, key=plan.get)
                plan[big] -= 1
        elif t == "PHARMACY":
            plan["PHARMACY"] = int(rng.integers(1, 4))
        elif t == "DIAGNOSTIC_LAB":
            plan["PATHOLOGY"] = int(rng.integers(2, 6))
        else:
            plan["RADIOLOGY"] = int(rng.integers(2, 6))
        slot = int(rng.integers(12))
        for spec in sorted(plan):
            for _ in range(plan[spec]):
                cid = ctx.world.new_id("CL", 5)
                lic_from = D(2012, 1, 1) + TD(days=int(rng.integers(0, 365 * 11)))
                lic_to = D(2026, 6, 30) + TD(days=int(rng.integers(0, 900)))
                leave = []
                for year in (2024, 2025):
                    month = slot % 12 + 1
                    slot += 5
                    start = D(year, month, int(rng.integers(1, 12)))
                    length = int(rng.integers(12, 25))
                    leave.append((start, start + TD(days=length - 1), "ANNUAL"))
                    if rng.random() < 0.3:
                        s2 = D(year, int(rng.integers(1, 13)), int(rng.integers(1, 25)))
                        leave.append((s2, s2 + TD(days=int(rng.integers(2, 5))), "SICK" if rng.random() < .5 else "CONFERENCE"))
                privileges = PRIVILEGES[spec]
                clin = dict(id=cid, provider=sk, specialty=spec, privileges=set(privileges),
                            lic_from=lic_from, lic_to=lic_to, leave=[(a, b) for a, b, _ in leave])
                ctx.clinicians[sk].append(clin)
                ctx.clin_by_id[cid] = clin
                info["specialties"].add(spec)
                role = ROLE_OF.get(spec, "SPECIALIST" if t != "CLINIC" or spec not in GP_SPECS else "GP")
                if spec in ("CARDIOLOGY", "GENERAL_SURGERY", "ORTHOPAEDICS") and t == "HOSPITAL" and rng.random() < .5:
                    role = "CONSULTANT"
                reg = REGULATOR[info["emirate"]]
                clin_rows.append({
                    "clinician_id": cid, "provider_sk": sk, "full_name_token": _token(rng, "NAMETOK-", 14),
                    "specialty": spec, "role": role,
                    "licence_no": {"DHA": "DHA-P-", "DOH": "DOH-GD-", "MOHAP": "MOHAP-P-"}[reg] + f"{int(rng.integers(10 ** 6, 10 ** 7))}",
                    "licence_valid_from": lic_from, "licence_valid_to": lic_to,
                    "privileges": ";".join(privileges),
                    "leave_periods": json.dumps([{"from": a.isoformat(), "to": b.isoformat(), "type": k}
                                                 for a, b, k in sorted(leave)]),
                    "emirate": info["emirate"], "tenant_id": TENANT,
                })
    ctx.world.append("clinician_roster", clin_rows)

    # --- equipment ------------------------------------------------------------
    pet_budget = 3
    for sk, info in ctx.providers.items():
        eq: dict[str, int] = {}
        if info["type"] == "HOSPITAL":
            eq = {"XRAY": int(rng.integers(2, 5)), "ULTRASOUND": int(rng.integers(2, 6)), "CT": int(rng.integers(1, 3))}
            if info["beds"] >= 150 or rng.random() < 0.5:
                eq["MRI"] = int(rng.integers(1, 3))
            if "CARDIOLOGY" in info["specialties"] and info["beds"] >= 100:
                eq["CATH_LAB"] = int(rng.integers(1, 3))
            if rng.random() < 0.4:
                eq["MAMMOGRAPHY"] = 1
            if rng.random() < 0.3:
                eq["DEXA"] = 1
        elif info["type"] == "RADIOLOGY_CENTRE":
            eq = {"XRAY": int(rng.integers(1, 3)), "ULTRASOUND": int(rng.integers(1, 4))}
            if rng.random() < 0.8:
                eq["CT"] = 1
            if rng.random() < 0.65:
                eq["MRI"] = int(rng.integers(1, 3))
            if rng.random() < 0.5:
                eq["MAMMOGRAPHY"] = 1
            if rng.random() < 0.45:
                eq["DEXA"] = 1
            if pet_budget and info["emirate"] in ("Dubai", "Abu Dhabi", "Sharjah"):
                eq["PET_CT"] = 1
                pet_budget -= 1
        elif info["type"] == "CLINIC" and "OBSTETRICS_GYNAECOLOGY" in info["specialties"]:
            eq = {"ULTRASOUND": 1}
        info["equipment"] = eq
        for etype, units in sorted(eq.items()):
            codes, mins, hours = EQUIPMENT[etype]
            if info["type"] == "HOSPITAL" and etype == "CT":
                hours = 24
            equipment.append({"provider_sk": sk, "equipment_type": etype, "units": units,
                              "activity_codes": ";".join(codes), "minutes_per_use": mins, "hours_per_day": hours})
            info.setdefault("capacity", {})[etype] = units * hours * 60 // mins
    ctx.world.append("equipment_inventory", equipment)

    # --- tariff -----------------------------------------------------------------
    tariff = []
    all_codes = [c for c in ctx.codes] + [p for p in ctx.drugs if not p.startswith("J")]
    for code in all_codes:
        for tier in TIERS:
            for payer in PAYERS:
                for year, (a, b) in ((2024, (D(2024, 1, 1), D(2024, 12, 31))), (2025, (D(2025, 1, 1), D(2025, 12, 31)))):
                    tariff.append({"activity_code": code, "network_tier": tier,
                                   "allowed_price": ctx.allowed_price(code, tier, payer, a),
                                   "valid_from": a, "valid_to": b, "payer_id": payer})
    ctx.world.append("tariff", tariff)


def _build_people(ctx: _Ctx, n_members: int, scale: float) -> None:
    rng, world = ctx.rng, ctx.world
    start, end = world.start, world.end
    n_agents = max(8, round(45 * scale))
    n_employers = max(15, round(120 * scale))
    agents = [f"AGT{i:03d}" for i in range(1, n_agents + 1)]
    agent_w = rng.lognormal(0, 0.5, n_agents)
    employers = []
    for i in range(1, n_employers + 1):
        emirate = _pick(rng, EMIRATE_NAMES, [e[2] for e in EMIRATES])
        product = _pick(rng, ["GROUP_ESSENTIAL", "GROUP_ENHANCED", "FAMILY_PREMIER"], [0.45, 0.4, 0.15])
        payer = _pick(rng, list(PRODUCTS[product]["payers"]), PRODUCTS[product]["payers"].values())
        employers.append(dict(id=f"EMP{i:04d}", emirate=emirate, product=product, payer=payer,
                              agent=_pick(rng, agents, agent_w), size=float(rng.pareto(1.2) + 1)))
    emp_w = np.array([e["size"] for e in employers])

    member_rows, cov_rows, events, roster, apps, prior, cob = [], [], [], [], [], [], []
    count = 0
    while count < n_members:
        individual = rng.random() < 0.15
        if individual:
            emp = None
            product = "INDIVIDUAL_PLUS"
            payer = _pick(rng, list(PRODUCTS[product]["payers"]), PRODUCTS[product]["payers"].values())
            emirate = _pick(rng, EMIRATE_NAMES, [e[2] for e in EMIRATES])
            agent = _pick(rng, agents, agent_w)
        else:
            emp = employers[int(rng.choice(len(employers), p=emp_w / emp_w.sum()))]
            product, payer, agent = emp["product"], emp["payer"], emp["agent"]
            emirate = emp["emirate"] if rng.random() < 0.85 else _pick(rng, EMIRATE_NAMES, [e[2] for e in EMIRATES])
        # the family's active window (bounded and staggered — stationarity)
        length = int(rng.integers(240, 821))
        ws = start + TD(days=int(rng.integers(-length + 45, (end - start).days - 45)))
        we = ws + TD(days=length)
        new_joiner = rng.random() < 0.35
        inception = ws if new_joiner else ws - TD(days=int(rng.integers(30, 1800)))
        cover_end = we if we < end else None
        # family composition
        principal_sex = "M" if rng.random() < 0.62 else "F"
        p_age = float(rng.uniform(22, 62))
        fam = [("PRINCIPAL", principal_sex, p_age)]
        if not individual or rng.random() < 0.5:
            if rng.random() < 0.55 and p_age > 24:
                fam.append(("SPOUSE", "F" if principal_sex == "M" else "M",
                            float(np.clip(p_age + rng.normal(-2, 4), 20, 70))))
                for _ in range(int(rng.choice([0, 1, 1, 2, 2, 3]))):
                    fam.append(("CHILD", "M" if rng.random() < .5 else "F",
                                float(np.clip(rng.uniform(0, min(p_age - 20, 21)), 0.2, 21))))
        principal_sk = None
        for relationship, sex, age0 in fam:
            count += 1
            msk = world.new_id("MBR", 6)
            principal_sk = principal_sk or msk
            dob = start - TD(days=int(age0 * 365.25))
            dob = dob.replace(day=min(dob.day, 28))
            # anthropometrics at period start
            a = age0
            if a < 18:
                height = 50 + 25 * a if a < 1 else 75 + 6.0 * min(a, 13) + (4.0 * (a - 13) if a > 13 else 0)
                height = float(np.clip(height + rng.normal(0, 4), 45, 190))
                bmi = float(np.clip(rng.normal(16 + 0.25 * max(a - 6, 0), 1.6), 13, 30))
            else:
                height = float(rng.normal(173 if sex == "M" else 160, 7))
                bmi = float(np.clip(rng.normal(27.5, 4.5), 17, 45))
            weight = round(bmi * (height / 100) ** 2, 1)
            chronic = []
            for code, spec in CHRONIC.items():
                if a < spec.get("min_age", 0):
                    continue
                p = _band(spec["prev"], a) * spec.get("sexw", {}).get(sex, 1.0)
                if code == "E66.9":
                    p = 1.0 if bmi >= 30 and a >= 18 else 0.0
                if rng.random() < p:
                    chronic.append(code)
            drug_choice = {}
            for code in chronic:
                picks = []
                for prob, options in CHRONIC[code].get("drugs", []):
                    if rng.random() < prob:
                        picks.append(_pick(rng, [o[0] for o in options], [o[1] for o in options]))
                drug_choice[code] = picks
            member = dict(sk=msk, dob=dob, sex=sex, weight=weight, height=round(height, 1), emirate=emirate,
                          product=product, payer=payer, tpa=TPA_OF[(payer, product)], agent=agent,
                          employer=None if emp is None else emp["id"], sponsor=principal_sk,
                          relationship=relationship, window=(max(ws, start), min(we, end)),
                          inception=inception, chronic=chronic, drugs=drug_choice, coverage=[],
                          util=float(rng.lognormal(0, 0.35)))
            ctx.members[msk] = member
            member_rows.append({
                "member_sk": msk, "source_member_id": f"CARD-{int(rng.integers(10 ** 9, 10 ** 10))}",
                "protected_id_token": _token(rng, "EIDTOK-", 24), "date_of_birth": dob, "sex": sex,
                "death_date": None, "death_source": None, "death_source_confidence": None,
                "sponsor_id": principal_sk, "employer_id": member["employer"], "tenant_id": TENANT,
                "relationship": relationship,
                "weight_kg": weight, "height_cm": round(height, 1), "emirate": emirate,
            })
            # coverage terms: policy years anchored on the inception anniversary
            k = 0
            while True:
                anniv = _add_years(inception, k)
                nxt = _add_years(inception, k + 1) - TD(days=1)
                k += 1
                if nxt < start:
                    continue
                if anniv > end or (cover_end is not None and anniv > cover_end):
                    break
                vf, vt = anniv, nxt
                if cover_end is not None and vt > cover_end:
                    vt = cover_end
                cid = world.new_id("COV", 7)
                cov = dict(coverage_id=cid, valid_from=vf, valid_to=vt, product=product, payer=payer,
                           network=PRODUCTS[product]["network"], agent=agent)
                member["coverage"].append(cov)
                cov_rows.append({
                    "coverage_id": cid, "member_sk": msk, "product": product, "payer_id": payer,
                    "valid_from": vf, "valid_to": vt, "network": PRODUCTS[product]["network"], "status": "ACTIVE",
                    "policy_inception_date": inception, "tenant_id": TENANT,
                    "agent_id": agent, "product_tier": PRODUCTS[product]["tier"],
                })
                first = vf == inception
                ev_time = _dt_at(vf - TD(days=int(rng.integers(3, 30))), int(rng.integers(8 * 60, 17 * 60)))
                events.append({"policy_event_sk": None, "coverage_id": cid, "event_type": "ADD" if first else "RENEWAL",
                               "actor": agent if individual else f"HR-{member['employer']}", "event_time": ev_time,
                               "effective_date": vf, "agent_id": agent, "tenant_id": TENANT})
                if cover_end is not None and vt == cover_end:
                    events.append({"policy_event_sk": None, "coverage_id": cid, "event_type": "DELETE",
                                   "actor": agent if individual else f"HR-{member['employer']}",
                                   "event_time": _dt_at(vt - TD(days=int(rng.integers(5, 25))), int(rng.integers(8 * 60, 17 * 60))),
                                   "effective_date": vt + TD(days=1), "agent_id": agent, "tenant_id": TENANT})
            if member["coverage"] and rng.random() < 0.05:
                cov0 = member["coverage"][0]
                t0 = min(cov0["valid_from"], member["window"][0]) - TD(days=int(rng.integers(1, 20)))
                events.append({"policy_event_sk": None, "coverage_id": cov0["coverage_id"], "event_type": "CORRECTION",
                               "actor": f"OPS-{int(rng.integers(1, 9)):02d}", "event_time": _dt_at(t0, int(rng.integers(8 * 60, 17 * 60))),
                               "effective_date": cov0["valid_from"], "agent_id": agent, "tenant_id": TENANT})
            if emp is not None:
                roster.append({"employer_id": emp["id"], "member_sk": msk, "sponsor_id": principal_sk,
                               "relationship": relationship, "valid_from": inception,
                               "valid_to": cover_end if cover_end is not None else OPEN_END, "tenant_id": TENANT})
            has_prior = (not new_joiner and rng.random() < 0.3) or (new_joiner and rng.random() < 0.45)
            apps.append({"application_sk": None, "member_sk": msk,
                         "coverage_id": member["coverage"][0]["coverage_id"] if member["coverage"] else None,
                         "application_date": inception - TD(days=int(rng.integers(7, 40))),
                         "declared_conditions": ";".join(sorted(c for c in chronic if c != "E66.9")) or "NONE",
                         "declared_prior_cover": bool(has_prior), "agent_id": agent, "tenant_id": TENANT})
            if has_prior:
                pv_to = inception - TD(days=1)
                pv_from = pv_to - TD(days=int(rng.integers(365, 365 * 5)))
                treated = [c for c in chronic if c != "E66.9" and rng.random() < 0.7]
                prior.append({"member_sk": msk, "prior_payer": f"SYN-PRIOR-INSURER-{int(rng.integers(1, 5))}",
                              "valid_from": pv_from, "valid_to": pv_to,
                              "conditions_treated": ";".join(treated) or "NONE", "tenant_id": TENANT})
            if rng.random() < 0.03 and member["coverage"] and a >= 18:
                other = _pick(rng, [p for p in PAYERS if p != payer])
                member["cob"] = other
                cob.append({"member_sk": msk, "primary_payer": payer, "secondary_payer": other,
                            "valid_from": member["coverage"][0]["valid_from"],
                            "valid_to": member["coverage"][-1]["valid_to"], "tenant_id": TENANT})

    for i, e in enumerate(sorted(events, key=lambda r: (r["event_time"], r["coverage_id"]))):
        e["policy_event_sk"] = f"PEV{i + 1:07d}"
    events.sort(key=lambda r: r["policy_event_sk"])
    for i, a in enumerate(apps):
        a["application_sk"] = f"APP{i + 1:07d}"
    world.append("member", member_rows)
    world.append("coverage_period", cov_rows)
    world.append("policy_event", events)
    world.append("employer_roster", roster)
    world.append("policy_application", apps)
    world.append("prior_coverage_history", prior)
    world.append("coordination_of_benefits", cob)
    ctx.agents = agents

    # benefit rules
    rules = []
    for product, pdef in PRODUCTS.items():
        label = pdef["tier"]
        for fam in FAMILIES:
            for year, (a, b) in ((2024, (D(2024, 1, 1), D(2024, 12, 31))), (2025, (D(2025, 1, 1), D(2025, 12, 31)))):
                pct = 1.0 if fam == "COSMETIC" else _SHARE[label][fam]
                if year == 2025:
                    pct = _SHARE_2025.get((label, fam), pct)
                rule = dict(product=product, service_family=fam, covered=fam != "COSMETIC",
                            benefit_limit=float(_LIMIT[fam] * (1.5 if label == "PREMIER" else 1.0)),
                            patient_share_pct=pct, authorization_required=fam in AUTH_FAMILIES,
                            exceptions=json.dumps({"benefit_family_rule": "inpatient claims use the case-rate family"}
                                                  if fam in ("INPATIENT", "DAY_SURGERY") else {}),
                            valid_from=a, valid_to=b, version_id=f"BRV-{product}-{year}",
                            recorded_at=_dt_at(a - TD(days=30), 9 * 60), source="Synthetic policy wording")
                ctx.benefit[(product, fam, year)] = rule
                rules.append(rule)
    world.append("benefit_rule_version", rules)


def _add_years(d: D, k: int) -> D:
    try:
        return d.replace(year=d.year + k)
    except ValueError:
        return d.replace(year=d.year + k, day=28)


# =============================================================================
# provider choice
# =============================================================================


def _eligible(ctx: _Ctx, member: dict, ptype: str | None = None, specialty: str | None = None,
              equipment: str | None = None, family: str | None = None) -> list[str]:
    tiers = PRODUCTS[member["product"]]["tiers"]
    out = []
    for emirates in ([member["emirate"]], NEIGHBOURS[member["emirate"]], EMIRATE_NAMES):
        for sk, info in ctx.providers.items():
            if info["emirate"] not in emirates or info["tier"] not in tiers:
                continue
            if ptype and info["type"] != ptype:
                continue
            if specialty and specialty not in info["specialties"]:
                continue
            if equipment and equipment not in info["equipment"]:
                continue
            out.append(sk)
        if out:
            return out
    return out


def _choose(ctx: _Ctx, candidates: list[str]) -> str | None:
    if not candidates:
        return None
    return _pick(ctx.rng, candidates, [ctx.providers[c]["weight"] for c in candidates])


def _home(ctx: _Ctx, member: dict, key: str, **kw) -> str | None:
    homes = member.setdefault("homes", {})
    if key not in homes:
        homes[key] = _choose(ctx, _eligible(ctx, member, **kw))
    return homes[key]


def _clinic_for(ctx: _Ctx, member: dict, specialty: str) -> str | None:
    """A provider (clinic preferred, else hospital) with this specialty, sticky per member."""
    homes = member.setdefault("homes", {})
    key = f"SPEC:{specialty}"
    if key in homes:
        return homes[key]
    cands = [c for c in _eligible(ctx, member, specialty=specialty) if ctx.providers[c]["type"] == "CLINIC"]
    if not cands:
        cands = _eligible(ctx, member, specialty=specialty)
    homes[key] = _choose(ctx, cands)
    return homes[key]


# =============================================================================
# event scheduling
# =============================================================================


class _Sched:
    def __init__(self, ctx: _Ctx) -> None:
        self.ctx = ctx
        self.events: list[dict] = []
        self.by_eid: dict[int, dict] = {}
        self.n = 0

    def add(self, rtype: str | None = None, root: Any = None, **ev) -> dict:
        """Schedule one event. Events sharing a ``root`` are kept or dropped together."""
        self.n += 1
        ev["eid"] = self.n
        ev.setdefault("parent", None)
        ev.setdefault("prio", 5)
        if ev["parent"] is not None and ev["parent"] in self.by_eid:
            p = self.by_eid[ev["parent"]]
            ev["root"], ev["rtype"] = p["root"], p["rtype"]
        else:
            ev["root"] = root if root is not None else ("EV", self.n)
            ev["rtype"] = rtype or "ACUTE"
        self.by_eid[self.n] = ev
        if "provider" in ev and ev["provider"] is None:
            return ev  # nowhere eligible to go: the event (and anything hanging off it) is not scheduled
        self.events.append(ev)
        return ev


def _in_window(m: dict, d: D) -> bool:
    return m["window"][0] <= d <= m["window"][1]


def _season(d: D) -> float:
    return 1.0 if d.month in (4, 5, 6, 7, 8, 9, 10) else 1.0  # applied via acceptance below


def _schedule(ctx: _Ctx, mult: dict[str, float]) -> _Sched:
    rng = ctx.rng
    s = _Sched(ctx)
    acute_w = np.array([sc["w"] for sc in ACUTE])
    for m in ctx.members.values():
        a, b = m["window"]
        if b < a:
            continue
        span = (b - a).days + 1
        mid_age = _age(m["dob"], a + TD(days=span // 2))
        # ---- acute consultations
        rate = (3.2 if mid_age < 6 else 2.0 if mid_age < 14 else 1.1 if mid_age < 45 else 1.5 if mid_age < 65 else 2.2)
        rate *= mult["acute"] * m["util"]
        for _ in range(int(rng.poisson(rate * span / 365.0))):
            d = a + TD(days=int(rng.integers(0, span)))
            age = _age(m["dob"], d)
            w = acute_w.copy()
            for i, sc in enumerate(ACUTE):
                lo, hi = sc.get("age", (0, 200))
                if not lo <= age <= hi:
                    w[i] = 0
                w[i] *= sc.get("sexw", {}).get(m["sex"], 1.0)
                if d.month in (11, 12, 1, 2, 3):
                    w[i] *= sc.get("season", 1.0)
            if w.sum() <= 0:
                continue
            sc = ACUTE[int(rng.choice(len(ACUTE), p=w / w.sum()))]
            _acute_episode(ctx, s, m, sc, d)
        # ---- chronic follow-up streams
        chronic_visits: dict[str, list[dict]] = {}
        for cond in m["chronic"]:
            spec = CHRONIC[cond]
            if not spec["interval"]:
                continue
            T = spec["interval"]
            if spec.get("chemo"):
                _chemo_stream(ctx, s, m, cond)
                continue
            d = a + TD(days=int(rng.integers(0, min(T, span))))
            visits = []
            while d <= b:
                if rng.random() < mult["chronic"]:
                    visits.append(_chronic_visit(ctx, s, m, cond, d))
                d += TD(days=int(T * rng.uniform(0.92, 1.12)))
            chronic_visits[cond] = [v for v in visits if v is not None]
        _refill_stream(ctx, s, m, chronic_visits)
        # ---- inpatient
        for cr, bands in EMERGENCY_IP.items():
            r = _band(bands, mid_age) * mult["ip"]
            cdef = ctx.case_rates[cr]
            if cdef.get("sex") and cdef["sex"] != m["sex"]:
                continue
            for _ in range(int(rng.poisson(r * span / 365.0))):
                d = a + TD(days=int(rng.integers(0, span)))
                age = _age(m["dob"], d)
                if not cdef["age"][0] <= age <= cdef["age"][1]:
                    continue
                s.add(kind="IP", member=m["sk"], date=d, cr=cr, prio=1, rtype="IP")
        for cr, bands in ELECTIVE_IP.items():
            r = _band(bands, mid_age) * mult["ip"]
            for _ in range(int(rng.poisson(r * span / 365.0))):
                d = a + TD(days=int(rng.integers(0, span)))
                _elective_chain(ctx, s, m, cr, d)
        # ---- pregnancy
        if m["sex"] == "F" and m["relationship"] in ("PRINCIPAL", "SPOUSE") and 20 <= mid_age <= 41:
            if rng.random() < 0.10 * span / 365.0 * mult["ip"] * 1.6:
                _pregnancy(ctx, s, m)
    return s


def _acute_episode(ctx: _Ctx, s: _Sched, m: dict, sc: dict, d: D) -> None:
    rng = ctx.rng
    age = _age(m["dob"], d)
    if sc.get("vaccine"):
        prov = _home(ctx, m, "GP", ptype="CLINIC", specialty="PAEDIATRICS" if age < 14 else None)
        s.add(kind="OP", member=m["sk"], date=d, provider=prov, scenario=sc["key"], dx=[sc["dx"]], setting="VACCINE")
        return
    ed = rng.random() < sc.get("ed", 0.0)
    tele = (not ed) and rng.random() < sc.get("tele", 0.0)
    if ed:
        prov = _home(ctx, m, "HOSP", ptype="HOSPITAL")
        setting = "ED"
        spec = "EMERGENCY_MEDICINE"
    elif sc.get("spec") and rng.random() < sc.get("specp", 0):
        spec = sc["spec"]
        prov = _clinic_for(ctx, m, spec)
        setting = "CLINIC"
    else:
        spec = "PAEDIATRICS" if age < 14 and rng.random() < 0.6 else None
        prov = _clinic_for(ctx, m, "PAEDIATRICS") if spec else _home(ctx, m, "GP", ptype="CLINIC")
        if prov and spec is None:
            gps = ctx.providers[prov]["specialties"] & set(GP_SPECS)
            spec = sorted(gps)[0] if gps else None
        setting = "CLINIC"
    if prov is None:
        return
    if tele:
        setting = "TELE"
    # dx: secondary = up to 1 chronic condition
    dx = [sc["dx"]]
    sec = [c for c in m["chronic"] if c in CHRONIC_SECONDARY_OK]
    if sec and rng.random() < 0.3:
        dx.append(sec[int(rng.integers(len(sec)))])
    ev = s.add(kind="OP", member=m["sk"], date=d, provider=prov, scenario=sc["key"], dx=dx, setting=setting,
               specialty=spec, level=_level_from(rng, sc["levels"]) if sc["levels"] else None)
    if tele:
        rxs = _rx_choices(ctx, m, sc, d)
        if rxs:
            s.add(kind="RX", member=m["sk"], date=d + TD(days=int(rng.integers(0, 2))), parent=ev["eid"],
                  items=rxs, dx=dx[:1], provider=_pharmacy_for(ctx, m), prio=7)
        return
    hospital = ctx.providers[prov]["type"] == "HOSPITAL"
    # office procedures and POC tests
    office = [code for p, code in sc.get("office", []) if rng.random() < p]
    ev["office"] = office
    ev["dme"] = [code for p, code in sc.get("dme", []) if rng.random() < p]
    labs = [c for p, codes in sc.get("labs", []) if rng.random() < p for c in codes]
    imaging = [c for p, codes in sc.get("imaging", []) if rng.random() < p for c in codes]
    rxs = _rx_choices(ctx, m, sc, d) if rng.random() < 0.55 or hospital_visit(ctx, prov) else []
    ev["injury"] = bool(sc.get("injury"))
    if hospital:
        ev["labs"], ev["imaging"], ev["rx_inhouse"] = labs, imaging, rxs
    else:
        if labs:
            lab = _home(ctx, m, "LAB", ptype="DIAGNOSTIC_LAB")
            if lab:
                s.add(kind="LAB", member=m["sk"], date=d + TD(days=int(rng.integers(0, 3))), parent=ev["eid"],
                      provider=lab, codes=labs, dx=dx[:1], prio=6)
        if imaging:
            rad = _radiology_for(ctx, m, imaging)
            if rad:
                s.add(kind="RAD", member=m["sk"], date=d + TD(days=int(rng.integers(0, 4))), parent=ev["eid"],
                      provider=rad, codes=imaging, dx=dx[:1], prio=6)
        if rxs:
            s.add(kind="RX", member=m["sk"], date=d + TD(days=int(rng.integers(0, 2))), parent=ev["eid"],
                  items=rxs, dx=dx[:1], provider=_pharmacy_for(ctx, m), prio=7)
    # specialist referral
    ref_spec = sc.get("refer")
    if ref_spec and rng.random() < ref_spec[1]:
        spec_prov = _clinic_for(ctx, m, ref_spec[0])
        if spec_prov:
            dd = d + TD(days=int(rng.integers(3, 22)))
            child = s.add(kind="OP", member=m["sk"], date=dd, provider=spec_prov, scenario=sc["key"], dx=dx[:1],
                          setting="REFERRAL", specialty=ref_spec[0], level=_level_from(rng, {3: .4, 4: .5, 5: .1}),
                          parent=ev["eid"])
            _specialist_workup(ctx, s, m, sc, child, dd)
    elif sc.get("chain") and rng.random() < 0.6:
        _imaging_chain(ctx, s, m, sc["chain"], ev, d)
    if sc.get("physio") and rng.random() < sc["physio"]:
        _physio_course(ctx, s, m, ev, d, dx[:1])
    if sc.get("psych") and spec == "PSYCHIATRY":
        ev["psych"] = True


def hospital_visit(ctx: _Ctx, provider: str) -> bool:
    return ctx.providers[provider]["type"] == "HOSPITAL"


def _rx_choices(ctx: _Ctx, m: dict, sc: dict, d: D) -> list[str]:
    rng = ctx.rng
    age = _age(m["dob"], d)
    out = []
    for p, products in sc.get("rx", []):
        if rng.random() >= p:
            continue
        prod = products[int(rng.integers(len(products)))]
        if age < 12:
            prod = {"RX1001": "RX1003", "RX1002": "RX1003", "RX1033": "RX1034", "RX1035": "RX1036",
                    "RX1004": "RX1003", "RX1005": "RX1003"}.get(prod, prod if prod in (
                        "RX1003", "RX1034", "RX1036", "RX1055", "RX1056", "RX1059", "RX1041") else None)
            if prod == "RX1041" and age < 6:
                prod = None
        if prod in _OTC and rng.random() < 0.6:
            prod = None  # bought over the counter, never claimed
        if prod and prod not in out:
            out.append(prod)
    return out


#: Cheap over-the-counter products members usually buy without claiming.
_OTC = {"RX1033", "RX1034", "RX1035", "RX1036", "RX1041", "RX1042", "RX1059"}


def _pharmacy_for(ctx: _Ctx, m: dict, loyalty: float = 0.55) -> str | None:
    """The member's usual pharmacy most of the time, otherwise another nearby network pharmacy."""
    home = _home(ctx, m, "PHARM", ptype="PHARMACY")
    if ctx.rng.random() < loyalty:
        return home
    return _choose(ctx, _eligible(ctx, m, ptype="PHARMACY")) or home


def _radiology_for(ctx: _Ctx, m: dict, codes: list[str]) -> str | None:
    needed = {EQUIP_OF_CODE[c] for c in codes}
    home = _home(ctx, m, "RAD", ptype="RADIOLOGY_CENTRE")
    if home and needed <= set(ctx.providers[home]["equipment"]):
        return home
    cands = [p for p in _eligible(ctx, m, ptype="RADIOLOGY_CENTRE") if needed <= set(ctx.providers[p]["equipment"])]
    if not cands:
        cands = [p for p in _eligible(ctx, m, ptype="HOSPITAL") if needed <= set(ctx.providers[p]["equipment"])]
    if not cands:
        cands = [p for p in ctx.providers if needed <= set(ctx.providers[p]["equipment"])
                 and ctx.providers[p]["tier"] in PRODUCTS[m["product"]]["tiers"]
                 and ctx.providers[p]["type"] in ("RADIOLOGY_CENTRE", "HOSPITAL")]
    return _choose(ctx, sorted(cands))


def _specialist_workup(ctx: _Ctx, s: _Sched, m: dict, sc: dict, ev: dict, d: D) -> None:
    rng = ctx.rng
    key = sc["key"]
    if key == "KNEEPAIN" and rng.random() < 0.5:
        _imaging_chain(ctx, s, m, "KNEE", ev, d)
    elif key == "CHESTPAIN":
        prov = ev["provider"]
        if "CARDIOLOGY" in ctx.providers[prov]["specialties"]:
            ev["office"] = ["93000"] if rng.random() < .5 else []
            ev["echo"] = rng.random() < 0.6
    elif key == "MIGRAINE" and rng.random() < 0.3:
        rad = _radiology_for(ctx, m, ["70551"])
        if rad:
            s.add(kind="RAD", member=m["sk"], date=d + TD(days=int(rng.integers(2, 15))), parent=ev["eid"],
                  provider=rad, codes=["70551"], dx=["G43.909"], prio=6)
    elif key in ("RADICULOPATHY", "ROTATOR"):
        _imaging_chain(ctx, s, m, sc["chain"], ev, d)


def _imaging_chain(ctx: _Ctx, s: _Sched, m: dict, chain: str, parent: dict, d: D) -> None:
    """XR first, MRI later — the care pathway the policy requires."""
    rng = ctx.rng
    xr, mri, dx = {"KNEE": ("73562", "73721", "M25.561"), "LUMBAR": ("72100", "72148", "M54.16"),
                   "SHOULDER": ("73030", "73221", "M75.101")}[chain]
    rad_xr = _radiology_for(ctx, m, [xr])
    if not rad_xr:
        return
    d1 = d + TD(days=int(rng.integers(0, 4)))
    s.add(kind="RAD", member=m["sk"], date=d1, parent=parent["eid"], provider=rad_xr, codes=[xr], dx=[dx], prio=6)
    if rng.random() < 0.6:
        d2 = d1 + TD(days=int(rng.integers(12, 60)))
        rad = _radiology_for(ctx, m, [mri])
        if rad:
            s.add(kind="RAD", member=m["sk"], date=d2, parent=parent["eid"], provider=rad, codes=[mri], dx=[dx],
                  prio=6, needs_prior=xr)


def _physio_course(ctx: _Ctx, s: _Sched, m: dict, parent: dict, d: D, dx: list[str]) -> None:
    rng = ctx.rng
    prov = _clinic_for(ctx, m, "PHYSIOTHERAPY")
    if not prov:
        return
    first = d + TD(days=int(rng.integers(3, 12)))
    for k in range(int(rng.integers(4, 9))):
        s.add(kind="PHYSIO", member=m["sk"], date=first + TD(days=7 * k + int(rng.integers(0, 3))),
              parent=parent["eid"], provider=prov, dx=dx, first=k == 0, prio=6)


def _chronic_visit(ctx: _Ctx, s: _Sched, m: dict, cond: str, d: D) -> dict | None:
    rng = ctx.rng
    spec = CHRONIC[cond]
    age = _age(m["dob"], d)
    specs = [x for x in spec["spec"] if not (x == "PAEDIATRICS" and age >= 16)]
    if age < 14 and "PAEDIATRICS" not in specs and cond in ("J45.909",):
        specs = ["PAEDIATRICS"] + specs
    want = specs[0]
    prov = _clinic_for(ctx, m, want) if want not in GP_SPECS else _home(ctx, m, "GP", ptype="CLINIC")
    if prov is None:
        for alt in specs[1:]:
            prov = _clinic_for(ctx, m, alt) if alt not in GP_SPECS else _home(ctx, m, "GP", ptype="CLINIC")
            if prov:
                want = alt
                break
    if prov is None:
        return None
    if want in GP_SPECS:
        gps = ctx.providers[prov]["specialties"] & set(GP_SPECS)
        want = sorted(gps)[0] if gps else None
    dx = [cond] + [c for c in m["chronic"] if c != cond and c in CHRONIC_SECONDARY_OK][:2]
    if spec.get("infusion"):
        hosp = _home(ctx, m, "INFUSION", ptype="HOSPITAL", specialty="GASTROENTEROLOGY")
        if not hosp:
            return None
        return s.add(kind="OP", member=m["sk"], date=d, provider=hosp, scenario="INFUSION", dx=dx, setting="INFUSION",
                     specialty="GASTROENTEROLOGY", level=3, infusion=spec["infusion"], chronic=cond,
                     rtype="CHRONIC", root=("CHR", m["sk"]))
    tele = spec.get("psych") and rng.random() < 0.2
    ev = s.add(kind="OP", member=m["sk"], date=d, provider=prov, scenario=f"CHRONIC:{cond}", dx=dx,
               setting="TELE" if tele else "CLINIC", specialty=want, level=_level_from(rng, {3: .45, 4: .5, 5: .05}),
               chronic=cond, psych=bool(spec.get("psych")), rtype="CHRONIC", root=("CHR", m["sk"]))
    if tele:
        return ev
    ev["office"] = [code for p, code in spec.get("office", []) if rng.random() < p]
    labs = [c for p, codes in spec.get("labs", []) if rng.random() < p for c in codes]
    hospital = ctx.providers[prov]["type"] == "HOSPITAL"
    if labs:
        if hospital:
            ev["labs"] = labs
        else:
            lab = _home(ctx, m, "LAB", ptype="DIAGNOSTIC_LAB")
            if lab:
                s.add(kind="LAB", member=m["sk"], date=d + TD(days=int(rng.integers(0, 3))), parent=ev["eid"],
                      provider=lab, codes=labs, dx=[cond], prio=6)
    for p, codes in spec.get("imaging_at_radiology", []):
        if rng.random() < p:
            rad = _radiology_for(ctx, m, codes)
            if rad:
                s.add(kind="RAD", member=m["sk"], date=d + TD(days=int(rng.integers(1, 10))), parent=ev["eid"],
                      provider=rad, codes=codes, dx=[cond], prio=6)
    if spec.get("echo") and rng.random() < spec["echo"]:
        ev["echo"] = True
    if spec.get("denosumab") and rng.random() < spec["denosumab"] and "INJECTION" in PRIVILEGES.get(want or "", []):
        ev["injectable"] = "J0897"
    return ev


def _refill_stream(ctx: _Ctx, s: _Sched, m: dict, visits: dict[str, list[dict]]) -> None:
    """One monthly pharmacy cycle per member carrying every active chronic prescription."""
    rng = ctx.rng
    drug_visits = {c: v for c, v in visits.items() if v and m["drugs"].get(c)}
    if not drug_visits:
        return
    pharm = _home(ctx, m, "PHARM", ptype="PHARMACY")
    if not pharm:
        return
    first = min(v[0]["date"] for v in drug_visits.values())
    d = first + TD(days=int(rng.integers(0, 3)))
    a, b = m["window"]
    short = any((ctx.drugs[p]["max_duration_days"] or 365) <= 30 for c in drug_visits for p in m["drugs"][c] if p in ctx.drugs)
    cycle = m.setdefault("refill_cycle", 30 if short else 90)
    while d <= b:
        items = []
        for cond, vs in drug_visits.items():
            prior = [v for v in vs if v["date"] <= d and (d - v["date"]).days <= cycle + 90]
            if prior:
                items.append((cond, prior[-1]["eid"]))
        if items:
            s.add(kind="REFILL", member=m["sk"], date=d, provider=_pharmacy_for(ctx, m, 0.8) or pharm, items=items,
                  prio=7, cycle=cycle,
                  rtype="CHRONIC", root=("CHR", m["sk"]))
        d += TD(days=cycle + int(rng.integers(0, 4)))


def _chemo_stream(ctx: _Ctx, s: _Sched, m: dict, cond: str) -> None:
    rng = ctx.rng
    a, b = m["window"]
    hosp = _home(ctx, m, "ONC", ptype="HOSPITAL", specialty="ONCOLOGY")
    if not hosp:
        return
    start = a + TD(days=int(rng.integers(0, max((b - a).days - 60, 1))))
    # surgery first for about half of them (mastectomy with histology), then cycles
    if rng.random() < 0.5:
        s.add(kind="IP", member=m["sk"], date=start, cr="CR120", prio=1, provider_hint=hosp,
              rtype="CHRONIC", root=("CHR", m["sk"]))
        start += TD(days=int(rng.integers(21, 42)))
    for k in range(int(rng.integers(6, 13))):
        d = start + TD(days=21 * k)
        if d > b:
            break
        s.add(kind="OP", member=m["sk"], date=d, provider=hosp, scenario="CHEMO", dx=["C50.911"], setting="CHEMO",
              specialty="ONCOLOGY", level=4, chronic=cond, first=k == 0, rtype="CHRONIC", root=("CHR", m["sk"]))


def _elective_chain(ctx: _Ctx, s: _Sched, m: dict, cr: str, d: D) -> None:
    rng = ctx.rng
    cdef = ctx.case_rates[cr]
    age = _age(m["dob"], d)
    if not cdef["age"][0] <= age <= cdef["age"][1]:
        return
    if cdef.get("sex") and cdef["sex"] != m["sex"]:
        return
    spec = cdef["specialty"]
    hosp = _hospital_for(ctx, m, cr)
    if hosp is None:
        return
    a, b = m["window"]
    lead = int(rng.integers(20, 90))
    consult_d = d - TD(days=lead)
    if consult_d < a:
        return
    dx = cdef["dx"][int(rng.integers(len(cdef["dx"])))]
    consult = s.add(kind="OP", member=m["sk"], date=consult_d, provider=hosp, scenario=f"PREOP:{cr}", dx=[dx],
                    setting="HOSPITAL_OP", specialty=spec, level=_level_from(rng, {3: .3, 4: .6, 5: .1}), rtype="IP")
    if cr in ("CR107",):
        consult["imaging"] = ["73562"]
    if cr == "CR102":
        consult["imaging"] = ["76700"]
    if cr == "CR119":
        rad = _radiology_for(ctx, m, ["73562"])
        if not rad:
            return
        consult["imaging"] = ["73562"]
        rad2 = _radiology_for(ctx, m, ["73721"])
        if not rad2:
            return
        s.add(kind="RAD", member=m["sk"], date=consult_d + TD(days=int(rng.integers(5, 15))), parent=consult["eid"],
              provider=rad2, codes=["73721"], dx=["M23.211"], prio=6, needs_prior="73562")
    s.add(kind="IP", member=m["sk"], date=d, cr=cr, prio=1, parent=consult["eid"], provider_hint=hosp, dx_hint=dx)
    if cdef.get("physio") and rng.random() < 0.7:
        los = cdef["los"][1]
        _physio_course(ctx, s, m, consult, d + TD(days=los + 5), ["Z47.1" if False else cdef["dx"][0]])


def _hospital_for(ctx: _Ctx, m: dict, cr: str) -> str | None:
    cdef = ctx.case_rates[cr]
    need_eq = cdef.get("needs")
    home = _home(ctx, m, "HOSP", ptype="HOSPITAL")
    def ok(p):
        info = ctx.providers[p]
        return cdef["specialty"] in info["specialties"] and (not need_eq or need_eq in info["equipment"]) and (
            "ANAESTHESIA" in info["specialties"])
    if home and ok(home):
        return home
    cands = [p for p in _eligible(ctx, m, ptype="HOSPITAL") if ok(p)]
    if not cands:
        cands = [p for p in ctx.providers if ctx.providers[p]["type"] == "HOSPITAL" and ok(p)
                 and ctx.providers[p]["tier"] in PRODUCTS[m["product"]]["tiers"]]
    return _choose(ctx, sorted(cands))


def _pregnancy(ctx: _Ctx, s: _Sched, m: dict) -> None:
    rng = ctx.rng
    a, b = m["window"]
    span = (b - a).days
    if span < 300:
        return
    conception = a + TD(days=int(rng.integers(-60, max(span - 285, 1))))
    delivery = conception + TD(days=int(rng.integers(266, 285)))
    if delivery > b or delivery - TD(days=200) < a:
        return
    ob = _clinic_for(ctx, m, "OBSTETRICS_GYNAECOLOGY")
    if not ob:
        return
    prev = None
    for week, extra in ((8, ["76801"]), (12, []), (20, ["76805"]), (28, []), (32, ["76805"]), (36, [])):
        d = conception + TD(days=7 * week + int(rng.integers(-3, 4)))
        if d < a:
            continue
        ev = s.add(kind="OP", member=m["sk"], date=d, provider=ob, scenario="ANTENATAL", dx=["Z34.90"],
                   setting="CLINIC", specialty="OBSTETRICS_GYNAECOLOGY", level=3 if week > 8 else 4,
                   rtype="IP", root=("PREG", m["sk"], conception))
        if extra and "ULTRASOUND" in ctx.providers[ob]["equipment"]:
            ev["office"] = extra
        elif extra:
            rad = _radiology_for(ctx, m, extra)
            if rad:
                s.add(kind="RAD", member=m["sk"], date=d + TD(days=int(rng.integers(0, 4))), parent=ev["eid"],
                      provider=rad, codes=extra, dx=["Z34.90"], prio=6)
        labs = ["85025"] + (["84702"] if week == 8 else [])
        if week in (8, 28):
            lab = _home(ctx, m, "LAB", ptype="DIAGNOSTIC_LAB")
            if lab and ctx.providers[ob]["type"] != "HOSPITAL":
                s.add(kind="LAB", member=m["sk"], date=d, parent=ev["eid"], provider=lab, codes=labs, dx=["Z34.90"], prio=6)
        if week in (8, 20):
            pharm = _pharmacy_for(ctx, m)
            s.add(kind="RX", member=m["sk"], date=d, parent=ev["eid"], items=["RX1046", "RX1045"] if week == 20 else ["RX1046"],
                  dx=["Z34.90"], provider=pharm, prio=7, course_days=30)
        prev = ev
    cr = "CR105" if rng.random() < 0.3 else "CR104"
    hosp = _hospital_for(ctx, m, cr)
    if hosp:
        s.add(kind="IP", member=m["sk"], date=delivery, cr=cr, prio=1, provider_hint=hosp,
              parent=prev["eid"] if prev else None, rtype="IP", root=("PREG", m["sk"], conception))


# =============================================================================
# the composer: one claim across every table
# =============================================================================


def _new_rows() -> dict[str, list[dict]]:
    return defaultdict(list)


def _emit(ctx: _Ctx, table: str, row: dict) -> None:
    ctx.rows[table].append(row)


def _dose_units(ctx: _Ctx, product: str, member: dict, on: D, days: int | None = None) -> tuple[float, float, int, float]:
    """(dispensed qty, days supply, units/day or ml/day, dose mg/day) for one course of ``product``."""
    d = ctx.drugs[product]
    weight = member["weight"]
    regimen = d["regimen"]
    form = d["form"]
    if form in ("SUSPENSION", "SYRUP"):
        target = {"RX1003": 45, "RX1034": 45, "RX1036": 25}.get(product, 30)
        mg_day = min(target * weight, (d["max_mg_per_kg_day"] or target) * weight * 0.95)
        ml_day = mg_day / d["strength_mg"]
        days = days or (7 if product == "RX1003" else 5)
        qty = math.ceil(ml_day * days / 100.0) * 100
        return float(qty), float(days), ml_day, round(mg_day, 1)
    if form in ("INHALER", "CREAM", "OINTMENT"):
        days = days or 30
        return 1.0, float(min(days, d["max_duration_days"] or 365)), 1.0, round(d["strength_mg"] / days, 3)
    if form == "SACHET":
        return 6.0, 3.0, 2.0, 0.0
    if product == "RX1044":
        return 4.0, 28.0, 1 / 7, round(d["strength_mg"] / 7, 3)
    if product == "RX1060":
        return 24.0, 28.0, 6 / 7, round(15 / 7, 3)
    if regimen is None:
        return 1.0, float(days or 30), 1.0, d["strength_mg"]
    per_day, course = regimen
    days = days or course
    days = min(days, d["max_duration_days"] or 365)
    if d["max_mg_per_kg_day"]:
        while per_day > 1 and per_day * d["strength_mg"] > d["max_mg_per_kg_day"] * weight * 0.95:
            per_day -= 1
    return float(per_day * days), float(days), float(per_day), round(per_day * d["strength_mg"], 3)


def _injectable_units(ctx: _Ctx, product: str, member: dict) -> tuple[float, float]:
    """(billed units, wastage units) for one administration, whole vials."""
    d = ctx.drugs[product]
    w = member["weight"]
    # adult doses, scaled down by weight for children so mg/kg/day stays within drug_policy limits
    dose_mg = {"J1745": 5 * w, "J9355": 6 * w, "J9312": 375 * 1.8, "J0135": 40, "J2506": 6, "J0897": 60,
               "J1650": min(40, 1.0 * w), "J0696": min(1000, 50 * w), "J1100": min(8, 0.3 * w),
               "J1885": min(30, 0.5 * w), "J2405": min(4, 0.15 * w)}[product]
    vial = d["vial_size_mg"] or d["strength_mg"]
    vials = math.ceil(dose_mg / vial - 1e-9)
    billed_mg = vials * vial
    units = billed_mg / d["strength_mg"]
    wastage = (billed_mg - dose_mg) / d["strength_mg"]
    return float(round(units)), float(max(round(wastage), 0))


def _compose(ctx: _Ctx, spec: dict) -> str | None:
    """Build one claim from ``spec`` into ``ctx.rows``. Returns the claim_sk (None if impossible)."""
    rng = ctx.rng
    member = ctx.members[spec["member"]]
    provider = spec["provider"]
    pinfo = ctx.providers[provider]
    date: D = spec["date"]
    claim_type = spec["claim_type"]
    cov = ctx.coverage_at(member["sk"], date) or (member["coverage"][-1] if member["coverage"] else None)
    if cov is None:
        return None
    payer = spec.get("payer") or cov["payer"]
    product = cov["product"]
    tpa = TPA_OF.get((payer, product), "SYN-TPA-1")
    contract = ctx.contracts.get((provider, payer), {"discount": 0.0, "tier": pinfo["tier"]})
    tier = contract["tier"]
    los = int(spec.get("los", 0))
    inpatient = claim_type == "INPATIENT"
    enc_type = spec.get("encounter_type") or {"PHARMACY": "PHARMACY", "LAB": "DIAGNOSTIC", "RADIOLOGY": "DIAGNOSTIC",
                                               "INPATIENT": "INPATIENT"}.get(claim_type, "OUTPATIENT")

    # ---------------------------------------------------------------- attending
    lines_in: list[dict] = spec["lines"]
    attending = spec.get("attending")
    att_spec = spec.get("attending_specialty")
    first_family = ctx.family_of(lines_in[0]["code"]) if lines_in else "EM_OFFICE_EST"
    att_family = spec.get("attending_family") or first_family
    if attending is None:
        attending = ctx.pick_clinician(provider, att_family, date, specialty=att_spec)
        if attending is None and att_spec:
            attending = ctx.pick_clinician(provider, att_family, date)
        if attending is None and spec.get("strict", True):
            return None

    # ---------------------------------------------------------------- timing
    if inpatient:
        adm_min = int(rng.integers(7 * 60, 10 * 60)) if spec.get("admission_type") == "ELECTIVE" else int(rng.integers(0, 24 * 60))
        start_dt = _dt_at(date, adm_min)
        end_dt = _dt_at(date + TD(days=los), int(rng.integers(10 * 60, 15 * 60)))
        if los == 0:
            end_dt = start_dt + TD(minutes=int(rng.integers(240, 420)))
    else:
        total = sum(int(l.get("minutes") or ctx.codes.get(l["code"], {}).get("minutes") or 5) * int(
            l.get("units", 1) if ctx.codes.get(l["code"], {}).get("is_time_based") else 1)
                    for l in lines_in if ctx.family_of(l["code"]) not in ("DRUG_ORAL", "SUPPLY"))
        total = max(total, 10)
        lo, hi = {"PHARMACY": (9 * 60, 22 * 60), "LAB": (7 * 60, 13 * 60), "RADIOLOGY": (8 * 60, 20 * 60)}.get(
            claim_type, (8 * 60, 20 * 60))
        if enc_type == "EMERGENCY":
            lo, hi = 0, 23 * 60
        start_min = None
        if spec.get("start_minute") is not None:
            start_min = int(spec["start_minute"])
            if attending:
                ctx.book(attending, date, start_min, total)
        elif attending and claim_type in ("OUTPATIENT",):
            start_min = ctx.find_slot(attending, date, total, lo, hi)
            if start_min is None:
                if spec.get("strict", True):
                    return None
                start_min = int(rng.integers(lo, hi))
        else:
            start_min = int(rng.integers(lo // 5, hi // 5)) * 5
        start_dt = _dt_at(date, start_min)
        end_dt = start_dt + TD(minutes=total + int(rng.integers(5, 20)))

    claim_sk = spec.get("claim_sk") or ctx.unique(ctx.claim_ids, "CLM", 8)
    digits = claim_sk[3:]
    enc_sk = spec.get("encounter_sk") or f"ENC{digits}"

    # ---------------------------------------------------------------- lines
    line_rows, disp_rows, obs_rows, dev_rows = [], [], [], []
    per_family_clin: dict[str, str] = {}
    cursor = start_dt
    lab_lines, img_lines = [], []
    for i, L in enumerate(lines_in, start=1):
        code = L["code"]
        fam = ctx.family_of(code)
        sdate = L.get("service_date") or (date + TD(days=int(L.get("day", 0))))
        units = float(L.get("units", 1))
        clin = L.get("clinician")
        if clin is None:
            if fam in ("DRUG_ORAL", "SUPPLY") or (fam in ("DRUG_INJECTABLE", "DME") and pinfo["type"] == "PHARMACY"):
                role_family = "DRUG_ORAL" if fam != "DME" else "DME"
                clin = per_family_clin.get(("PHARM", sdate)) or ctx.pick_clinician(provider, role_family, sdate, "PHARMACY")
                per_family_clin[("PHARM", sdate)] = clin
            elif fam in ("ANAESTHESIA", "SEDATION"):
                clin = per_family_clin.get(("ANAES", sdate)) or ctx.pick_clinician(provider, fam, sdate, "ANAESTHESIA")
                per_family_clin[("ANAES", sdate)] = clin
            elif fam.startswith("LAB_") and fam != "LAB_POC" or (fam == "SPECIMEN" and pinfo["type"] == "DIAGNOSTIC_LAB") \
                    or (fam == "LAB_POC" and pinfo["type"] == "DIAGNOSTIC_LAB"):
                clin = per_family_clin.get(("PATH", sdate)) or ctx.pick_clinician(provider, fam, sdate, "PATHOLOGY")
                per_family_clin[("PATH", sdate)] = clin
            elif fam.startswith("IMAGING_"):
                if attending and fam in ctx.clin_by_id.get(attending, {}).get("privileges", set()) and pinfo["type"] == "CLINIC":
                    clin = attending
                else:
                    clin = ctx.pick_clinician(provider, fam, sdate, "RADIOLOGY") or ctx.pick_clinician(provider, fam, sdate)
            elif fam == "PHYSIO":
                clin = per_family_clin.get(("PHYSIO", sdate)) or ctx.pick_clinician(provider, fam, sdate, "PHYSIOTHERAPY")
                per_family_clin[("PHYSIO", sdate)] = clin
            else:
                clin = ctx.pick_clinician(provider, fam, sdate, prefer=attending) if attending else \
                    ctx.pick_clinician(provider, fam, sdate)
                if clin is None and fam in ("DRUG_INJECTABLE",):
                    clin = ctx.pick_clinician(provider, fam, sdate)
        if clin is None and spec.get("strict", True):
            return None
        # times
        cinfo = ctx.codes.get(code, {})
        minutes = int(L.get("minutes") or cinfo.get("minutes") or 0)
        if cinfo.get("is_time_based"):
            minutes = minutes * int(units)
        if L.get("start") is not None:
            l_start = L["start"]
        elif inpatient:
            if sdate == date:
                l_start = start_dt + TD(minutes=int(rng.integers(30, 240))) if fam not in ("CASE_RATE", "ROOM_DAY") else start_dt
            else:
                l_start = _dt_at(sdate, int(rng.integers(7 * 60, 18 * 60)) if fam != "ROOM_DAY" else 0)
        else:
            l_start = cursor
        l_end = l_start + TD(minutes=minutes) if minutes else l_start
        if not inpatient and fam not in ("DRUG_ORAL", "SUPPLY") and minutes:
            cursor = l_end
        if clin and minutes and (inpatient or fam.startswith("IMAGING_") or fam in ("PHYSIO", "PSYCHOTHERAPY", "ECHO",
                                                                                    "ENDOSCOPY", "NEURODIAG")):
            if clin != attending or inpatient:
                s_min = l_start.hour * 60 + l_start.minute
                booked = ctx.book(clin, l_start.date(), s_min, minutes)
                tries = 0
                while not booked and tries < 30:
                    l_start += TD(minutes=15)
                    l_end += TD(minutes=15)
                    s_min = l_start.hour * 60 + l_start.minute
                    if l_start.date() != sdate:
                        break
                    booked = ctx.book(clin, l_start.date(), s_min, minutes)
                    tries += 1
        # equipment capacity
        eq = EQUIP_OF_CODE.get(code)
        if eq and eq in pinfo["equipment"]:
            ctx.equip_use[(provider, eq, sdate)] += 1
        # price
        zero = bool(L.get("zero"))
        if L.get("unit_price") is not None:
            unit_price = float(L["unit_price"])
        elif zero:
            unit_price = 0.0
        else:
            allowed = ctx.allowed_price(code, tier, payer, sdate)
            unit_price = round(allowed * (1 - contract["discount"]), 2)
        gross = round(float(L.get("gross", unit_price * units)), 2)
        list_amount = 0.0 if zero else round(ctx.allowed_price(code, tier, payer, sdate) * units, 2)
        bfam = spec.get("benefit_family") if inpatient else ctx.service_family(code)
        if inpatient and not bfam:
            bfam = "INPATIENT"
        pct = ctx.share_pct(product, bfam, sdate)
        pshare = round(gross * pct, 2)
        line_sk = f"{claim_sk}-L{i:02d}"
        indicator = L.get("indicator")
        row = {
            "line_sk": line_sk, "claim_sk": claim_sk, "activity_type": ctx.activity_type(code), "activity_code": code,
            "units": units, "gross_amount": gross, "net_amount": round(gross - pshare, 2), "patient_share": pshare,
            "rendering_clinician_id": clin, "ordering_clinician_id": L.get("ordering", spec.get("ordering")),
            "indicator": indicator, "authorization_id": None, "service_date": sdate, "tenant_id": TENANT,
            "service_start_time": l_start, "service_end_time": l_end,
            "performing_entity_id": provider, "wastage_units": float(L.get("wastage", 0.0)) if ctx.is_drug(code) else None,
            "unit_price": unit_price, "device_serial": None, "product": code if ctx.is_drug(code) else None,
            "activity_description": ctx.description(code),
            "_bfam": bfam, "_list": list_amount,
        }
        if fam in ("IMPLANT", "DME") or L.get("device"):
            serial = ctx.unique(ctx.serials, "SN-", 10)
            row["device_serial"] = serial
            dev_rows.append({"serial_number": serial, "device_code": code, "provider_sk": provider, "condition": "NEW",
                             "acquisition": "PURCHASE",
                             "useful_life_days": {"C1776": 7300, "C1874": 7300, "V2632": 9125, "C1781": 7300,
                                                  "E0114": 730, "L1832": 365, "E0570": 1825, "E0601": 1825}.get(code, 1095),
                             "claim_sk": claim_sk, "line_sk": line_sk, "member_sk": member["sk"], "issued_date": sdate,
                             "tenant_id": TENANT})
        line_rows.append(row)
        if ctx.is_drug(code):
            rx = L.get("rx") or {}
            d = ctx.drugs[code]
            disp_rows.append({
                "rx_sk": f"RXD{digits}-{i:02d}", "claim_sk": claim_sk,
                "prescriber_id": rx.get("prescriber", L.get("ordering", spec.get("ordering")) or clin),
                "pharmacy_id": provider, "prescribed_product": rx.get("prescribed_product", code),
                "billed_product": code, "dispensed_product": rx.get("dispensed_product", code),
                "prescribed_qty": float(rx.get("prescribed_qty", units)), "dispensed_qty": float(rx.get("dispensed_qty", units)),
                "days_supply": float(rx.get("days_supply", 1)), "fill_date": sdate, "tenant_id": TENANT,
                "member_sk": member["sk"], "line_sk": line_sk, "authorization_id": None,
                "prescribed_date": rx.get("prescribed_date", date), "billed_qty": units, "billed_amount": gross,
                "strength_mg": d["strength_mg"], "form": d["form"], "dose_mg_per_day": rx.get("dose_mg_per_day"),
            })
        if fam.startswith("LAB_") or code == "36415":
            if code != "36415":
                lab_lines.append(row)
        if fam.startswith("IMAGING_"):
            img_lines.append(row)

    # ---------------------------------------------------------------- diagnoses
    dx_codes = list(dict.fromkeys(spec["dx"]))
    poa = spec.get("poa", {})
    dx_rows = []
    for seq, code in enumerate(dx_codes, start=1):
        desc, chapter = ctx.icd.get(code, (code, ref.icd_chapter(code)))
        dx_rows.append({"claim_sk": claim_sk, "encounter_sk": enc_sk, "code": code, "code_system": "ICD-10-CM",
                        "code_version": "ICD-10-CM FY2025" if date >= D(2024, 10, 1) else "ICD-10-CM FY2024",
                        "diagnosis_type": "PRINCIPAL" if seq == 1 else "SECONDARY", "sequence": seq,
                        "present_on_admission": poa.get(code, "Y") if inpatient else None,
                        "description": desc, "chapter": chapter, "tenant_id": TENANT})

    # ---------------------------------------------------------------- observations
    lab_doc = img_doc = None
    if lab_lines:
        lab_doc = f"SYNTHETIC_lab_report_{claim_sk}_en.txt"
    if img_lines:
        img_doc = f"SYNTHETIC_imaging_report_{claim_sk}_en.txt"
    lab_results = []
    for row in lab_lines:
        comps = ctx.world.context["panels"].get(row["activity_code"], [row["activity_code"]])
        t = row["service_start_time"] + TD(minutes=int(rng.integers(90, 1200)))
        for comp in comps:
            value, unit = _lab_value(rng, comp, dx_codes[0])
            obs_rows.append({"observation_sk": ctx.world.new_id("OBS", 8), "line_sk": row["line_sk"],
                             "observation_type": "RESULT", "value": value, "unit": unit, "attachment_ref": lab_doc,
                             "event_time": t, "tenant_id": TENANT, "observation_code": comp, "claim_sk": claim_sk,
                             "member_sk": member["sk"]})
            lab_results.append((comp, ctx.description(comp), value, unit))
    for row in img_lines:
        t = row["service_end_time"] + TD(minutes=int(rng.integers(30, 1440)))
        obs_rows.append({"observation_sk": ctx.world.new_id("OBS", 8), "line_sk": row["line_sk"],
                         "observation_type": "REPORT", "value": _impression(ctx, row["activity_code"], dx_codes[0]),
                         "unit": None, "attachment_ref": img_doc, "event_time": t, "tenant_id": TENANT,
                         "observation_code": row["activity_code"], "claim_sk": claim_sk, "member_sk": member["sk"]})
    if inpatient:
        room = next((r for r in line_rows if ctx.family_of(r["activity_code"]) == "ROOM_DAY"), line_rows[0])
        for kind, t in (("ADMIT", start_dt), ("DISCHARGE", end_dt)):
            obs_rows.append({"observation_sk": ctx.world.new_id("OBS", 8), "line_sk": room["line_sk"],
                             "observation_type": "ENCOUNTER_EVENT", "value": t.isoformat(sep=" "), "unit": None,
                             "attachment_ref": None, "event_time": t, "tenant_id": TENANT, "observation_code": kind,
                             "claim_sk": claim_sk, "member_sk": member["sk"]})

    # ---------------------------------------------------------------- authorisation
    auth_rows, auth_line_rows = [], []
    if spec.get("auth", "auto") is not False:
        need = [r for r in line_rows if inpatient or ctx.needs_auth(product, r["_bfam"], r["service_date"])]
        if need:
            pa = spec.get("authorization_sk") or ctx.unique(ctx.auth_ids, "PA", 8)
            elective = spec.get("admission_type") != "EMERGENCY"
            if inpatient and not elective:
                vf = _dt_at(date, 0)
            else:
                vf = _dt_at(date - TD(days=int(rng.integers(1, 10))), int(rng.integers(8 * 60, 17 * 60)))
            vt = _dt_at(date + TD(days=max(los, 0) + 30), 23 * 60 + 59)
            total = sum(r["gross_amount"] for r in need)
            auth_rows.append({"authorization_sk": pa, "request_id": f"REQ{pa[2:]}", "response_id": f"RSP{pa[2:]}",
                              "status": "APPROVED", "valid_from": vf, "valid_to": vt, "provider_sk": provider,
                              "facility_id": provider, "approved_amount": round(total * float(rng.uniform(1.05, 1.25)), 2),
                              "member_sk": member["sk"], "tenant_id": TENANT})
            by_code: dict[str, list[dict]] = defaultdict(list)
            for r in need:
                r["authorization_id"] = pa
                by_code[r["activity_code"]].append(r)
            for j, (code, rs) in enumerate(sorted(by_code.items()), start=1):
                units = sum(r["units"] for r in rs)
                value = sum(r["gross_amount"] for r in rs)
                extra = 1 if ctx.family_of(code) == "ROOM_DAY" else 0
                auth_line_rows.append({"authorization_line_sk": f"{pa}-{j:02d}", "authorization_sk": pa,
                                       "activity_code": code, "approved_units": units + extra,
                                       "approved_value": round(value * 1.1 + (value / max(units, 1)) * extra, 2),
                                       "conditions": "Valid for the named facility and member only",
                                       "denial_code": None, "tenant_id": TENANT})
            for dr in disp_rows:
                lr = next(r for r in line_rows if r["line_sk"] == dr["line_sk"])
                dr["authorization_id"] = lr["authorization_id"]

    # ---------------------------------------------------------------- header
    gross = round(sum(r["gross_amount"] for r in line_rows), 2)
    pshare = round(sum(r["patient_share"] for r in line_rows), 2)
    net = round(gross - pshare, 2)
    discount = round(sum(r["_list"] for r in line_rows) - gross, 2)
    last_service = max(r["service_date"] for r in line_rows)
    discharge = date + TD(days=los) if inpatient else last_service
    lag = spec.get("submission_lag")
    if lag is None:
        lag = int(rng.integers(1, 21)) if not inpatient else int(rng.integers(3, 25))
    submission = discharge + TD(days=int(lag))
    token = hashlib.sha256(f"{member['sk']}|{provider}|{date.isoformat()}|{claim_sk}".encode()).hexdigest()[:20]
    header = {
        "claim_sk": claim_sk, "source_claim_id": f"{provider}-{digits}", "tenant_id": TENANT,
        "member_sk": member["sk"], "provider_sk": provider, "payer_id": payer, "tpa": tpa, "encounter_sk": enc_sk,
        "submission_date": submission, "settlement_date": None, "gross_amount": gross, "net_amount": net,
        "patient_share": pshare, "source_currency": "AED", "is_cashless": bool(spec.get("cashless", rng.random() > 0.02)),
        "claim_type": claim_type, "accident_indicator": bool(spec.get("accident", False)), "discount": max(discount, 0.0),
        "cross_payer_match_token": f"XPM-{token}",
    }

    # ---------------------------------------------------------------- encounter
    encounter = {
        "encounter_sk": enc_sk, "claim_sk": claim_sk, "encounter_type": enc_type, "facility_id": provider,
        "start_time": start_dt, "end_time": end_dt, "admission_date": date if inpatient else None,
        "discharge_date": discharge if inpatient else None, "length_of_stay_days": los if inpatient else 0,
        "location": f"{pinfo['emirate']} / " + {"INPATIENT": "Ward", "DAY_CASE": "Day-care unit", "EMERGENCY": "Emergency department",
                                                 "PHARMACY": "Pharmacy counter", "DIAGNOSTIC": "Diagnostic suite",
                                                 "TELEHEALTH": "Virtual"}.get(enc_type, "Outpatient clinic"),
        "tenant_id": TENANT, "member_sk": member["sk"], "bed_id": spec.get("bed_id"),
        "duration_minutes": round((end_dt - start_dt).total_seconds() / 60.0, 1),
        "observation_status": spec.get("observation_status"), "admission_type": spec.get("admission_type"),
        "discharge_type": spec.get("discharge_type", "HOME") if inpatient else None,
    }

    # ---------------------------------------------------------------- remittance + adjudication
    decision = spec.get("decision")
    priced = [r for r in line_rows if r["net_amount"] > 0]
    if decision is None:
        u = rng.random()
        decision = "PAID" if u < 0.88 or not priced else ("PARTIAL" if u < 0.95 else "DENIED")
        if decision == "PARTIAL" and len(priced) < 2:
            decision = "PAID"
    settle = submission + TD(days=int(rng.integers(15, 46)))
    header["settlement_date"] = settle
    denial = spec.get("denial_code") or (DENIAL_CODES[int(rng.integers(len(DENIAL_CODES)))] if decision != "PAID" else None)
    denied_lines: set[str] = set()
    if decision == "DENIED":
        denied_lines = {r["line_sk"] for r in line_rows}
    elif decision == "PARTIAL":
        k = max(1, int(round(len(priced) * float(rng.uniform(0.15, 0.4)))))
        denied_lines = {r["line_sk"] for r in rng.permutation(np.array(priced, dtype=object))[:k]}
    pay_ref = None if decision == "DENIED" else ctx.unique(ctx.pay_refs, f"PAY-{payer[-1]}-", 9)
    rem_rows = []
    paid_total = 0.0
    for j, r in enumerate(line_rows, start=1):
        if r["line_sk"] in denied_lines:
            dec, pay, adj, dcode = "DENIED", 0.0, r["net_amount"], denial
        else:
            dec, pay, adj, dcode = "PAID", r["net_amount"], 0.0, None
        paid_total += pay
        rem_rows.append({"remittance_sk": f"RA{digits}-{j:02d}", "claim_sk": claim_sk, "line_sk": r["line_sk"],
                         "decision": dec, "denial_code": dcode, "adjustment": round(adj, 2), "payment_amount": round(pay, 2),
                         "payment_reference": pay_ref if pay > 0 or dec == "PAID" else None, "settlement_date": settle,
                         "tenant_id": TENANT})
    adj_rows = _adjudication(ctx, claim_sk, net, round(paid_total, 2), decision, denial, submission, settle, inpatient)

    # ---------------------------------------------------------------- documents
    doc_rows = _documents(ctx, spec, claim_sk, member, provider, date, discharge, los, dx_codes, line_rows,
                          lab_doc, lab_results, img_doc, img_lines, attending, start_dt, gross)

    # ---------------------------------------------------------------- referral
    ref_row = None
    if spec.get("referral"):
        rf = spec["referral"]
        ref_row = {"referral_sk": ctx.world.new_id("REF", 7), "referrer_id": rf["from_clin"],
                   "recipient_id": attending or (line_rows[0]["rendering_clinician_id"] if line_rows else None),
                   "referrer_provider_sk": rf["from_provider"], "recipient_provider_sk": provider,
                   "member_sk": member["sk"], "referral_date": rf["date"], "direction": "OUTBOUND",
                   "specialty": rf.get("specialty") or (ctx.clin_by_id[attending]["specialty"] if attending in ctx.clin_by_id else None),
                   "reason_code": dx_codes[0], "resulting_claim_sk": claim_sk, "tenant_id": TENANT}

    # ---------------------------------------------------------------- emit
    for r in line_rows:
        r.pop("_bfam", None)
        r.pop("_list", None)
    _emit(ctx, "claim_header", header)
    for r in line_rows:
        _emit(ctx, "claim_line", r)
    for r in dx_rows:
        _emit(ctx, "diagnosis", r)
    if spec.get("encounter_sk") is None:
        _emit(ctx, "encounter", encounter)
    for r in obs_rows:
        _emit(ctx, "observation", r)
    for r in auth_rows:
        _emit(ctx, "authorization", r)
    for r in auth_line_rows:
        _emit(ctx, "authorization_line", r)
    for r in disp_rows:
        _emit(ctx, "prescription_dispense", r)
    for r in dev_rows:
        _emit(ctx, "device_inventory", r)
    for r in rem_rows:
        _emit(ctx, "remittance", r)
    for r in adj_rows:
        _emit(ctx, "adjudication_event", r)
    for r in doc_rows:
        _emit(ctx, "document", r)
    if ref_row:
        _emit(ctx, "referral", ref_row)
    _emit(ctx, "claim_version", {"claim_sk": claim_sk, "version_no": int(spec.get("version_no", 1)),
                                 "relationship": spec.get("relationship", "ORIGINAL"),
                                 "prior_claim_sk": spec.get("prior_claim_sk"),
                                 "changed_fields": json.dumps(spec.get("changed_fields", {}), sort_keys=True),
                                 "recorded_at": _dt_at(submission, int(rng.integers(0, 24 * 60))),
                                 "resubmission_type": spec.get("resubmission_type"), "tenant_id": TENANT})
    # pharmacy inventory bookkeeping (make_claim keeps inventory consistent)
    for dr in disp_rows:
        ctx.dispensed[(provider, dr["billed_product"], dr["fill_date"].strftime("%Y-%m"))] += dr["dispensed_qty"]
    ctx.last_claim = dict(header=header, lines=line_rows, attending=attending, decision=decision, denial=denial,
                          denied_lines=denied_lines, dx=dx_codes, encounter=encounter, settle=settle, spec=spec)
    return claim_sk


def _lab_value(rng: np.random.Generator, code: str, principal: str) -> tuple[str, str]:
    mean, sd, unit = LAB_VALUES.get(code, (1.0, 0.2, "units"))
    if sd == 0:
        return unit, ""
    if code == "83036" and not principal.startswith("E11"):
        mean, sd = 5.4, 0.3
    value = max(float(rng.normal(mean, sd)), mean * 0.2)
    return (f"{value:.2f}" if value < 20 else f"{value:.0f}"), unit


def _impression(ctx: _Ctx, code: str, principal: str) -> str:
    desc = ctx.description(code)
    dx = ctx.icd.get(principal, (principal,))[0]
    return f"{desc}: findings consistent with the clinical history ({dx.lower()}); no acute unexpected abnormality."


def _adjudication(ctx: _Ctx, claim_sk: str, net: float, paid: float, decision: str, denial: str | None,
                  submission: D, settle: D, inpatient: bool) -> list[dict]:
    rng = ctx.rng
    rows = []
    manual = decision != "PAID" or inpatient and rng.random() < 0.4 or net > 8000 and rng.random() < 0.3
    fail = manual and decision == "PAID" and rng.random() < 0.15
    t0 = _dt_at(submission, int(rng.integers(60, 5 * 60)))
    rows.append({"event_sk": ctx.world.new_id("AEV", 8), "claim_sk": claim_sk, "actor": "SYSTEM",
                 "actor_role": "AUTO_ADJUDICATION_ENGINE", "event_type": "AUTO_EDIT",
                 "decision": "PEND" if manual else "APPROVE", "override_reason": None, "amount_before": net,
                 "amount_after": net if not manual else net, "event_time": t0,
                 "system_edit_result": "FAIL" if fail else ("PEND" if manual else "PASS"), "tenant_id": TENANT})
    if manual:
        adj = ctx.adjudicators[int(rng.integers(len(ctx.adjudicators)))]
        day = _weekday_shift(submission + TD(days=int(rng.integers(2, max((settle - submission).days - 3, 3)))))
        if day >= settle:
            day = settle - TD(days=1)
        t1 = _dt_at(day, int(rng.integers(8 * 60, 17 * 60)))
        reason = None
        if fail:
            reason = _pick(rng, ["Authorisation confirmed with the provider; edit was a false positive.",
                                 "Supporting documents reviewed; service medically necessary.",
                                 "Duplicate edit triggered by a resubmission of a corrected line; not a duplicate."])
        rows.append({"event_sk": ctx.world.new_id("AEV", 8), "claim_sk": claim_sk, "actor": adj,
                     "actor_role": "CLAIMS_ADJUDICATOR", "event_type": "MANUAL_REVIEW",
                     "decision": {"PAID": "APPROVE", "PARTIAL": "PARTIAL_APPROVE", "DENIED": "DENY"}[decision],
                     "override_reason": reason, "amount_before": net, "amount_after": paid, "event_time": t1,
                     "system_edit_result": "OVERRIDDEN" if fail else ("DENIAL:" + denial if denial else "REVIEWED"),
                     "tenant_id": TENANT})
    else:
        rows.append({"event_sk": ctx.world.new_id("AEV", 8), "claim_sk": claim_sk, "actor": "SYSTEM",
                     "actor_role": "AUTO_ADJUDICATION_ENGINE", "event_type": "AUTO_ADJUDICATION", "decision": "APPROVE",
                     "override_reason": None, "amount_before": net, "amount_after": paid,
                     "event_time": t0 + TD(minutes=int(rng.integers(1, 30))), "system_edit_result": "PASS",
                     "tenant_id": TENANT})
    return rows


# ---------------------------------------------------------------------------- documents

_FOLLOW_UP = ["Follow up in the outpatient clinic in one to two weeks.",
              "Review with the family physician within ten days.",
              "Return earlier if symptoms recur or worsen.",
              "Outpatient review arranged with the treating specialist."]


def _doc(ctx: _Ctx, claim_sk: str, member: dict, provider: str, doc_type: str, language: str, text: str,
         created: DT, attachment: str) -> dict:
    ocr = float(ctx.rng.uniform(0.9, 0.995)) if ctx.rng.random() > 0.08 else float(ctx.rng.uniform(0.72, 0.9))
    return {"document_sk": ctx.world.new_id("DOC", 7), "claim_sk": claim_sk, "member_sk": member["sk"],
            "provider_sk": provider, "doc_type": doc_type, "language": language, "text": text,
            "ocr_confidence": round(ocr, 3), "created_at": created, "version_no": 1, "prior_document_sk": None,
            "attachment_ref": attachment, "is_synthetic": True, "tenant_id": TENANT}


def _documents(ctx: _Ctx, spec: dict, claim_sk: str, member: dict, provider: str, date: D, discharge: D, los: int,
               dx_codes: list[str], lines: list[dict], lab_doc, lab_results, img_doc, img_lines, attending,
               start_dt: DT, gross: float) -> list[dict]:
    if spec.get("documents", True) is False:
        return []
    rng = ctx.rng
    out = []
    head = "*** SYNTHETIC DOCUMENT — GENERATED FOR SOFTWARE TESTING. NOT A REAL CLINICAL RECORD. ***\n"
    foot = "\n--- END OF SYNTHETIC DOCUMENT --- Every fact in this file is invented.\n"
    dx_desc = ctx.icd.get(dx_codes[0], (dx_codes[0],))[0]
    terms = []
    for r in lines:
        terms += ctx.world.context["medical_necessity"].get(r["activity_code"], [])
    terms = list(dict.fromkeys(terms))
    if claim_sk and spec["claim_type"] == "INPATIENT":
        from fwa.nlp.synthetic import render_discharge_summary

        language = "ar" if rng.random() < 0.3 else "en"
        procs = [r for r in lines if ctx.family_of(r["activity_code"]) not in ("ROOM_DAY", "CASE_RATE", "DRUG_ORAL")
                 and ctx.codes.get(r["activity_code"], {}).get("service_family") in ("INPATIENT", "DAY_SURGERY", "MATERNITY")
                 and ctx.family_of(r["activity_code"]).startswith(("SURG", "OBSTETRIC", "CARDIAC"))]
        drugs = sorted({ctx.description(r["activity_code"]) for r in lines if ctx.is_drug(r["activity_code"])})
        secondaries = [f"{c} {ctx.icd.get(c, (c,))[0]}" for c in dx_codes[1:]]
        vitals = (f"Vital signs on admission: BP {int(rng.integers(105, 150))}/{int(rng.integers(60, 95))} mmHg, "
                  f"HR {int(rng.integers(62, 112))}/min, T {rng.uniform(36.4, 38.6):.1f} C, SpO2 {int(rng.integers(93, 100))}%.")
        addendum = "\n".join([
            "PROCEDURES: " + ("; ".join(f"{r['activity_code']} {ctx.description(r['activity_code'])} on {r['service_date'].isoformat()}"
                                        for r in procs) or "None (medical management)."),
            "SECONDARY DIAGNOSES: " + ("; ".join(secondaries) or "None recorded."),
            "DISCHARGE MEDICATIONS: " + ("; ".join(drugs) or "None."),
            vitals,
            ("INDICATION / NECESSITY: " + "; ".join(terms) + ".") if terms else "",
            _pick(rng, _FOLLOW_UP),
            f"Attending clinician: {attending}.",
        ])
        pharm = sum(r["gross_amount"] for r in lines if ctx.is_drug(r["activity_code"]))
        fields = {"provider": provider, "member": member["sk"], "claim": claim_sk, "admit": date.isoformat(),
                  "discharge": discharge.isoformat(), "los": los, "los_text": los, "dx_code": dx_codes[0],
                  "dx_text": dx_desc, "pharm": f"{(pharm / gross if gross else 0):.0%}",
                  "extra": _pick(rng, ["A routine review was documented on each day of the admission.",
                                       "Vital signs remained within expected limits throughout.",
                                       "The treating team documented daily progress notes.",
                                       "No transfer to a higher level of care was required.",
                                       "Discharge planning commenced on the second day of admission."])}
        text = render_discharge_summary(fields, language, addendum)
        out.append(_doc(ctx, claim_sk, member, provider, "DISCHARGE_SUMMARY", language, text,
                        _dt_at(discharge, int(rng.integers(9 * 60, 16 * 60))),
                        f"SYNTHETIC_discharge_{claim_sk}_{language}.txt"))
        if procs:
            p = procs[0]
            op = (head + "OPERATIVE NOTE (SYNTHETIC)\n"
                  f"Facility: {provider}  Patient: {member['sk']}  Claim: {claim_sk}\n"
                  f"Date: {p['service_date'].isoformat()}  Surgeon: {p['rendering_clinician_id']}\n"
                  f"Pre-operative diagnosis: {dx_codes[0]} {dx_desc}\n"
                  f"Procedure: {p['activity_code']} {ctx.description(p['activity_code'])}\n"
                  + (f"Indication: {'; '.join(terms)}.\n" if terms else "")
                  + f"Anaesthesia time: {int(rng.integers(45, 180))} minutes. Estimated blood loss "
                    f"{int(rng.integers(10, 300))} ml. No intra-operative complication.\n" + foot)
            out.append(_doc(ctx, claim_sk, member, provider, "OPERATIVE_NOTE", "en", op,
                            p["service_start_time"] + TD(hours=3), f"SYNTHETIC_operative_note_{claim_sk}_en.txt"))
    if lab_doc:
        body = "\n".join(f"{c:<7} {desc[:48]:<48} {v} {u}" for c, desc, v, u in lab_results)
        text = (head + "LABORATORY REPORT (SYNTHETIC)\n"
                f"Laboratory: {provider}  Patient: {member['sk']}  Claim: {claim_sk}\n"
                f"Clinical information: {dx_codes[0]} {dx_desc}\nCollected: {start_dt.isoformat(sep=' ')}\n\n{body}\n" + foot)
        out.append(_doc(ctx, claim_sk, member, provider, "LAB_REPORT", "en", text,
                        start_dt + TD(hours=int(rng.integers(3, 30))), lab_doc))
    if img_doc:
        body = "\n".join(f"Examination: {r['activity_code']} {ctx.description(r['activity_code'])}\n"
                         f"Impression: {_impression(ctx, r['activity_code'], dx_codes[0])}" for r in img_lines)
        text = (head + "IMAGING REPORT (SYNTHETIC)\n"
                f"Facility: {provider}  Patient: {member['sk']}  Claim: {claim_sk}\n"
                f"Clinical indication: {dx_codes[0]} {dx_desc}\n{body}\n"
                f"Reporting radiologist: {img_lines[0]['rendering_clinician_id']}\n" + foot)
        out.append(_doc(ctx, claim_sk, member, provider, "IMAGING_REPORT", "en", text,
                        img_lines[0]["service_end_time"] + TD(hours=int(rng.integers(1, 30))), img_doc))
    fams = {ctx.service_family(r["activity_code"]) for r in lines}
    if spec["claim_type"] != "INPATIENT" and (terms or fams & {"ADVANCED_IMAGING", "SPECIALTY_DRUG"}):
        orderer = spec.get("ordering") or attending
        text = (head + "CLINICAL NOTE (SYNTHETIC)\n"
                f"Patient: {member['sk']}  Claim: {claim_sk}  Requesting clinician: {orderer}\n"
                f"Diagnosis: {dx_codes[0]} {dx_desc}\n"
                f"Requested: {', '.join(r['activity_code'] for r in lines if ctx.service_family(r['activity_code']) in ('ADVANCED_IMAGING', 'SPECIALTY_DRUG') or r['activity_code'] in ctx.world.context['medical_necessity'])}\n"
                + (f"Clinical justification: {'; '.join(terms)}.\n" if terms else "Clinical justification: see history.\n")
                + foot)
        out.append(_doc(ctx, claim_sk, member, provider, "CLINICAL_NOTE", "en", text,
                        _dt_at(date - TD(days=int(rng.integers(0, 5))), int(rng.integers(8 * 60, 18 * 60))),
                        f"SYNTHETIC_clinical_note_{claim_sk}_en.txt"))
    if spec.get("referral") and spec["referral"].get("letter"):
        rf = spec["referral"]
        text = (head + "REFERRAL LETTER (SYNTHETIC)\n"
                f"From: {rf['from_clin']} ({rf['from_provider']})  To: {provider}\n"
                f"Patient: {member['sk']}  Date: {rf['date'].isoformat()}\n"
                f"Reason for referral: {dx_codes[0]} {dx_desc}. Kindly assess and advise.\n" + foot)
        out.append(_doc(ctx, claim_sk, member, provider, "REFERRAL_LETTER", "en", text,
                        _dt_at(rf["date"], int(rng.integers(8 * 60, 18 * 60))),
                        f"SYNTHETIC_referral_letter_{claim_sk}_en.txt"))
    return out


# =============================================================================
# events → claim specs
# =============================================================================


class _History:
    """Per-member clinical history used to keep the base clean."""

    def __init__(self) -> None:
        self.code_last: dict[tuple[str, str], D] = {}
        self.group_fill: dict[tuple[str, str], tuple[D, float]] = {}
        self.ip: dict[str, list[tuple[D, D]]] = defaultdict(list)
        self.day_claims: set[tuple[str, D, str]] = set()
        self.member_day_kind: set[tuple[str, D, str]] = set()


def _em_code(ctx: _Ctx, member: str, provider: str, level: int, setting: str) -> str:
    level = int(min(max(level or 3, 1), 5))
    if setting == "ED":
        return f"9928{level}"
    if setting == "REFERRAL" and (member, provider) not in ctx.seen_mp:
        return f"9924{max(level, 2)}"
    if (member, provider) not in ctx.seen_mp:
        return f"9920{max(level, 2)}"
    return f"9921{level}"


def _repeat_ok(ctx: _Ctx, hist: _History, member: str, code: str, d: D) -> bool:
    gap = ctx.world.context["repeat_interval"].get(code)
    last = hist.code_last.get((member, code))
    if last is not None and last == d:
        return False  # the same service already billed for this member today
    if gap and last is not None and (d - last).days < gap:
        return False
    return True


def _drug_item(ctx: _Ctx, hist: _History, m: dict, product: str, d: D, prescriber: str | None, prescribed: D,
               course_days: int | None = None) -> dict | None:
    """A validated dispensing line (step therapy, overlap, dose) or None."""
    if product not in ctx.drugs:  # a supply (test strips): one pack per 30 days
        last = hist.group_fill.get((m["sk"], product))
        if last is not None and (d - last[0]).days < 0.85 * last[1]:
            return None
        days = float(course_days or 30)
        hist.group_fill[(m["sk"], product)] = (d, days)
        return {"code": product, "units": float(max(1, round(days / 30))), "ordering": prescriber}
    drug = ctx.drugs[product]
    step = ctx.world.context["step_therapy"].get(product)
    if step:
        group, lookback = step
        last = hist.group_fill.get((m["sk"], group))
        if last is None or (d - last[0]).days > lookback or last[0] >= d:
            first_line = {"METFORMIN": "RX1011", "ATORVASTATIN": "RX1017", "METHOTREXATE": "RX1060"}[group]
            product = first_line
            drug = ctx.drugs[product]
    if product.startswith("J"):
        units, wastage = _injectable_units(ctx, product, m)
        qty, days, dose = units, float(drug["max_duration_days"] or 30), None
    else:
        qty, days, _, dose = _dose_units(ctx, product, m, d, course_days)
        wastage = 0.0
    group = drug["equivalence_group"]
    last = hist.group_fill.get((m["sk"], group))
    if last is not None and (d - last[0]).days < 0.85 * last[1]:
        return None
    hist.group_fill[(m["sk"], group)] = (d, days)
    return {"code": product, "units": qty, "wastage": wastage, "ordering": prescriber,
            "rx": {"prescriber": prescriber, "prescribed_date": prescribed, "days_supply": days,
                   "prescribed_qty": qty, "dispensed_qty": qty, "dose_mg_per_day": dose}}


def _ip_blocked(hist: _History, member: str, d: D, pad_after: int = 0) -> bool:
    for a, b in hist.ip[member]:
        if a <= d <= b + TD(days=pad_after):
            return True
    return False


def _build_claims(ctx: _Ctx, sched: _Sched, target: int) -> None:
    rng = ctx.rng
    events = sched.events
    keep = _keep_probabilities(events, target)
    ctx.keep_probabilities = keep
    roots = sorted({(e["rtype"], str(e["root"])) for e in events})
    kept_roots = {r for r in roots if rng.random() < keep[r[0]]}
    events = [e for e in events if (e["rtype"], str(e["root"])) in kept_roots]
    events.sort(key=lambda e: (e["date"], e["prio"], e["member"], e["eid"]))
    hist = _History()
    done: dict[int, dict] = {}
    stats = ctx.stats
    for ev in events:
        m = ctx.members[ev["member"]]
        d = ev["date"]
        kind = ev["kind"]
        stats[(kind, "scheduled")] += 1
        if not _in_window(m, d) or ctx.coverage_at(m["sk"], d) is None:
            stats[(kind, "no_cover")] += 1
            continue
        if ev.get("parent") and ev["parent"] not in done:
            stats[(kind, "no_parent")] += 1
            continue
        if kind != "IP" and _ip_blocked(hist, m["sk"], d):
            stats[(kind, "ip_block")] += 1
            continue
        spec = _spec_for_event(ctx, hist, ev, m, done)
        if spec is None:
            stats[(kind, "no_spec")] += 1
            continue
        key = (m["sk"], d, spec["provider"])
        visit_key = (m["sk"], d, "VISIT")
        if key in hist.day_claims or (kind in ("OP", "PHYSIO") and visit_key in hist.member_day_kind):
            stats[(kind, "same_day")] += 1
            continue
        ctx.last_claim = None
        claim = _compose(ctx, spec)
        if claim is None:
            stats[(kind, "compose_fail")] += 1
            continue
        stats[(kind, "ok")] += 1
        hist.day_claims.add(key)
        if kind in ("OP", "PHYSIO"):
            hist.member_day_kind.add(visit_key)
        info = ctx.last_claim
        done[ev["eid"]] = {"claim": claim, "clin": info["attending"], "provider": spec["provider"], "date": d}
        ctx.seen_mp.add((m["sk"], spec["provider"]))
        for r in info["lines"]:
            hist.code_last[(m["sk"], r["activity_code"])] = r["service_date"]
        if kind == "IP":
            hist.ip[m["sk"]].append((d, d + TD(days=int(spec["los"]))))
        # legitimate resubmission after a documentation denial
        if info["decision"] == "PARTIAL" and info["denial"] in RESUBMITTABLE and rng.random() < 0.45:
            _resubmit(ctx, info)


#: Target claim mix by claim type (task specification).
TARGET_MIX = {"OUTPATIENT": 0.55, "PHARMACY": 0.20, "LAB": 0.10, "RADIOLOGY": 0.07, "INPATIENT": 0.08}
_KIND_TYPE = {"OP": "OUTPATIENT", "PHYSIO": "OUTPATIENT", "RX": "PHARMACY", "REFILL": "PHARMACY", "LAB": "LAB",
              "RAD": "RADIOLOGY", "IP": "INPATIENT"}
#: Expected share of scheduled events that become claims, by kind (refills lose a few to overlaps).
_YIELD = {"OP": 0.95, "PHYSIO": 0.9, "RX": 0.9, "REFILL": 0.85, "LAB": 0.93, "RAD": 0.95, "IP": 0.9}


def _keep_probabilities(events: list[dict], target: int) -> dict[str, float]:
    """Keep-probability per root type so the kept events approximate ``target`` claims in TARGET_MIX."""
    from scipy.optimize import lsq_linear

    # acute and chronic care share one probability (so chronic care is never traded away);
    # the inpatient pathways get their own, which is what sets the inpatient share.
    group = {"ACUTE": "GENERAL", "CHRONIC": "GENERAL", "IP": "IP"}
    rtypes = sorted({group[e["rtype"]] for e in events})
    ctypes = list(TARGET_MIX)
    A = np.zeros((len(ctypes), len(rtypes)))
    for e in events:
        A[ctypes.index(_KIND_TYPE[e["kind"]]), rtypes.index(group[e["rtype"]])] += _YIELD[e["kind"]]
    T = np.array([target * TARGET_MIX[c] for c in ctypes])
    W = 1.0 / np.maximum(T, 1.0)
    A_w = np.vstack([A * W[:, None], 3.0 * A.sum(axis=0)[None, :] / max(target, 1)])
    T_w = np.concatenate([T * W, [3.0]])
    sol = lsq_linear(A_w, T_w, bounds=(0.0, 1.0))
    by_group = {r: float(np.clip(p, 0.0, 1.0)) for r, p in zip(rtypes, sol.x)}
    return {rt: by_group.get(g, 1.0) for rt, g in group.items()}


def _resubmit(ctx: _Ctx, info: dict) -> None:
    rng = ctx.rng
    spec = dict(info["spec"])
    header = info["header"]
    denied = [r for r in info["lines"] if r["line_sk"] in info["denied_lines"]]
    if not denied:
        return
    lines = []
    for r in denied:
        lines.append({"code": r["activity_code"], "units": r["units"], "service_date": r["service_date"],
                      "clinician": r["rendering_clinician_id"], "ordering": r["ordering_clinician_id"],
                      "start": r["service_start_time"], "unit_price": r["unit_price"], "indicator": r["indicator"],
                      "wastage": r.get("wastage_units") or 0.0,
                      "zero": r["gross_amount"] == 0})
    attachment = f"SYNTHETIC_clinical_note_resubmission_{header['claim_sk']}_en.txt"
    field = {"DOC-001": "attachment_ref", "DOC-003": "attachment_ref", "CLAI-012": "ordering_clinician_id"}[info["denial"]]
    changed = {field: [None, attachment if field == "attachment_ref" else "provided"]}
    lag = (info["settle"] - info["header"]["submission_date"]).days + int(rng.integers(5, 25))
    discharge = info["encounter"]["discharge_date"] or max(r["service_date"] for r in info["lines"])
    base_lag = (info["header"]["submission_date"] - discharge).days
    spec.update(lines=lines, decision="PAID", encounter_sk=header["encounter_sk"], attending=info["attending"],
                relationship="RESUBMISSION", version_no=2, prior_claim_sk=header["claim_sk"], changed_fields=changed,
                resubmission_type="CORRECTION", submission_lag=base_lag + lag, strict=False, referral=None,
                documents=True, claim_sk=None, start_minute=None, auth=False)
    for line in lines:
        line.setdefault("start", None)
    claim = _compose(ctx, spec)
    if claim is None:
        return
    # carry the original authorisation onto resubmitted lines that had one
    auth_by_code = {r["activity_code"]: r["authorization_id"] for r in denied}
    for r in ctx.rows["claim_line"][-len(lines):]:
        r["authorization_id"] = auth_by_code.get(r["activity_code"])
    text = ("*** SYNTHETIC DOCUMENT — GENERATED FOR SOFTWARE TESTING. NOT A REAL CLINICAL RECORD. ***\n"
            f"CLINICAL NOTE (SYNTHETIC) — supporting documentation for resubmission of {header['claim_sk']}.\n"
            "The previously missing documentation is attached. No billed code, unit or amount has changed.\n"
            "--- END OF SYNTHETIC DOCUMENT ---\n")
    ctx.rows["document"].append(_doc(ctx, claim, ctx.members[header["member_sk"]], header["provider_sk"],
                                     "CLINICAL_NOTE", "en", text, _dt_at(info["settle"] + TD(days=2), 10 * 60),
                                     attachment))


def _spec_for_event(ctx: _Ctx, hist: _History, ev: dict, m: dict, done: dict) -> dict | None:
    rng = ctx.rng
    kind = ev["kind"]
    d = ev["date"]
    parent = done.get(ev.get("parent")) if ev.get("parent") else None
    if kind == "OP":
        return _op_spec(ctx, hist, ev, m, parent)
    if kind in ("LAB", "RAD"):
        codes = []
        for c in ev["codes"]:
            if not _repeat_ok(ctx, hist, m["sk"], c, d):
                continue
            if ev.get("needs_prior") and c == ev["codes"][0]:
                last = hist.code_last.get((m["sk"], ev["needs_prior"]))
                if last is None or (d - last).days > 150 or last > d:
                    c = ev["needs_prior"]
                    if not _repeat_ok(ctx, hist, m["sk"], c, d):
                        continue
            codes.append(c)
        if kind == "RAD":
            prov = ctx.providers[ev["provider"]]
            codes = [c for c in codes if EQUIP_OF_CODE.get(c) in prov["equipment"]]
        if not codes:
            return None
        lines = ([{"code": "36415"}] if kind == "LAB" else []) + [{"code": c} for c in codes]
        for L in lines:
            L["ordering"] = parent["clin"] if parent else None
        adv = any(ctx.service_family(c) == "ADVANCED_IMAGING" for c in codes)
        return dict(member=m["sk"], provider=ev["provider"], date=d,
                    claim_type="LAB" if kind == "LAB" else "RADIOLOGY", lines=lines, dx=ev["dx"],
                    ordering=parent["clin"] if parent else None,
                    attending_family=ctx.family_of(lines[-1]["code"]),
                    referral={"from_clin": parent["clin"], "from_provider": parent["provider"], "date": parent["date"],
                              "letter": adv} if parent else None)
    if kind in ("RX", "REFILL"):
        items = []
        dx = []
        if kind == "RX":
            prescriber = parent["clin"] if parent else None
            for p in ev["items"]:
                it = _drug_item(ctx, hist, m, p, d, prescriber, parent["date"] if parent else d, ev.get("course_days"))
                if it:
                    items.append(it)
            dx = ev["dx"]
        else:
            for cond, veid in ev["items"]:
                visit = done.get(veid)
                if visit is None:
                    continue
                for p in m["drugs"].get(cond, []):
                    course = int(ev.get("cycle", 30))
                    it = _drug_item(ctx, hist, m, p, d, visit["clin"], visit["date"], course)
                    if it:
                        items.append(it)
                        if cond not in dx:
                            dx.append(cond)
        if not items:
            return None
        return dict(member=m["sk"], provider=ev["provider"], date=d, claim_type="PHARMACY", lines=items, dx=dx,
                    ordering=items[0]["ordering"], attending_family="DRUG_ORAL", attending_specialty="PHARMACY")
    if kind == "PHYSIO":
        prov = ev["provider"]
        lines = []
        if ev.get("first"):
            lines.append({"code": "97161" if rng.random() < .5 else "97162"})
        lines.append({"code": "97110", "units": 2})
        if rng.random() < 0.6:
            lines.append({"code": "97140", "units": 1})
        elif rng.random() < 0.3:
            lines.append({"code": "97035", "units": 1})
        return dict(member=m["sk"], provider=prov, date=d, claim_type="OUTPATIENT", lines=lines, dx=ev["dx"],
                    attending_family="PHYSIO", attending_specialty="PHYSIOTHERAPY",
                    ordering=parent["clin"] if parent else None,
                    referral={"from_clin": parent["clin"], "from_provider": parent["provider"], "date": parent["date"],
                              "letter": False} if parent and ev.get("first") else None)
    if kind == "IP":
        return _ip_spec(ctx, hist, ev, m, parent)
    return None


def _op_spec(ctx: _Ctx, hist: _History, ev: dict, m: dict, parent: dict | None) -> dict | None:
    rng = ctx.rng
    d = ev["date"]
    prov = ev["provider"]
    pinfo = ctx.providers[prov]
    setting = ev.get("setting", "CLINIC")
    spec_name = ev.get("specialty")
    if spec_name and spec_name not in pinfo["specialties"]:
        spec_name = None
    lines: list[dict] = []
    dx = ev["dx"]
    enc_type = {"ED": "EMERGENCY", "TELE": "TELEHEALTH"}.get(setting, "OUTPATIENT")
    if setting == "VACCINE":
        lines = [{"code": "90471"}, {"code": "90686"}]
        return dict(member=m["sk"], provider=prov, date=d, claim_type="OUTPATIENT", lines=lines, dx=dx,
                    attending_family="VACCINE", encounter_type="OUTPATIENT")
    if setting == "INFUSION":
        units, wastage = _injectable_units(ctx, ev["infusion"], m)
        lines = [{"code": _em_code(ctx, m["sk"], prov, 3, "CLINIC")}, {"code": "96365"}, {"code": "96366", "units": 1},
                 {"code": ev["infusion"], "units": units, "wastage": wastage,
                  "rx": {"days_supply": 56, "prescribed_qty": units, "dispensed_qty": units, "dose_mg_per_day": None}}]
        labs = ["85025", "86140"] if rng.random() < 0.5 else []
        lines += [{"code": "36415"}] + [{"code": c} for c in labs] if labs else []
        return dict(member=m["sk"], provider=prov, date=d, claim_type="OUTPATIENT", lines=lines, dx=dx,
                    attending_specialty="GASTROENTEROLOGY", attending_family="INFUSION")
    if setting == "CHEMO":
        units, wastage = _injectable_units(ctx, "J9355", m)
        lines = [{"code": _em_code(ctx, m["sk"], prov, 4, "CLINIC")}, {"code": "96413"}, {"code": "96415", "units": 1},
                 {"code": "J9355", "units": units, "wastage": wastage,
                  "rx": {"days_supply": 21, "prescribed_qty": units, "dispensed_qty": units, "dose_mg_per_day": None}},
                 {"code": "36415"}, {"code": "85025"}]
        if rng.random() < 0.3:
            pu, pw = _injectable_units(ctx, "J2506", m)
            lines.append({"code": "J2506", "units": pu, "wastage": pw,
                          "rx": {"days_supply": 21, "prescribed_qty": pu, "dispensed_qty": pu, "dose_mg_per_day": None}})
        return dict(member=m["sk"], provider=prov, date=d, claim_type="OUTPATIENT", lines=lines, dx=dx,
                    attending_specialty="ONCOLOGY", attending_family="CHEMO")
    # ---- E&M
    level = ev.get("level") or 3
    if setting == "TELE":
        em = _pick(rng, ["99212", "99213", "99214"], [.3, .5, .2]) if not ev.get("psych") else "99213"
        lines.append({"code": em, "indicator": "95"})
        att_fam = "TELEHEALTH"
    elif setting == "ED":
        lines.append({"code": _em_code(ctx, m["sk"], prov, level, "ED")})
        att_fam = "EM_ED"
    else:
        code = _em_code(ctx, m["sk"], prov, level, setting)
        if ev.get("psych") and rng.random() < 0.5 and (m["sk"], prov) in ctx.seen_mp:
            code = "90834"
        lines.append({"code": code})
        att_fam = ctx.family_of(code)
    procs = list(ev.get("office", []))
    if procs and ctx.family_of(lines[0]["code"]).startswith("EM_") and any(
            ctx.family_of(p) in ("PROC_MINOR", "PROC_ORTHO") for p in procs):
        lines[0]["indicator"] = "25"
    for p in procs:
        fam = ctx.family_of(p)
        if fam.startswith("IMAGING_") and EQUIP_OF_CODE.get(p) not in pinfo["equipment"]:
            continue
        if not _repeat_ok(ctx, hist, m["sk"], p, d):
            continue
        lines.append({"code": p})
    if ev.get("echo") and "ECHO" in "".join(PRIVILEGES.get(spec_name or "", [])) and _repeat_ok(ctx, hist, m["sk"], "93306", d):
        lines.append({"code": "93306"})
    if ev.get("injectable"):
        u, w = _injectable_units(ctx, ev["injectable"], m)
        lines.append({"code": "96372"})
        lines.append({"code": ev["injectable"], "units": u, "wastage": w,
                      "rx": {"days_supply": float(ctx.drugs[ev["injectable"]]["max_duration_days"] or 30), "prescribed_qty": u,
                             "dispensed_qty": u, "dose_mg_per_day": None}})
    for code in ev.get("dme", []):
        lines.append({"code": code})
    hospital = pinfo["type"] == "HOSPITAL"
    if hospital:
        labs = [c for c in ev.get("labs", []) if _repeat_ok(ctx, hist, m["sk"], c, d)]
        if labs:
            lines.append({"code": "36415"})
            lines += [{"code": c} for c in labs]
        for c in ev.get("imaging", []):
            if EQUIP_OF_CODE.get(c) in pinfo["equipment"] and _repeat_ok(ctx, hist, m["sk"], c, d):
                lines.append({"code": c, "indicator": "RT" if c in ("73562", "73721") else None})
        # ED/hospital-dispensed take-home medicines and injections
        if setting == "ED":
            inj = {"S93.401A": "J1885", "A09": "J2405", "S61.411A": None, "R07.9": None}.get(dx[0])
            if inj and rng.random() < 0.6:
                u, w = _injectable_units(ctx, inj, m)
                lines.append({"code": "96372" if inj != "J2405" else "96374"})
                lines.append({"code": inj, "units": u, "wastage": w,
                              "rx": {"days_supply": 1, "prescribed_qty": u, "dispensed_qty": u, "dose_mg_per_day": None}})
        for p in ev.get("rx_inhouse", []):
            it = _drug_item(ctx, hist, m, p, d, None, d)
            if it:
                it["ordering"] = None
                lines.append(it)
    ordering = parent["clin"] if parent else None
    referral = None
    if parent and setting == "REFERRAL":
        referral = {"from_clin": parent["clin"], "from_provider": parent["provider"], "date": parent["date"],
                    "letter": True}
    elif ev["scenario"].startswith("PREOP") and parent:
        referral = {"from_clin": parent["clin"], "from_provider": parent["provider"], "date": parent["date"], "letter": True}
    spec = dict(member=m["sk"], provider=prov, date=d, claim_type="OUTPATIENT", lines=lines, dx=dx,
                attending_specialty=spec_name, attending_family=att_fam, encounter_type=enc_type,
                referral=referral, accident=bool(ev.get("injury")) and rng.random() < 0.22)
    # medicines ordered by this visit's clinician are dispensed with this claim: fill ordering later
    spec["_self_order"] = True
    return _finalise_self_orders(ctx, spec)


def _finalise_self_orders(ctx: _Ctx, spec: dict) -> dict:
    """In-house labs, imaging and medicines are ordered by the attending: resolve after picking them."""
    prov = spec["provider"]
    att = ctx.pick_clinician(prov, spec["attending_family"], spec["date"], specialty=spec.get("attending_specialty"))
    if att is None:
        att = ctx.pick_clinician(prov, spec["attending_family"], spec["date"])
    if att is None:
        return None
    spec["attending"] = att
    for L in spec["lines"]:
        fam = ctx.family_of(L["code"])
        if fam.startswith(("LAB_", "IMAGING_")) or fam in ("DRUG_ORAL",) or (fam == "DRUG_INJECTABLE"):
            L.setdefault("ordering", att)
            if L.get("ordering") is None:
                L["ordering"] = att
            if "rx" in L and L["rx"].get("prescriber") is None:
                L["rx"]["prescriber"] = att
    spec.pop("_self_order", None)
    return spec


def _ip_spec(ctx: _Ctx, hist: _History, ev: dict, m: dict, parent: dict | None) -> dict | None:
    rng = ctx.rng
    d = ev["date"]
    cr = ev["cr"]
    cdef = ctx.case_rates[cr]
    lo, hi = cdef["los"]
    los = int(rng.integers(lo, hi + 1))
    # no overlapping or bundle-window readmissions in the base
    for a, b in hist.ip[m["sk"]]:
        if a - TD(days=45) <= d <= b + TD(days=45):
            return None
    for (mm, day, _p) in list(hist.day_claims):
        pass
    prov = ev.get("provider_hint") or _hospital_for(ctx, m, cr)
    if prov is None:
        return None
    pinfo = ctx.providers[prov]
    if cdef.get("needs") and cdef["needs"] not in pinfo["equipment"]:
        return None
    # the member must not have claims inside the stay (they are not scheduled yet: later events are blocked)
    principal = ev.get("dx_hint") or cdef["dx"][int(rng.integers(len(cdef["dx"])))]
    chapter = ref.icd_chapter(principal)
    secondaries = [c for c in m["chronic"] if c in CHRONIC_SECONDARY_OK]
    if chapter == "XV_pregnancy_childbirth":
        secondaries = [c for c in secondaries if c in ("E66.9", "E03.9", "J45.909")]
    poa = {c: "Y" for c in secondaries}
    for code, p in ref.ACUTE_SECONDARY.get(chapter, {}).items():
        if code == principal or code in secondaries:
            continue
        if rng.random() < p:
            secondaries.append(code)
            poa[code] = "Y" if rng.random() < 0.8 else "N"
    weights = [ctx.world.context["drg_grouper"].get(c, (False, False, 0.0)) for c in secondaries]
    severity = "A" if any(w[1] for w in weights) else ("B" if any(w[0] for w in weights) else "C")
    cr_code = f"{cr}{severity}"
    admission_type = cdef["admission"]
    specialty = cdef["specialty"]
    day_case = cdef["day_case"]
    # attending, surgeon, anaesthetist must be available across the stay's first day
    lines: list[dict] = [{"code": cr_code, "day": 0}]
    proc = cdef.get("procedure")
    laterality = {"H25.11": "RT", "H25.12": "LT", "M17.11": "RT", "M23.211": "RT", "S72.001A": "RT"}.get(principal)
    if cdef.get("pre_procedure"):
        lines.append({"code": cdef["pre_procedure"], "day": 0, "zero": True})
    if proc:
        lines.append({"code": proc, "day": 0 if admission_type == "ELECTIVE" or day_case else min(1, los),
                      "zero": True, "indicator": laterality})
    anaes = cdef.get("anaesthesia")
    if anaes and proc:
        pmins = ctx.codes[proc]["minutes"] or 60
        lines.append({"code": anaes, "day": lines[-1]["day"], "units": math.ceil((pmins + 30) / 15), "zero": True})
    implant = cdef.get("implant")
    if implant:
        lines.append({"code": implant, "day": lines[-1]["day"] if proc else 0, "device": True, "indicator": laterality})
    # E&M
    if los == 0:
        lines.append({"code": "99238", "day": 0, "zero": True})
    else:
        lines.append({"code": f"9922{1 + (severity != 'C') + (severity == 'A')}", "day": 0, "zero": True})
        for k in range(1, los):
            lines.append({"code": "99232" if severity != "C" and k == 1 else "99231", "day": k, "zero": True})
        lines.append({"code": "99239" if los >= 4 else "99238", "day": los, "zero": True})
    # rooms
    room = cdef["room"]
    if los == 0:
        lines.append({"code": "RM-DAYCARE", "day": 0, "zero": True})
    else:
        for k in range(los):
            code = room if not (room == "RM-ICU" and k >= 2) else "RM-WARD"
            lines.append({"code": code, "day": k, "zero": True})
    # labs, imaging, echo
    labs = [c for c in cdef.get("labs", []) if _repeat_ok(ctx, hist, m["sk"], c, d)]
    if labs:
        lines.append({"code": "36415", "day": 0, "zero": True})
        lines += [{"code": c, "day": 0, "zero": True} for c in labs]
        if los >= 3 and "85025" in labs:
            lines.append({"code": "85025", "day": 2, "zero": True})
    for c in cdef.get("imaging", []):
        if EQUIP_OF_CODE.get(c) in pinfo["equipment"] and _repeat_ok(ctx, hist, m["sk"], c, d):
            lines.append({"code": c, "day": 0, "zero": True, "indicator": laterality if c in ("73562", "73502") else None})
    if cdef.get("echo") and "CARDIOLOGY" in pinfo["specialties"] and _repeat_ok(ctx, hist, m["sk"], "93306", d):
        lines.append({"code": "93306", "day": min(1, los), "zero": True})
    if cdef.get("physio") and los >= 2:
        lines.append({"code": "97161", "day": 1, "zero": True})
        for k in range(1, los):
            lines.append({"code": "97110", "day": k, "units": 2, "zero": True})
    if cdef.get("hydration"):
        lines.append({"code": "96360", "day": 0, "zero": True})
    if cdef.get("nebuliser"):
        for k in range(max(los, 1)):
            lines.append({"code": "94640", "day": k, "zero": True})
    # drugs: injectables per day (each day its own line), orals as one course
    for p in cdef.get("drugs", []):
        drug = ctx.drugs[p]
        if p.startswith("J"):
            u, w = _injectable_units(ctx, p, m)
            for k in range(max(min(los, 5), 1)):
                lines.append({"code": p, "day": k, "units": u, "wastage": w, "zero": not drug["is_high_cost"],
                              "rx": {"days_supply": 1, "prescribed_qty": u, "dispensed_qty": u, "dose_mg_per_day": None}})
        else:
            if principal.startswith("O") and p == "RX1033":
                pass
            it = _drug_item(ctx, hist, m, p, d + TD(days=los), None, d + TD(days=los))
            if it:
                it["zero"] = True
                it["day"] = los
                lines.append(it)
    spec = dict(member=m["sk"], provider=prov, date=d, claim_type="INPATIENT", lines=lines,
                dx=[principal] + [c for c in secondaries if c != principal], poa=poa, los=los,
                admission_type=admission_type, encounter_type="DAY_CASE" if day_case else "INPATIENT",
                attending_specialty=specialty, attending_family="CASE_RATE",
                benefit_family=ctx.codes[cr_code]["service_family"],
                observation_status="OBSERVATION" if cr == "CR118" else "ADMITTED",
                discharge_type="HOME")
    att = ctx.pick_clinician(prov, "CASE_RATE", d, specialty=specialty)
    if att is None:
        return None
    # the attending must also be available on discharge day for the discharge E&M, else a colleague
    spec["attending"] = att
    for L in lines:
        fam = ctx.family_of(L["code"])
        sd = d + TD(days=int(L.get("day", 0)))
        if fam in ("EM_INPATIENT_INIT", "EM_INPATIENT_SUBSEQ", "EM_DISCHARGE", "CASE_RATE", "ROOM_DAY") or \
                fam.startswith(("SURG", "OBSTETRIC", "CARDIAC")) or fam == "IMPLANT":
            c = ctx.pick_clinician(prov, fam, sd, specialty=specialty, prefer=att)
            if c is None:
                return None
            L["clinician"] = c
        if fam.startswith(("LAB_", "IMAGING_")) or fam in ("ECHO",) or ctx.is_drug(L["code"]):
            L["ordering"] = att
            if "rx" in L:
                L["rx"]["prescriber"] = att
    if parent:
        spec["referral"] = {"from_clin": parent["clin"], "from_provider": parent["provider"], "date": parent["date"],
                            "letter": parent["provider"] != prov}
    spec["bed_id"] = _bed(ctx, prov, d, los, room)
    spec["accident"] = principal.startswith("S") and rng.random() < 0.15
    return spec


def _bed(ctx: _Ctx, prov: str, d: D, los: int, room: str) -> str:
    beds = ctx.beds[prov]
    n = ctx.providers[prov]["beds"] or 50
    ward = {"RM-ICU": "ICU", "RM-DAYCARE": "DCU", "RM-SEMI": "SP", "RM-PVT": "PV"}.get(room, "GW")
    for _ in range(50):
        bed = f"{prov}-{ward}-{int(ctx.rng.integers(1, n + 1)):03d}"
        if beds.get(bed, D(1900, 1, 1)) < d:
            beds[bed] = d + TD(days=max(los, 0))
            return bed
    bed = f"{prov}-{ward}-X{len(beds):03d}"
    beds[bed] = d + TD(days=los)
    return bed


# =============================================================================
# after the claims: samples, TPL, COB, inventory
# =============================================================================


def _post(ctx: _Ctx) -> None:
    rng, world = ctx.rng, ctx.world
    rows = ctx.rows
    headers = rows["claim_header"]
    # --- coordination of benefits: the secondary payer pays the member's share
    for h in headers:
        m = ctx.members[h["member_sk"]]
        if m.get("cob"):
            rows["other_payer_remittance"].append({
                "claim_sk": h["claim_sk"], "other_payer_id": m["cob"], "payment_amount": round(h["patient_share"], 2),
                "settlement_date": h["settlement_date"] + TD(days=int(rng.integers(7, 30))),
                "cross_payer_match_token": h["cross_payer_match_token"], "tenant_id": TENANT})
    # --- third-party liability for accident claims
    for h in headers:
        if not h["accident_indicator"]:
            continue
        acc = _pick(rng, ["ROAD_TRAFFIC", "WORKPLACE", "SPORTS", "SLIP_AND_FALL"], [.45, .3, .15, .1])
        liable = {"ROAD_TRAFFIC": f"Synthetic Motor Insurer {int(rng.integers(1, 4))}",
                  "WORKPLACE": "Employer liability cover (synthetic)", "SPORTS": "Club liability cover (synthetic)",
                  "SLIP_AND_FALL": "Premises liability cover (synthetic)"}[acc]
        enc_date = h["submission_date"] - TD(days=5)
        rows["third_party_liability"].append({"claim_sk": h["claim_sk"], "member_sk": h["member_sk"], "accident_type": acc,
                                              "liable_party": liable, "reported_date": enc_date, "tenant_id": TENANT})
        if rng.random() < 0.4:
            sd = h["settlement_date"] + TD(days=int(rng.integers(30, 150)))
            if sd <= world.end + TD(days=90):
                amount = round(h["gross_amount"] * float(rng.uniform(0.5, 1.0)), 2)
                rows["third_party_settlement"].append({"claim_sk": h["claim_sk"], "settlement_amount": amount,
                                                       "settlement_date": sd, "payer": liable, "tenant_id": TENANT})
                rows["recovery_ledger"].append({
                    "ledger_sk": world.new_id("LED", 6), "provider_sk": h["provider_sk"], "claim_sk": h["claim_sk"],
                    "credit_amount": amount, "credit_date": sd, "applied": True,
                    "applied_date": sd + TD(days=int(rng.integers(3, 30))), "tenant_id": TENANT})
    # --- routine provider refunds (applied)
    paid = [h for h in headers if h["net_amount"] > 0]
    for i in rng.choice(len(paid), size=min(len(paid), max(3, len(paid) // 400)), replace=False):
        h = paid[int(i)]
        cd = h["settlement_date"] + TD(days=int(rng.integers(20, 120)))
        rows["recovery_ledger"].append({
            "ledger_sk": world.new_id("LED", 6), "provider_sk": h["provider_sk"], "claim_sk": h["claim_sk"],
            "credit_amount": round(h["net_amount"] * float(rng.uniform(0.02, 0.15)), 2), "credit_date": cd,
            "applied": True, "applied_date": cd + TD(days=int(rng.integers(3, 30))), "tenant_id": TENANT})
    # --- member confirmation (2%), receipts (10% of claims with a patient share), attendance (5% of visits)
    enc_by_claim = {e["claim_sk"]: e for e in rows["encounter"]}
    for h in headers:
        u = rng.random()
        if u < 0.02:
            rows["member_confirmation"].append({
                "confirmation_sk": world.new_id("MCF", 6), "member_sk": h["member_sk"], "claim_sk": h["claim_sk"],
                "service_confirmed": True, "response": "CONFIRMED",
                "response_date": h["submission_date"] + TD(days=int(rng.integers(3, 20))),
                "channel": _pick(rng, ["SMS", "APP", "CALL"], [.5, .35, .15]), "tenant_id": TENANT})
        if h["patient_share"] > 0 and not ctx.members[h["member_sk"]].get("cob") and rng.random() < 0.10:
            enc = enc_by_claim.get(h["claim_sk"])
            rd = enc["start_time"].date() if enc else h["submission_date"]
            rows["member_receipt"].append({
                "receipt_sk": world.new_id("RCT", 7), "member_sk": h["member_sk"], "claim_sk": h["claim_sk"],
                "provider_sk": h["provider_sk"], "amount_paid_aed": h["patient_share"],
                "item_description": {"PHARMACY": "Co-payment, medicines", "LAB": "Co-payment, laboratory",
                                     "RADIOLOGY": "Co-payment, imaging", "INPATIENT": "Co-payment, admission"}.get(
                                         h["claim_type"], "Co-payment, consultation"),
                "receipt_date": rd, "tenant_id": TENANT})
        if h["claim_type"] == "OUTPATIENT" and rng.random() < 0.05:
            enc = enc_by_claim.get(h["claim_sk"])
            if enc is not None and enc["encounter_type"] in ("OUTPATIENT", "EMERGENCY"):
                rows["attendance_record"].append({
                    "record_sk": world.new_id("ATT", 7), "member_sk": h["member_sk"], "provider_sk": h["provider_sk"],
                    "claim_sk": h["claim_sk"], "attendance_date": enc["start_time"].date(), "attended": True,
                    "source": "CLINIC_CHECKIN_LOG", "tenant_id": TENANT})
    # --- a few benign complaints
    texts = {"WAITING_TIME": "SYNTHETIC complaint: waited over an hour past the appointment time.",
             "STAFF_ATTITUDE": "SYNTHETIC complaint: reception staff were unhelpful when rescheduling.",
             "BILLING_QUERY": "SYNTHETIC complaint: asked why a co-payment applied; explained and resolved.",
             "PARKING": "SYNTHETIC complaint: no parking available near the clinic entrance."}
    for i in rng.choice(len(headers), size=max(2, len(headers) // 600), replace=False):
        h = headers[int(i)]
        kind = _pick(rng, list(texts))
        rows["complaint"].append({"complaint_sk": world.new_id("CMP", 6), "member_sk": h["member_sk"],
                                  "provider_sk": h["provider_sk"], "claim_sk": h["claim_sk"], "complaint_type": kind,
                                  "complaint_date": h["submission_date"] + TD(days=int(rng.integers(0, 10))),
                                  "text": texts[kind], "tenant_id": TENANT})


def _pharmacy_inventory(ctx: _Ctx) -> list[dict]:
    rng = ctx.rng
    months = pd.period_range(ctx.world.start, ctx.world.end, freq="M").strftime("%Y-%m").tolist()
    by_pp: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (ph, prod, period), qty in ctx.dispensed.items():
        by_pp[(ph, prod)][period] = by_pp[(ph, prod)].get(period, 0.0) + qty
    out = []
    for (ph, prod), per in sorted(by_pp.items()):
        avg = sum(per.values()) / max(len(per), 1)
        opening = float(math.ceil(avg * rng.uniform(0.8, 1.6)))
        first = min(per)
        for mth in months:
            if mth < first:
                continue
            disp = per.get(mth, 0.0)
            target = float(math.ceil(avg * rng.uniform(0.5, 1.5)))
            purchased = max(0.0, target - opening + disp)
            closing = opening + purchased - disp
            out.append({"pharmacy_id": ph, "product": prod, "period": mth, "opening_stock": opening,
                        "purchased_qty": purchased, "closing_stock": closing, "tenant_id": TENANT})
            opening = closing
    return out


def _calibrate_morbidity(ctx: _Ctx) -> None:
    """Expected prevalence of each secondary code by principal chapter, measured on the clean base."""
    dx = pd.DataFrame(ctx.rows["diagnosis"])
    hdr = pd.DataFrame(ctx.rows["claim_header"])[["claim_sk", "claim_type"]]
    dx = dx.merge(hdr, on="claim_sk")
    dx = dx[dx["claim_type"] == "INPATIENT"]
    principal = dx[dx["sequence"] == 1][["claim_sk", "chapter"]].rename(columns={"chapter": "principal_chapter"})
    n_by_ch = principal.groupby("principal_chapter").size()
    sec = dx[dx["sequence"] > 1].merge(principal, on="claim_sk")
    counts = sec.groupby(["principal_chapter", "code"]).claim_sk.nunique()
    rows = []
    prior = ctx.world.tables["morbidity_model"]
    seen = set()
    for (ch, code), k in counts.items():
        n = n_by_ch.get(ch, 0)
        if n >= 5:
            rows.append({"principal_chapter": ch, "secondary_code": code,
                         "expected_prevalence": round(float(k) / float(n), 4)})
            seen.add((ch, code))
    for r in prior.itertuples(index=False):
        if (r.principal_chapter, r.secondary_code) not in seen:
            rows.append({"principal_chapter": r.principal_chapter, "secondary_code": r.secondary_code,
                         "expected_prevalence": float(r.expected_prevalence)})
    ctx.world.tables["morbidity_model"] = pd.DataFrame(rows).sort_values(
        ["principal_chapter", "secondary_code"]).reset_index(drop=True)


# =============================================================================
# public API
# =============================================================================


def build(world: World, n_claims: int = DEFAULT_CLAIMS, *, verbose: bool = False) -> None:
    """Create the clean population (~``n_claims`` claims) and register :func:`make_claim`."""
    if "codes" not in world.context:
        ref.build_reference(world)
    ctx = _Ctx(world)
    ctx.dispensed = defaultdict(float)
    ctx.stats = defaultdict(int)
    world.context["base_ctx"] = ctx
    scale = min(1.0, n_claims / DEFAULT_CLAIMS) if n_claims < DEFAULT_CLAIMS else n_claims / DEFAULT_CLAIMS
    prov_scale = min(scale, 1.0) if n_claims <= DEFAULT_CLAIMS else scale ** 0.5
    n_members = max(400, int(round(9000 * n_claims / DEFAULT_CLAIMS)))
    _build_providers(ctx, prov_scale)
    _build_people(ctx, n_members, prov_scale)
    mult = {"acute": 1.0, "chronic": 1.0, "ip": 8.0}
    sched = _schedule(ctx, mult)
    ctx.rows = _new_rows()
    _build_claims(ctx, sched, int(n_claims / 1.03))
    _post(ctx)
    _calibrate_morbidity(ctx)
    for name, rows in ctx.rows.items():
        world.append(name, pd.DataFrame(rows))
    world.append("pharmacy_inventory", _pharmacy_inventory(ctx))
    ctx.rows = None
    world.claim_factory = make_claim
    if verbose:
        h = world.tables["claim_header"]
        print(f"  base: {len(h):,} claims, {len(world.tables['claim_line']):,} lines, "
              f"{h['member_sk'].nunique():,} members with claims, {len(ctx.providers)} providers")
        print("  mix: " + ", ".join(f"{k} {v:.1%}" for k, v in h['claim_type'].value_counts(normalize=True).items()))
        kinds = sorted({k for k, _ in ctx.stats})
        for k in kinds:
            print(f"    {k:7s} " + " ".join(f"{r}={ctx.stats[(k, r)]}" for r in
                  ("scheduled", "ok", "no_cover", "no_parent", "ip_block", "no_spec", "same_day", "compose_fail")))


def make_claim(world: World, *, member_sk: str, provider_sk: str, service_date: D, claim_type: str,
               lines: list | None = None, diagnoses: list | None = None, clinician_id: str | None = None,
               ordering_clinician_id: str | None = None, los: int | None = None, case_rate: str | None = None,
               admission_type: str | None = None, encounter_type: str | None = None,
               decision: str | None = "PAID", denial_code: str | None = None, submission_lag_days: int | None = None,
               payer_id: str | None = None, accident: bool = False, authorization: bool | str = "auto",
               documents: bool = True, referral_from: str | None = None, start_minute: int | None = None,
               strict: bool = False) -> str:
    """Build ONE complete, internally consistent claim and append it to every claim table.

    Parameters
    ----------
    member_sk, provider_sk : str
        Existing member and provider. Cover is taken from the member's coverage row active on
        ``service_date`` (else the latest one) — nothing is validated, so an injector may plant a
        claim outside cover on purpose.
    service_date : datetime.date
        First day of service (admission date for INPATIENT).
    claim_type : {"OUTPATIENT", "PHARMACY", "LAB", "RADIOLOGY", "INPATIENT"}
    lines : list of str or dict, optional
        Activity codes, or dicts with ``code`` (required) and any of: ``units`` (1), ``day``
        (offset from service_date, 0) or ``service_date``, ``clinician`` (rendering),
        ``ordering``, ``indicator`` (modifier), ``unit_price`` (overrides tariff × discount),
        ``gross`` (overrides unit_price × units), ``zero`` (package component at 0),
        ``wastage`` (drug wastage units), ``minutes``, ``start`` (datetime), ``device`` (bool),
        ``rx`` (dict: prescriber, prescribed_date, days_supply, prescribed_qty, dispensed_qty,
        dispensed_product, prescribed_product, dose_mg_per_day). Default: a plausible set for the
        claim type (an E&M visit for URTI; a paracetamol fill; a CBC; a chest X-ray; a pneumonia
        case rate).
    diagnoses : list of str, optional
        ICD-10-CM codes, principal first. Default: the chosen scenario's diagnosis.
    clinician_id : str, optional
        Attending/rendering clinician (not validated — plant unlicensed/on-leave ones this way).
    ordering_clinician_id : str, optional
        Ordering clinician/prescriber for lab, imaging and drug lines.
    los, case_rate, admission_type
        INPATIENT only. ``case_rate`` is a base code like ``"CR110"`` (default) — the severity
        suffix is added from the secondary diagnoses unless ``lines`` are given explicitly.
    encounter_type : str, optional
        Overrides the default (OUTPATIENT/PHARMACY/DIAGNOSTIC/INPATIENT/DAY_CASE…).
    decision : {"PAID", "PARTIAL", "DENIED", None}
        Remittance decision; ``None`` draws the base mix (88/7/5). Default ``"PAID"``.
    denial_code, submission_lag_days, payer_id, accident
        As named. ``submission_lag_days`` counts from discharge / last service date.
    authorization : "auto" | False
        ``"auto"`` creates an APPROVED, valid authorisation for lines that need one;
        ``False`` creates none (plant missing authorisations this way).
    documents : bool
        Generate the documents the claim would carry (discharge summary, reports, notes).
    referral_from : str, optional
        Clinician id of a referrer: adds a referral row pointing at this claim.
    start_minute : int, optional
        Encounter start as minutes after midnight (plant overlapping sessions).
    strict : bool
        When True, return ``None``-equivalent failure (RuntimeError) if no privileged
        clinician is available; default False picks any clinician on the roster.

    Returns
    -------
    str
        The new ``claim_sk`` (``World.make_claim`` marks it used).
    """
    ctx: _Ctx = world.context["base_ctx"]
    rng = ctx.rng
    m = ctx.members[member_sk]
    ptype = ctx.providers[provider_sk]["type"]
    if isinstance(service_date, DT):
        service_date = service_date.date()
    hist = _History()
    spec: dict[str, Any]
    if claim_type == "INPATIENT" and lines is None:
        ev = {"date": service_date, "cr": case_rate or "CR110", "provider_hint": provider_sk,
              "dx_hint": diagnoses[0] if diagnoses else None}
        spec = _ip_spec(ctx, hist, ev, m, None) or {}
        if not spec:
            raise RuntimeError(f"No clinician at {provider_sk} can attend a {ev['cr']} admission on {service_date}.")
        if los is not None and los != spec["los"]:
            spec["lines"] = [L for L in spec["lines"] if int(L.get("day", 0)) <= los]
            spec["los"] = int(los)
        if diagnoses:
            spec["dx"] = list(diagnoses)
    else:
        norm = _default_lines(ctx, m, provider_sk, service_date, claim_type, diagnoses) if lines is None else \
            [dict(code=l) if isinstance(l, str) else dict(l) for l in lines]
        for L in norm:
            if "activity_code" in L and "code" not in L:
                L["code"] = L.pop("activity_code")
        dx = list(diagnoses) if diagnoses else _default_dx(ctx, claim_type, norm)
        spec = dict(member=member_sk, provider=provider_sk, date=service_date, claim_type=claim_type, lines=norm,
                    dx=dx, los=int(los or 0), admission_type=admission_type)
        fam = ctx.family_of(norm[0]["code"]) if norm else "EM_OFFICE_EST"
        spec["attending_family"] = fam
        if claim_type == "INPATIENT":
            spec["benefit_family"] = ctx.service_family(norm[0]["code"])
            spec["bed_id"] = _bed(ctx, provider_sk, service_date, int(los or 0), "RM-WARD")
            spec["discharge_type"] = "HOME"
        if ptype == "PHARMACY":
            spec["attending_specialty"] = "PHARMACY"
            spec["attending_family"] = "DRUG_ORAL"
    if clinician_id:
        spec["attending"] = clinician_id
        for L in spec["lines"]:
            fam = ctx.family_of(L["code"])
            if L.get("clinician") is None and not fam.startswith(("LAB_", "IMAGING_", "ANAES")) and fam not in (
                    "ROOM_DAY",) and (ptype != "PHARMACY"):
                L["clinician"] = clinician_id
    if ordering_clinician_id:
        spec["ordering"] = ordering_clinician_id
        for L in spec["lines"]:
            fam = ctx.family_of(L["code"])
            if fam.startswith(("LAB_", "IMAGING_")) or ctx.is_drug(L["code"]):
                L["ordering"] = ordering_clinician_id
                if "rx" in L:
                    L["rx"]["prescriber"] = ordering_clinician_id
    spec.update(decision=decision, denial_code=denial_code, submission_lag=submission_lag_days, payer=payer_id,
                accident=accident, auth=authorization, documents=documents, strict=strict,
                start_minute=start_minute)
    if encounter_type:
        spec["encounter_type"] = encounter_type
    if admission_type:
        spec["admission_type"] = admission_type
    if referral_from:
        rc = ctx.clin_by_id.get(referral_from, {})
        spec["referral"] = {"from_clin": referral_from, "from_provider": rc.get("provider"), "date": service_date,
                            "letter": False}
    if spec.get("attending") is None and ctx.pick_clinician(provider_sk, spec.get("attending_family", ""), service_date,
                                                           spec.get("attending_specialty")) is None:
        # fall back to any rostered clinician so injectors always get a claim
        roster = ctx.clinicians.get(provider_sk) or []
        if roster:
            spec["attending"] = roster[int(rng.integers(len(roster)))]["id"]
    saved = ctx.rows
    ctx.rows = _new_rows()
    try:
        claim = _compose(ctx, spec)
        if claim is None:
            raise RuntimeError(f"make_claim could not build a claim for {member_sk} at {provider_sk} on {service_date}.")
        produced = ctx.rows
    finally:
        ctx.rows = saved
    for name, rows in produced.items():
        if rows:
            world.append(name, pd.DataFrame(rows))
    # keep pharmacy stock consistent: the extra dispensing was purchased the same month
    for r in produced.get("prescription_dispense", []):
        _inventory_add(world, r["pharmacy_id"], r["billed_product"], r["fill_date"].strftime("%Y-%m"), r["dispensed_qty"])
    ctx.seen_mp.add((member_sk, provider_sk))
    return claim


def _inventory_add(world: World, pharmacy: str, product: str, period: str, qty: float) -> None:
    inv = world.tables.get("pharmacy_inventory")
    if inv is None or inv.empty:
        world.append("pharmacy_inventory", [{"pharmacy_id": pharmacy, "product": product, "period": period,
                                             "opening_stock": 0.0, "purchased_qty": float(qty), "closing_stock": 0.0,
                                             "tenant_id": TENANT}])
        return
    mask = (inv["pharmacy_id"] == pharmacy) & (inv["product"] == product) & (inv["period"] == period)
    if mask.any():
        inv.loc[mask, "purchased_qty"] = inv.loc[mask, "purchased_qty"].astype(float) + float(qty)
    else:
        world.append("pharmacy_inventory", [{"pharmacy_id": pharmacy, "product": product, "period": period,
                                             "opening_stock": 0.0, "purchased_qty": float(qty), "closing_stock": 0.0,
                                             "tenant_id": TENANT}])


def _default_dx(ctx: _Ctx, claim_type: str, lines: list[dict]) -> list[str]:
    codes = [L["code"] for L in lines]
    for code, prefixes in ctx.world.context["indications"].items():
        if code in codes:
            for p in prefixes:
                matches = [c for c in ctx.icd if c.startswith(p)]
                if matches:
                    return [matches[0]]
    for c in codes:
        if c in ctx.drugs and ctx.drugs[c]["indication_prefixes"]:
            p = ctx.drugs[c]["indication_prefixes"].split(";")[0]
            matches = [x for x in ctx.icd if x.startswith(p)]
            if matches:
                return [matches[0]]
    return {"LAB": ["R50.9"], "RADIOLOGY": ["R05.9"], "PHARMACY": ["J06.9"], "INPATIENT": ["J18.9"]}.get(
        claim_type, ["J06.9"])


def _default_lines(ctx: _Ctx, m: dict, provider: str, d: D, claim_type: str, diagnoses) -> list[dict]:
    if claim_type == "PHARMACY":
        q, days, _, dose = _dose_units(ctx, "RX1033", m, d)
        return [{"code": "RX1033", "units": q, "rx": {"days_supply": days, "prescribed_qty": q, "dispensed_qty": q,
                                                      "dose_mg_per_day": dose}}]
    if claim_type == "LAB":
        return [{"code": "36415"}, {"code": "85025"}]
    if claim_type == "RADIOLOGY":
        return [{"code": "71046"}]
    code = "9921" + str(int(ctx.rng.choice([2, 3, 3, 4]))) if (m["sk"], provider) in ctx.seen_mp else "99203"
    return [{"code": code}]
