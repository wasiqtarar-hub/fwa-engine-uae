"""Reference tables for the SYNTHETIC UAE demo dataset.

``build_reference(world)`` writes every REFERENCE table declared in
``fwa.canonical.supplementary.REFERENCE_TABLES`` into ``world.tables`` and
leaves the catalogues the base population needs in ``world.context``:

* ``world.context["codes"]``      — code → dict (one row of activity_code_reference)
* ``world.context["drugs"]``      — product → dict (one row of drug_policy, plus a dosing regimen)
* ``world.context["icd"]``        — ICD-10-CM code → (description, chapter)
* ``world.context["case_rates"]`` — base case-rate code → definition (principal dx, procedure, LOS …)

Vocabularies shared with injectors and controls (all upper-case strings):

``activity_type``
    ``CPT`` (numeric procedure / E&M / lab / imaging codes), ``HCPCS`` (devices,
    injectable drugs J-codes, supplies), ``DRUG`` (oral/topical products ``RX####``),
    ``DRG`` (inpatient case-rate codes ``CR###A/B/C``), ``SERVICE`` (room/day codes ``RM-*``).
``service_family``
    CONSULTATION, TELEHEALTH, EMERGENCY, INPATIENT, DAY_SURGERY, PROCEDURE, LAB, IMAGING,
    ADVANCED_IMAGING, PHARMACY, SPECIALTY_DRUG, DEVICE, PHYSIOTHERAPY, MENTAL_HEALTH,
    MATERNITY, COSMETIC (never covered).
``code_family``
    finer families (``EM_OFFICE_EST``, ``LAB_PANEL``, ``IMAGING_MRI`` …); clinician
    ``privileges`` in ``clinician_roster`` are semicolon lists of these.

Case-rate codes carry a severity suffix — ``A`` (with an MCC), ``B`` (with a CC),
``C`` (neither) — chosen from the secondary diagnoses through ``drg_grouper``.

Every code, product and diagnosis is real-looking but the prices, policies and
limits are INVENTED for software testing. Nothing here is a real payer's tariff.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import pandas as pd

__all__ = ["build_reference", "icd_chapter", "ICD10", "CODES", "DRUGS", "CASE_RATES", "CHAPTER_LABELS"]

EPOCH = _dt.date(2018, 1, 1)
OPEN_END = _dt.date(2099, 12, 31)

# --------------------------------------------------------------------------- ICD-10

#: Chapter naming follows ``config/peer_groups.yaml`` (roman numeral + slug).
_CHAPTER_RANGES: list[tuple[str, str, str]] = [
    ("A00", "B99", "I_infectious"),
    ("C00", "D49", "II_neoplasms"),
    ("D50", "D89", "III_blood_immune"),
    ("E00", "E89", "IV_endocrine_metabolic"),
    ("F01", "F99", "V_mental_behavioural"),
    ("G00", "G99", "VI_nervous_system"),
    ("H00", "H59", "VII_eye_adnexa"),
    ("H60", "H95", "VIII_ear_mastoid"),
    ("I00", "I99", "IX_circulatory"),
    ("J00", "J99", "X_respiratory"),
    ("K00", "K95", "XI_digestive"),
    ("L00", "L99", "XII_skin_subcutaneous"),
    ("M00", "M99", "XIII_musculoskeletal"),
    ("N00", "N99", "XIV_genitourinary"),
    ("O00", "O9A", "XV_pregnancy_childbirth"),
    ("P00", "P96", "XVI_perinatal"),
    ("Q00", "Q99", "XVII_congenital"),
    ("R00", "R99", "XVIII_symptoms_signs"),
    ("S00", "T88", "XIX_injury_poisoning"),
    ("U00", "U85", "XXII_special_purposes"),
    ("V00", "Y99", "XX_external_causes"),
    ("Z00", "Z99", "XXI_health_status_contact"),
]

CHAPTER_LABELS = {
    "I_infectious": "Certain infectious and parasitic diseases",
    "II_neoplasms": "Neoplasms",
    "III_blood_immune": "Blood and immune mechanism",
    "IV_endocrine_metabolic": "Endocrine, nutritional and metabolic",
    "V_mental_behavioural": "Mental and behavioural disorders",
    "VI_nervous_system": "Nervous system",
    "VII_eye_adnexa": "Eye and adnexa",
    "VIII_ear_mastoid": "Ear and mastoid process",
    "IX_circulatory": "Circulatory system",
    "X_respiratory": "Respiratory system",
    "XI_digestive": "Digestive system",
    "XII_skin_subcutaneous": "Skin and subcutaneous tissue",
    "XIII_musculoskeletal": "Musculoskeletal and connective tissue",
    "XIV_genitourinary": "Genitourinary system",
    "XV_pregnancy_childbirth": "Pregnancy, childbirth and the puerperium",
    "XVI_perinatal": "Perinatal conditions",
    "XVII_congenital": "Congenital malformations",
    "XVIII_symptoms_signs": "Symptoms, signs and abnormal findings",
    "XIX_injury_poisoning": "Injury, poisoning and external causes",
    "XX_external_causes": "External causes of morbidity",
    "XXI_health_status_contact": "Factors influencing health status",
    "XXII_special_purposes": "Codes for special purposes",
}


def icd_chapter(code: str) -> str:
    """ICD-10 chapter by three-character prefix (Z51.1 is filed under neoplasms, as in the peer map)."""
    code = str(code).strip().upper()
    if code.startswith("Z51.1"):
        return "II_neoplasms"
    head = code[:3]
    for lo, hi, chapter in _CHAPTER_RANGES:
        if lo <= head <= hi:
            return chapter
    return "UNMAPPED"


#: ICD-10-CM codes used by the generator: code → description. Chapters derive from icd_chapter.
ICD10: dict[str, str] = {
    # infectious
    "A09": "Infectious gastroenteritis and colitis, unspecified",
    "A41.9": "Sepsis, unspecified organism",
    "B07.9": "Viral wart, unspecified",
    "B34.9": "Viral infection, unspecified",
    "U07.1": "COVID-19",
    # neoplasms
    "C50.911": "Malignant neoplasm of unspecified site of right female breast",
    "C18.9": "Malignant neoplasm of colon, unspecified",
    "D23.9": "Other benign neoplasm of skin, unspecified",
    # blood
    "D50.9": "Iron deficiency anaemia, unspecified",
    "E04.1": "Nontoxic single thyroid nodule",
    "D62": "Acute posthaemorrhagic anaemia",
    # endocrine
    "E03.9": "Hypothyroidism, unspecified",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E11.65": "Type 2 diabetes mellitus with hyperglycaemia",
    "E55.9": "Vitamin D deficiency, unspecified",
    "E66.9": "Obesity, unspecified",
    "E78.5": "Hyperlipidaemia, unspecified",
    "E86.0": "Dehydration",
    "E87.1": "Hypo-osmolality and hyponatraemia",
    "E87.6": "Hypokalaemia",
    # mental
    "F32.9": "Major depressive disorder, single episode, unspecified",
    "F41.1": "Generalised anxiety disorder",
    "F90.0": "Attention-deficit hyperactivity disorder, predominantly inattentive type",
    "F17.210": "Nicotine dependence, cigarettes, uncomplicated",
    # nervous
    "G40.909": "Epilepsy, unspecified, not intractable",
    "G43.909": "Migraine, unspecified, not intractable",
    "G47.00": "Insomnia, unspecified",
    # eye / ear
    "H10.9": "Unspecified conjunctivitis",
    "H25.11": "Age-related nuclear cataract, right eye",
    "H25.12": "Age-related nuclear cataract, left eye",
    "H61.23": "Impacted cerumen, bilateral",
    "H66.90": "Otitis media, unspecified, unspecified ear",
    # circulatory
    "I10": "Essential (primary) hypertension",
    "I20.9": "Angina pectoris, unspecified",
    "I21.4": "Non-ST elevation (NSTEMI) myocardial infarction",
    "I25.10": "Atherosclerotic heart disease of native coronary artery",
    "I48.91": "Unspecified atrial fibrillation",
    "I50.9": "Heart failure, unspecified",
    "I50.23": "Acute on chronic systolic heart failure",
    "I63.9": "Cerebral infarction, unspecified",
    # respiratory
    "J02.9": "Acute pharyngitis, unspecified",
    "J03.90": "Acute tonsillitis, unspecified",
    "J06.9": "Acute upper respiratory infection, unspecified",
    "J18.9": "Pneumonia, unspecified organism",
    "J20.9": "Acute bronchitis, unspecified",
    "J30.9": "Allergic rhinitis, unspecified",
    "J34.2": "Deviated nasal septum",
    "J44.1": "Chronic obstructive pulmonary disease with acute exacerbation",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "J45.901": "Unspecified asthma with acute exacerbation",
    "J45.909": "Unspecified asthma, uncomplicated",
    "J96.00": "Acute respiratory failure, unspecified",
    # digestive
    "K21.9": "Gastro-oesophageal reflux disease without oesophagitis",
    "K29.70": "Gastritis, unspecified, without bleeding",
    "K35.80": "Unspecified acute appendicitis",
    "K40.90": "Unilateral inguinal hernia, without obstruction or gangrene",
    "K50.90": "Crohn's disease, unspecified, without complications",
    "K59.00": "Constipation, unspecified",
    "K80.20": "Calculus of gallbladder without cholecystitis, without obstruction",
    # skin
    "L02.91": "Cutaneous abscess, unspecified",
    "L03.115": "Cellulitis of right lower limb",
    "L30.9": "Dermatitis, unspecified",
    "L70.0": "Acne vulgaris",
    "L91.8": "Other hypertrophic disorders of the skin (skin tag)",
    # musculoskeletal
    "M05.79": "Rheumatoid arthritis with rheumatoid factor of multiple sites",
    "M17.11": "Unilateral primary osteoarthritis, right knee",
    "M23.211": "Derangement of anterior horn of medial meniscus, right knee",
    "M25.561": "Pain in right knee",
    "M54.50": "Low back pain, unspecified",
    "M54.16": "Radiculopathy, lumbar region",
    "M75.101": "Unspecified rotator cuff tear, right shoulder",
    "M81.0": "Age-related osteoporosis without current pathological fracture",
    # genitourinary
    "N10": "Acute pyelonephritis",
    "N17.9": "Acute kidney failure, unspecified",
    "N18.30": "Chronic kidney disease, stage 3 unspecified",
    "N39.0": "Urinary tract infection, site not specified",
    "N20.0": "Calculus of kidney",
    "N40.0": "Benign prostatic hyperplasia without lower urinary tract symptoms",
    "N92.0": "Excessive and frequent menstruation with regular cycle",
    # pregnancy
    "O80": "Encounter for full-term uncomplicated delivery",
    "O82": "Encounter for caesarean delivery without indication",
    "O24.419": "Gestational diabetes mellitus in pregnancy, unspecified control",
    "O99.019": "Anaemia complicating pregnancy, unspecified trimester",
    "O13.9": "Gestational hypertension, unspecified trimester",
    # symptoms
    "R05.9": "Cough, unspecified",
    "R07.9": "Chest pain, unspecified",
    "R10.9": "Unspecified abdominal pain",
    "R50.9": "Fever, unspecified",
    "R51.9": "Headache, unspecified",
    "R65.20": "Severe sepsis without septic shock",
    # injury
    "S61.411A": "Laceration without foreign body of right hand, initial encounter",
    "S72.001A": "Fracture of unspecified part of neck of right femur, initial encounter",
    "S93.401A": "Sprain of unspecified ligament of right ankle, initial encounter",
    # health status
    "Z00.00": "Encounter for general adult medical examination without abnormal findings",
    "Z23": "Encounter for immunisation",
    "Z12.31": "Encounter for screening mammogram for malignant neoplasm of breast",
    "Z34.90": "Encounter for supervision of normal pregnancy, unspecified trimester",
    "Z79.4": "Long term (current) use of insulin",
    "Z87.891": "Personal history of nicotine dependence",
    "Z41.1": "Encounter for cosmetic surgery",
}

# ----------------------------------------------------------------------- activity codes

# (code, description, activity_type, service_family, code_family, price_aed, extra)
_R = dict  # shorthand


def _codes() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(code, desc, atype, family, cfam, price, *, level=None, sex=None, age_min=0, age_max=120,
            minutes=None, time_based=False, specialty=None, facilities="CLINIC;HOSPITAL",
            complexity="LOW", tele=False, scarce=False, implant=False):
        rows.append({
            "activity_code": code, "description": desc, "activity_type": atype,
            "service_family": family, "code_family": cfam, "code_level": level,
            "sex_restriction": sex, "age_min": age_min, "age_max": age_max,
            "minutes": minutes, "is_time_based": bool(time_based), "specialty_required": specialty,
            "facility_types": facilities, "complexity": complexity,
            "is_telehealth_eligible": bool(tele), "is_scarce_equipment": bool(scarce),
            "is_implant": bool(implant), "unit_price_reference": float(price),
        })

    # --- evaluation and management: office, new and established (levels 1-5)
    add("99201", "Office visit, new patient, level 1 (DELETED 2021)", "CPT", "CONSULTATION",
        "EM_OFFICE_NEW", 120, level=1, minutes=10)
    for code, lvl, mins, price, cx in (("99202", 2, 20, 180, "LOW"), ("99203", 3, 30, 260, "LOW"),
                                       ("99204", 4, 45, 380, "MODERATE"), ("99205", 5, 60, 520, "HIGH")):
        add(code, f"Office or other outpatient visit, new patient, level {lvl}", "CPT", "CONSULTATION",
            "EM_OFFICE_NEW", price, level=lvl, minutes=mins, complexity=cx, tele=lvl <= 4)
    for code, lvl, mins, price, cx in (("99211", 1, 5, 90, "MINIMAL"), ("99212", 2, 10, 150, "LOW"),
                                       ("99213", 3, 20, 230, "LOW"), ("99214", 4, 30, 340, "MODERATE"),
                                       ("99215", 5, 40, 480, "HIGH")):
        add(code, f"Office or other outpatient visit, established patient, level {lvl}", "CPT",
            "CONSULTATION", "EM_OFFICE_EST", price, level=lvl, minutes=mins, complexity=cx,
            tele=2 <= lvl <= 4)
    for code, lvl, mins, price, cx in (("99242", 2, 20, 300, "LOW"), ("99243", 3, 30, 400, "LOW"),
                                       ("99244", 4, 40, 550, "MODERATE"), ("99245", 5, 55, 700, "HIGH")):
        add(code, f"Office consultation (specialist, on referral), level {lvl}", "CPT", "CONSULTATION",
            "EM_CONSULT", price, level=lvl, minutes=mins, complexity=cx, specialty="SPECIALIST")
    for code, lvl, price, cx in (("99281", 1, 150, "MINIMAL"), ("99282", 2, 250, "LOW"),
                                 ("99283", 3, 400, "MODERATE"), ("99284", 4, 650, "HIGH"),
                                 ("99285", 5, 950, "HIGH")):
        add(code, f"Emergency department visit, level {lvl}", "CPT", "EMERGENCY", "EM_ED", price,
            level=lvl, minutes=10 * lvl, complexity=cx, facilities="HOSPITAL")
    for code, lvl, mins, price in (("99221", 1, 40, 400), ("99222", 2, 55, 550), ("99223", 3, 75, 750)):
        add(code, f"Initial hospital inpatient care, level {lvl}", "CPT", "INPATIENT",
            "EM_INPATIENT_INIT", price, level=lvl, minutes=mins, facilities="HOSPITAL",
            complexity=("LOW", "MODERATE", "HIGH")[lvl - 1])
    for code, lvl, mins, price in (("99231", 1, 25, 200), ("99232", 2, 35, 300), ("99233", 3, 50, 420)):
        add(code, f"Subsequent hospital inpatient care, level {lvl}", "CPT", "INPATIENT",
            "EM_INPATIENT_SUBSEQ", price, level=lvl, minutes=mins, facilities="HOSPITAL",
            complexity=("LOW", "MODERATE", "HIGH")[lvl - 1])
    add("99238", "Hospital discharge day management, 30 minutes or less", "CPT", "INPATIENT",
        "EM_DISCHARGE", 250, minutes=30, time_based=True, facilities="HOSPITAL")
    add("99239", "Hospital discharge day management, more than 30 minutes", "CPT", "INPATIENT",
        "EM_DISCHARGE", 350, minutes=45, time_based=True, facilities="HOSPITAL")
    add("99291", "Critical care, first 30-74 minutes", "CPT", "INPATIENT", "CRITICAL_CARE", 1100,
        minutes=60, time_based=True, facilities="HOSPITAL", complexity="HIGH")
    add("99292", "Critical care, each additional 30 minutes", "CPT", "INPATIENT", "CRITICAL_CARE", 500,
        minutes=30, time_based=True, facilities="HOSPITAL", complexity="HIGH")
    for code, mins, price in (("99441", 10, 80), ("99442", 20, 140), ("99443", 30, 200)):
        add(code, f"Telephone evaluation and management, {mins - 9}-{mins} minutes", "CPT", "TELEHEALTH",
            "TELEHEALTH", price, minutes=mins, time_based=True, tele=True)
    # --- mental health (time based)
    add("90791", "Psychiatric diagnostic evaluation", "CPT", "MENTAL_HEALTH", "PSYCHOTHERAPY", 600,
        minutes=60, time_based=True, complexity="MODERATE")
    for code, mins, price in (("90832", 30, 300), ("90834", 45, 420), ("90837", 60, 560)):
        add(code, f"Psychotherapy, {mins} minutes with patient", "CPT", "MENTAL_HEALTH", "PSYCHOTHERAPY",
            price, minutes=mins, time_based=True, tele=True)
    # --- physiotherapy
    add("97161", "Physical therapy evaluation, low complexity", "CPT", "PHYSIOTHERAPY", "PHYSIO", 250,
        minutes=20, complexity="LOW")
    add("97162", "Physical therapy evaluation, moderate complexity", "CPT", "PHYSIOTHERAPY", "PHYSIO", 320,
        minutes=30, complexity="MODERATE")
    for code, desc, price in (("97110", "Therapeutic exercises, each 15 minutes", 120),
                              ("97140", "Manual therapy techniques, each 15 minutes", 120),
                              ("97530", "Therapeutic activities, each 15 minutes", 130),
                              ("97035", "Ultrasound therapy, each 15 minutes", 70)):
        add(code, desc, "CPT", "PHYSIOTHERAPY", "PHYSIO", price, minutes=15, time_based=True)
    add("97014", "Electrical stimulation, unattended", "CPT", "PHYSIOTHERAPY", "PHYSIO", 60, minutes=15)
    # --- office procedures and diagnostics
    for code, desc, price, cfam, family, mins in (
        ("10060", "Incision and drainage of abscess, simple", 450, "PROC_MINOR", "PROCEDURE", 20),
        ("10061", "Incision and drainage of abscess, complicated", 750, "PROC_MINOR", "PROCEDURE", 30),
        ("12001", "Simple repair of superficial wound, 2.5 cm or less", 400, "PROC_MINOR", "PROCEDURE", 20),
        ("12002", "Simple repair of superficial wound, 2.6 to 7.5 cm", 550, "PROC_MINOR", "PROCEDURE", 25),
        ("11042", "Debridement, subcutaneous tissue, first 20 sq cm", 650, "PROC_MINOR", "PROCEDURE", 25),
        ("97597", "Debridement, open wound, selective, first 20 sq cm", 350, "PROC_MINOR", "PROCEDURE", 20),
        ("17110", "Destruction of benign lesions (warts), up to 14", 300, "PROC_MINOR", "PROCEDURE", 15),
        ("11200", "Removal of skin tags, up to 15 lesions", 350, "PROC_MINOR", "PROCEDURE", 15),
        ("69210", "Removal of impacted cerumen, instrumentation, unilateral", 180, "PROC_MINOR", "PROCEDURE", 10),
        ("20610", "Arthrocentesis/injection, major joint, without ultrasound", 550, "PROC_ORTHO", "PROCEDURE", 15),
        ("20611", "Arthrocentesis/injection, major joint, with ultrasound guidance", 750, "PROC_ORTHO", "PROCEDURE", 20),
        ("36415", "Collection of venous blood by venipuncture", 25, "SPECIMEN", "LAB", 5),
        ("96372", "Therapeutic injection, subcutaneous or intramuscular", 60, "INJECTION", "PROCEDURE", 5),
        ("96374", "Therapeutic injection, intravenous push, single drug", 120, "INFUSION", "PROCEDURE", 10),
        ("96360", "Intravenous hydration, initial 31 minutes to 1 hour", 250, "INFUSION", "PROCEDURE", 60),
        ("96365", "Intravenous infusion, therapeutic, initial up to 1 hour", 350, "INFUSION", "PROCEDURE", 60),
        ("96366", "Intravenous infusion, therapeutic, each additional hour", 150, "INFUSION", "PROCEDURE", 60),
        ("96413", "Chemotherapy administration, intravenous infusion, up to 1 hour", 900, "CHEMO", "PROCEDURE", 60),
        ("96415", "Chemotherapy administration, intravenous infusion, each additional hour", 300, "CHEMO", "PROCEDURE", 60),
        ("94640", "Pressurised or nonpressurised inhalation treatment (nebuliser)", 90, "DIAG_OFFICE", "PROCEDURE", 15),
        ("94010", "Spirometry", 220, "DIAG_OFFICE", "PROCEDURE", 20),
        ("93000", "Electrocardiogram, routine, with interpretation and report", 150, "DIAG_OFFICE", "PROCEDURE", 10),
        ("93005", "Electrocardiogram, tracing only", 80, "DIAG_OFFICE", "PROCEDURE", 10),
        ("93010", "Electrocardiogram, interpretation and report only", 70, "DIAG_OFFICE", "PROCEDURE", 5),
        ("93306", "Transthoracic echocardiography, complete, with Doppler", 1100, "ECHO", "PROCEDURE", 40),
        ("95816", "Electroencephalogram, awake and drowsy", 900, "NEURODIAG", "PROCEDURE", 60),
        ("90471", "Immunisation administration, single vaccine", 50, "INJECTION", "PROCEDURE", 5),
        ("90686", "Influenza virus vaccine, quadrivalent, split virus, preservative free", 90, "VACCINE", "PROCEDURE", 0),
        ("99152", "Moderate sedation, initial 15 minutes", 250, "SEDATION", "PROCEDURE", 15),
    ):
        add(code, desc, "CPT", family, cfam, price, minutes=mins,
            facilities="CLINIC;HOSPITAL;DIAGNOSTIC_LAB" if code == "36415" else "CLINIC;HOSPITAL",
            time_based=code in ("96360", "96365", "96366", "96413", "96415", "99152"),
            complexity="MODERATE" if price >= 500 else "LOW")
    # --- endoscopy
    for code, desc, price in (("43235", "Upper GI endoscopy, diagnostic", 2200),
                              ("43239", "Upper GI endoscopy with biopsy", 2600),
                              ("45378", "Colonoscopy, diagnostic", 2800),
                              ("45380", "Colonoscopy with biopsy", 3200)):
        add(code, desc, "CPT", "DAY_SURGERY", "ENDOSCOPY", price, minutes=30, facilities="HOSPITAL",
            complexity="MODERATE")
    # --- surgery
    for code, desc, price, cfam, sex, age_min, mins, cx in (
        ("44970", "Laparoscopic appendectomy", 7500, "SURG_ABDOMINAL", None, 3, 60, "HIGH"),
        ("47562", "Laparoscopic cholecystectomy", 9000, "SURG_ABDOMINAL", None, 12, 90, "HIGH"),
        ("49320", "Diagnostic laparoscopy", 4000, "SURG_ABDOMINAL", None, 1, 45, "MODERATE"),
        ("49505", "Repair of initial inguinal hernia, age 5 years or over", 6000, "SURG_ABDOMINAL", None, 5, 60, "MODERATE"),
        ("19303", "Mastectomy, simple, complete", 14000, "SURG_BREAST", "F", 18, 120, "HIGH"),
        ("19318", "Breast reduction", 15000, "SURG_BREAST", "F", 18, 150, "HIGH"),
        ("59400", "Routine obstetric care incl. vaginal delivery (global)", 7000, "OBSTETRIC", "F", 15, 240, "HIGH"),
        ("59409", "Vaginal delivery only", 5000, "OBSTETRIC", "F", 15, 180, "HIGH"),
        ("59510", "Routine obstetric care incl. caesarean delivery (global)", 10000, "OBSTETRIC", "F", 15, 90, "HIGH"),
        ("59514", "Caesarean delivery only", 8000, "OBSTETRIC", "F", 15, 75, "HIGH"),
        ("66984", "Extracapsular cataract removal with intraocular lens insertion", 6500, "SURG_EYE", None, 40, 30, "MODERATE"),
        ("15823", "Blepharoplasty, upper eyelid, with excessive skin", 5000, "SURG_EYE", None, 18, 60, "MODERATE"),
        ("27447", "Total knee arthroplasty", 28000, "SURG_ORTHO", None, 40, 120, "HIGH"),
        ("27125", "Hemiarthroplasty, hip, partial", 24000, "SURG_ORTHO", None, 50, 120, "HIGH"),
        ("29881", "Knee arthroscopy with meniscectomy", 9000, "SURG_ORTHO", None, 16, 60, "HIGH"),
        ("29877", "Knee arthroscopy with debridement/chondroplasty", 7000, "SURG_ORTHO", None, 16, 50, "MODERATE"),
        ("93458", "Left heart catheterisation with coronary angiography", 9500, "CARDIAC_INVASIVE", None, 30, 60, "HIGH"),
        ("92920", "Percutaneous transluminal coronary angioplasty, single major artery", 18000, "CARDIAC_INVASIVE", None, 30, 90, "HIGH"),
        ("30520", "Septoplasty or submucous resection", 6000, "SURG_ENT", None, 16, 60, "MODERATE"),
    ):
        family = "MATERNITY" if cfam == "OBSTETRIC" else ("DAY_SURGERY" if code in ("15823", "30520", "29881", "29877", "66984") else "INPATIENT")
        add(code, desc, "CPT", family, cfam, price, sex=sex, age_min=age_min, minutes=mins,
            facilities="HOSPITAL", complexity=cx)
    for code, desc, price in (("15830", "Excision, excessive skin and subcutaneous tissue, abdomen (abdominoplasty)", 18000),
                              ("30400", "Rhinoplasty, primary, lateral and alar cartilages", 16000),
                              ("15780", "Dermabrasion, total face", 6000)):
        add(code, desc, "CPT", "COSMETIC", "COSMETIC", price, facilities="HOSPITAL", minutes=90)
    # --- anaesthesia (time based, 15-minute units)
    for code, desc in (("00840", "Anaesthesia for intraperitoneal procedures, lower abdomen"),
                       ("00790", "Anaesthesia for intraperitoneal procedures, upper abdomen"),
                       ("00832", "Anaesthesia for hernia repairs, lower abdomen"),
                       ("01402", "Anaesthesia for total knee arthroplasty"),
                       ("01230", "Anaesthesia for open procedures, upper two-thirds of femur"),
                       ("00142", "Anaesthesia for procedures on eye, lens surgery"),
                       ("01961", "Anaesthesia for caesarean delivery"),
                       ("01967", "Neuraxial labour analgesia"),
                       ("00400", "Anaesthesia for procedures on integumentary system of anterior trunk"),
                       ("01382", "Anaesthesia for diagnostic arthroscopic procedures of knee joint"),
                       ("00537", "Anaesthesia for cardiac catheterisation")):
        add(code, desc, "CPT", "INPATIENT", "ANAESTHESIA", 180, minutes=15, time_based=True,
            facilities="HOSPITAL", complexity="MODERATE")
    # --- laboratory
    labs = [
        ("80048", "Basic metabolic panel", 120, "LAB_PANEL"), ("80053", "Comprehensive metabolic panel", 180, "LAB_PANEL"),
        ("80061", "Lipid panel", 130, "LAB_PANEL"), ("80076", "Hepatic function panel", 130, "LAB_PANEL"),
        ("85025", "Complete blood count with automated differential", 70, "LAB_HAEM"),
        ("85027", "Complete blood count, automated", 55, "LAB_HAEM"), ("85048", "Leukocyte (WBC) count", 25, "LAB_HAEM"),
        ("85018", "Haemoglobin", 25, "LAB_HAEM"), ("85652", "Erythrocyte sedimentation rate", 40, "LAB_HAEM"),
        ("85610", "Prothrombin time", 60, "LAB_HAEM"),
        ("81001", "Urinalysis, automated, with microscopy", 40, "LAB_URINE"),
        ("81002", "Urinalysis, dipstick, non-automated, without microscopy", 25, "LAB_POC"),
        ("84443", "Thyroid stimulating hormone (TSH)", 110, "LAB_ENDO"), ("84439", "Thyroxine, free", 100, "LAB_ENDO"),
        ("83036", "Haemoglobin A1c", 110, "LAB_CHEM"), ("82947", "Glucose, quantitative, blood", 25, "LAB_CHEM"),
        ("82565", "Creatinine, blood", 30, "LAB_CHEM"), ("84132", "Potassium, serum", 25, "LAB_CHEM"),
        ("84295", "Sodium, serum", 25, "LAB_CHEM"), ("82435", "Chloride, blood", 25, "LAB_CHEM"),
        ("82374", "Carbon dioxide (bicarbonate)", 25, "LAB_CHEM"), ("84520", "Urea nitrogen, quantitative", 25, "LAB_CHEM"),
        ("82310", "Calcium, total", 30, "LAB_CHEM"), ("82040", "Albumin, serum", 25, "LAB_CHEM"),
        ("84155", "Protein, total, serum", 25, "LAB_CHEM"), ("82247", "Bilirubin, total", 30, "LAB_CHEM"),
        ("84075", "Phosphatase, alkaline", 30, "LAB_CHEM"), ("84460", "Transferase, alanine amino (ALT)", 35, "LAB_CHEM"),
        ("84450", "Transferase, aspartate amino (AST)", 35, "LAB_CHEM"), ("82465", "Cholesterol, serum, total", 35, "LAB_CHEM"),
        ("84478", "Triglycerides", 35, "LAB_CHEM"), ("83718", "Lipoprotein, HDL cholesterol", 40, "LAB_CHEM"),
        ("86140", "C-reactive protein", 80, "LAB_CHEM"), ("82306", "Vitamin D, 25 hydroxy", 250, "LAB_CHEM"),
        ("82607", "Cyanocobalamin (vitamin B12)", 150, "LAB_CHEM"), ("83540", "Iron", 60, "LAB_CHEM"),
        ("82728", "Ferritin", 120, "LAB_CHEM"), ("84153", "Prostate specific antigen, total", 200, "LAB_CHEM"),
        ("84484", "Troponin, quantitative", 220, "LAB_CHEM"), ("83880", "Natriuretic peptide (BNP)", 350, "LAB_CHEM"),
        ("84702", "Gonadotropin, chorionic (hCG), quantitative", 120, "LAB_ENDO"),
        ("87086", "Culture, bacterial, urine, quantitative colony count", 150, "LAB_MICRO"),
        ("87040", "Culture, bacterial, blood, aerobic", 180, "LAB_MICRO"),
        ("87880", "Infectious agent antigen detection, Streptococcus group A (rapid)", 90, "LAB_POC"),
        ("87635", "SARS-CoV-2 amplified probe technique", 150, "LAB_MICRO"),
        ("88305", "Surgical pathology, gross and microscopic examination, level IV", 350, "LAB_PATH"),
    ]
    for code, desc, price, cfam in labs:
        add(code, desc, "CPT", "LAB", cfam, price, facilities="CLINIC;HOSPITAL;DIAGNOSTIC_LAB",
            sex="M" if code == "84153" else ("F" if code == "84702" else None),
            age_min=40 if code == "84153" else (12 if code == "84702" else 0))
    # --- imaging
    imaging = [
        ("71045", "Radiologic examination, chest, single view", 120, "IMAGING_XR", "IMAGING", 10),
        ("71046", "Radiologic examination, chest, 2 views", 160, "IMAGING_XR", "IMAGING", 10),
        ("73030", "Radiologic examination, shoulder, minimum 2 views", 180, "IMAGING_XR", "IMAGING", 10),
        ("73562", "Radiologic examination, knee, 3 views", 180, "IMAGING_XR", "IMAGING", 10),
        ("73610", "Radiologic examination, ankle, minimum 3 views", 170, "IMAGING_XR", "IMAGING", 10),
        ("72100", "Radiologic examination, lumbosacral spine, 2 or 3 views", 220, "IMAGING_XR", "IMAGING", 15),
        ("73502", "Radiologic examination, hip, 2-3 views", 200, "IMAGING_XR", "IMAGING", 10),
        ("76700", "Ultrasound, abdominal, complete", 450, "IMAGING_US", "IMAGING", 20),
        ("76705", "Ultrasound, abdominal, limited", 300, "IMAGING_US", "IMAGING", 15),
        ("76856", "Ultrasound, pelvic (non-obstetric), complete", 400, "IMAGING_US", "IMAGING", 20),
        ("76801", "Ultrasound, pregnant uterus, first trimester", 450, "IMAGING_US", "IMAGING", 20),
        ("76805", "Ultrasound, pregnant uterus, after first trimester", 500, "IMAGING_US", "IMAGING", 25),
        ("76536", "Ultrasound, soft tissues of head and neck (thyroid)", 380, "IMAGING_US", "IMAGING", 20),
        ("77067", "Screening mammography, bilateral", 450, "IMAGING_MAMMO", "IMAGING", 20),
        ("77080", "Dual-energy X-ray absorptiometry (DEXA), axial skeleton", 350, "IMAGING_DEXA", "IMAGING", 15),
        ("70450", "CT head or brain, without contrast", 900, "IMAGING_CT", "ADVANCED_IMAGING", 20),
        ("71250", "CT thorax, without contrast", 1200, "IMAGING_CT", "ADVANCED_IMAGING", 20),
        ("71260", "CT thorax, with contrast", 1500, "IMAGING_CT", "ADVANCED_IMAGING", 25),
        ("74176", "CT abdomen and pelvis, without contrast", 1600, "IMAGING_CT", "ADVANCED_IMAGING", 25),
        ("74177", "CT abdomen and pelvis, with contrast", 1900, "IMAGING_CT", "ADVANCED_IMAGING", 30),
        ("70551", "MRI brain, without contrast", 2200, "IMAGING_MRI", "ADVANCED_IMAGING", 45),
        ("70552", "MRI brain, with contrast", 2600, "IMAGING_MRI", "ADVANCED_IMAGING", 50),
        ("70553", "MRI brain, without contrast followed by with contrast", 3000, "IMAGING_MRI", "ADVANCED_IMAGING", 60),
        ("72148", "MRI lumbar spine, without contrast", 2300, "IMAGING_MRI", "ADVANCED_IMAGING", 45),
        ("73721", "MRI any joint of lower extremity (knee), without contrast", 2200, "IMAGING_MRI", "ADVANCED_IMAGING", 45),
        ("73221", "MRI any joint of upper extremity (shoulder), without contrast", 2200, "IMAGING_MRI", "ADVANCED_IMAGING", 45),
        ("78815", "PET with concurrently acquired CT, skull base to mid-thigh", 7500, "IMAGING_NM", "ADVANCED_IMAGING", 90),
    ]
    for code, desc, price, cfam, family, mins in imaging:
        add(code, desc, "CPT", family, cfam, price, minutes=mins,
            facilities=("CLINIC;HOSPITAL;RADIOLOGY_CENTRE" if cfam == "IMAGING_US" else "HOSPITAL;RADIOLOGY_CENTRE"),
            scarce=family == "ADVANCED_IMAGING",
            sex="F" if code in ("76801", "76805", "76856", "77067") else None,
            age_min=12 if code in ("76801", "76805") else (35 if code == "77067" else 0),
            complexity="MODERATE" if family == "ADVANCED_IMAGING" else "LOW")
    # --- devices, implants and supplies
    for code, desc, price, cfam, implant, facilities in (
        ("V2632", "Posterior chamber intraocular lens", 1800, "IMPLANT", True, "HOSPITAL"),
        ("C1776", "Joint device (implantable), knee or hip", 15000, "IMPLANT", True, "HOSPITAL"),
        ("C1874", "Stent, coated/covered, with delivery system (drug-eluting)", 6500, "IMPLANT", True, "HOSPITAL"),
        ("C1781", "Mesh (implantable)", 1500, "IMPLANT", True, "HOSPITAL"),
        ("E0114", "Crutches, underarm, pair, with pads, tips and handgrips", 250, "DME", False, "CLINIC;HOSPITAL"),
        ("L1832", "Knee orthosis, adjustable knee joints, prefabricated", 900, "DME", False, "CLINIC;HOSPITAL"),
        ("E0570", "Nebuliser, with compressor", 450, "DME", False, "CLINIC;HOSPITAL;PHARMACY"),
        ("E0601", "Continuous positive airway pressure (CPAP) device", 3500, "DME", False, "CLINIC;HOSPITAL"),
        ("A4253", "Blood glucose test strips, per 50 strips", 120, "SUPPLY", False, "PHARMACY;CLINIC;HOSPITAL"),
    ):
        add(code, desc, "HCPCS", "DEVICE", cfam, price, implant=implant, facilities=facilities)
    # --- room and day codes
    for code, desc, price in (("RM-WARD", "Room and board, general ward, per day", 900),
                              ("RM-SEMI", "Room and board, semi-private, per day", 1200),
                              ("RM-PVT", "Room and board, private room, per day", 1600),
                              ("RM-ICU", "Intensive / coronary care unit, per day", 4500),
                              ("RM-DAYCARE", "Day-care bed, per admission", 600)):
        add(code, desc, "SERVICE", "INPATIENT", "ROOM_DAY", price, facilities="HOSPITAL")
    # --- case rates (DRG-like, three severity levels)
    for base, cr in CASE_RATES.items():
        for suffix, factor, label in (("A", 1.60, "with major complication or comorbidity"),
                                      ("B", 1.25, "with complication or comorbidity"),
                                      ("C", 1.00, "without complication or comorbidity")):
            add(f"{base}{suffix}", f"Case rate: {cr['description']}, {label}", "DRG",
                "DAY_SURGERY" if cr["day_case"] else "INPATIENT", "CASE_RATE",
                round(cr["base_price"] * factor, 2), level={"A": 3, "B": 2, "C": 1}[suffix],
                sex=cr.get("sex"), facilities="HOSPITAL",
                complexity={"A": "HIGH", "B": "MODERATE", "C": "LOW"}[suffix])
    # --- injectable drugs billed as HCPCS J-codes (also rows of drug_policy)
    for d in DRUGS:
        if d["product"].startswith("J"):
            add(d["product"], d["description"], "HCPCS",
                "SPECIALTY_DRUG" if d["is_high_cost"] else "PHARMACY", "DRUG_INJECTABLE",
                d["unit_price"], facilities="CLINIC;HOSPITAL;PHARMACY")
    return rows


#: Inpatient case-rate definitions. ``procedure`` is the principal procedure (None for medical).
CASE_RATES: dict[str, dict[str, Any]] = {
    "CR101": dict(description="Laparoscopic appendectomy", dx=["K35.80"], procedure="44970",
                  anaesthesia="00840", base_price=14000, los=(1, 3), day_case=False, specialty="GENERAL_SURGERY",
                  age=(5, 80), admission="EMERGENCY", imaging=["76705"], labs=["85025", "86140", "80048"],
                  drugs=["J0696", "RX1033"], room="RM-WARD", implant=None),
    "CR102": dict(description="Laparoscopic cholecystectomy", dx=["K80.20"], procedure="47562",
                  anaesthesia="00790", base_price=16000, los=(1, 2), day_case=False, specialty="GENERAL_SURGERY",
                  age=(20, 85), admission="ELECTIVE", imaging=["76700"], labs=["85025", "80053"],
                  drugs=["RX1033"], room="RM-WARD", implant=None),
    "CR103": dict(description="Inguinal hernia repair", dx=["K40.90"], procedure="49505",
                  anaesthesia="00832", base_price=11000, los=(0, 0), day_case=True, specialty="GENERAL_SURGERY",
                  age=(18, 85), admission="ELECTIVE", imaging=[], labs=["85025"], drugs=["RX1033"],
                  room="RM-DAYCARE", implant="C1781"),
    "CR104": dict(description="Vaginal delivery", dx=["O80"], procedure="59409", anaesthesia="01967",
                  base_price=10000, los=(1, 2), day_case=False, specialty="OBSTETRICS_GYNAECOLOGY",
                  age=(18, 44), sex="F", admission="EMERGENCY", imaging=[], labs=["85025"],
                  drugs=["RX1033"], room="RM-SEMI", implant=None),
    "CR105": dict(description="Caesarean delivery", dx=["O82"], procedure="59514", anaesthesia="01961",
                  base_price=16000, los=(2, 4), day_case=False, specialty="OBSTETRICS_GYNAECOLOGY",
                  age=(18, 44), sex="F", admission="ELECTIVE", imaging=[], labs=["85025"],
                  drugs=["J1650", "RX1033"], room="RM-SEMI", implant=None),
    "CR106": dict(description="Cataract extraction with lens implant", dx=["H25.11", "H25.12"], procedure="66984",
                  anaesthesia="00142", base_price=8500, los=(0, 0), day_case=True, specialty="OPHTHALMOLOGY",
                  age=(50, 90), admission="ELECTIVE", imaging=[], labs=[], drugs=[], room="RM-DAYCARE",
                  implant="V2632"),
    "CR107": dict(description="Total knee arthroplasty", dx=["M17.11"], procedure="27447", anaesthesia="01402",
                  base_price=38000, los=(3, 5), day_case=False, specialty="ORTHOPAEDICS", age=(50, 88),
                  admission="ELECTIVE", imaging=["73562"], labs=["85025", "80048"], drugs=["J1650", "RX1033"],
                  room="RM-SEMI", implant="C1776", physio=True),
    "CR108": dict(description="Hip fracture hemiarthroplasty", dx=["S72.001A"], procedure="27125",
                  anaesthesia="01230", base_price=36000, los=(5, 8), day_case=False, specialty="ORTHOPAEDICS",
                  age=(60, 95), admission="EMERGENCY", imaging=["73502", "71046"], labs=["85025", "80048", "85610"],
                  drugs=["J1650", "RX1033"], room="RM-WARD", implant="C1776", physio=True),
    "CR109": dict(description="Coronary angioplasty with stent", dx=["I21.4"], procedure="92920",
                  pre_procedure="93458", anaesthesia=None, base_price=32000, los=(3, 5), day_case=False,
                  specialty="CARDIOLOGY", age=(35, 90), admission="EMERGENCY", imaging=["71046"],
                  labs=["84484", "85025", "80048"], echo=True, drugs=["RX1026", "RX1025", "RX1017"],
                  room="RM-ICU", implant="C1874", needs="CATH_LAB"),
    "CR110": dict(description="Pneumonia, medical", dx=["J18.9"], procedure=None, anaesthesia=None,
                  base_price=9000, los=(3, 6), day_case=False, specialty="INTERNAL_MEDICINE", age=(1, 95),
                  admission="EMERGENCY", imaging=["71046"], labs=["85025", "80048", "86140", "87040"],
                  drugs=["J0696", "RX1006"], room="RM-WARD", implant=None),
    "CR111": dict(description="Heart failure, medical", dx=["I50.9"], procedure=None, anaesthesia=None,
                  base_price=11000, los=(4, 7), day_case=False, specialty="INTERNAL_MEDICINE", age=(45, 95),
                  admission="EMERGENCY", imaging=["71046"], labs=["83880", "80048", "85025"], echo=True,
                  drugs=["RX1028", "RX1024"], room="RM-WARD", implant=None),
    "CR112": dict(description="Ischaemic stroke, medical", dx=["I63.9"], procedure=None, anaesthesia=None,
                  base_price=16000, los=(5, 9), day_case=False, specialty="INTERNAL_MEDICINE", age=(45, 95),
                  admission="EMERGENCY", imaging=["70450", "70551"], labs=["85025", "80053", "80061"],
                  drugs=["RX1025", "RX1017"], room="RM-WARD", implant=None, needs="MRI"),
    "CR113": dict(description="Diabetes with hyperglycaemia, medical", dx=["E11.65"], procedure=None,
                  anaesthesia=None, base_price=8000, los=(2, 4), day_case=False, specialty="INTERNAL_MEDICINE",
                  age=(25, 90), admission="EMERGENCY", imaging=[], labs=["80048", "83036", "81001"],
                  drugs=["RX1010"], room="RM-WARD", implant=None),
    "CR114": dict(description="Acute pyelonephritis, medical", dx=["N10"], procedure=None, anaesthesia=None,
                  base_price=7000, los=(2, 4), day_case=False, specialty="INTERNAL_MEDICINE", age=(12, 90),
                  admission="EMERGENCY", imaging=["76700"], labs=["85025", "81001", "87086", "80048"],
                  drugs=["J0696", "RX1008"], room="RM-WARD", implant=None),
    "CR115": dict(description="Gastroenteritis with dehydration, medical", dx=["A09"], procedure=None,
                  anaesthesia=None, base_price=5000, los=(1, 3), day_case=False, specialty="INTERNAL_MEDICINE",
                  age=(1, 90), admission="EMERGENCY", imaging=[], labs=["85025", "80048"],
                  drugs=["J2405", "RX1059"], room="RM-WARD", implant=None, hydration=True),
    "CR116": dict(description="Asthma exacerbation, medical", dx=["J45.901"], procedure=None, anaesthesia=None,
                  base_price=6500, los=(1, 3), day_case=False, specialty="INTERNAL_MEDICINE", age=(4, 80),
                  admission="EMERGENCY", imaging=["71046"], labs=["85025"], drugs=["J1100", "RX1038"],
                  nebuliser=True, room="RM-WARD", implant=None),
    "CR117": dict(description="Cellulitis, medical", dx=["L03.115"], procedure=None, anaesthesia=None,
                  base_price=7000, los=(3, 5), day_case=False, specialty="INTERNAL_MEDICINE", age=(10, 90),
                  admission="EMERGENCY", imaging=[], labs=["85025", "86140", "80048"],
                  drugs=["J0696", "RX1004"], room="RM-WARD", implant=None),
    "CR118": dict(description="Chest pain, observation", dx=["R07.9"], procedure=None, anaesthesia=None,
                  base_price=4500, los=(1, 1), day_case=False, specialty="CARDIOLOGY", age=(25, 90),
                  admission="EMERGENCY", imaging=["71046"], labs=["84484", "85025", "80048"], echo=True,
                  drugs=["RX1025"], room="RM-WARD", implant=None),
    "CR119": dict(description="Knee arthroscopy with meniscectomy", dx=["M23.211"], procedure="29881",
                  anaesthesia="01382", base_price=12000, los=(0, 0), day_case=True, specialty="ORTHOPAEDICS",
                  age=(16, 70), admission="ELECTIVE", imaging=[], labs=[], drugs=["RX1033"],
                  room="RM-DAYCARE", implant=None),
    "CR120": dict(description="Simple mastectomy", dx=["C50.911"], procedure="19303", anaesthesia="00400",
                  base_price=22000, los=(2, 3), day_case=False, specialty="GENERAL_SURGERY", age=(30, 85),
                  sex="F", admission="ELECTIVE", imaging=["71046"], labs=["85025", "80053", "88305"],
                  drugs=["RX1033"], room="RM-SEMI", implant=None),
}

# ------------------------------------------------------------------------------ drugs

#: product, description, equivalence_group (molecule), therapeutic_class, strength_mg (per unit),
#: form, unit_price (AED per unit), max_duration_days, max_mg_per_kg_day, high_cost, controlled,
#: vial_size_mg, indication_prefixes, regimen (units/day, typical course days).
_DRUG_ROWS: list[tuple] = [
    ("RX1001", "Amoxicillin 500 mg capsule (generic)", "AMOXICILLIN", "ANTIBIOTIC", 500, "CAPSULE", 1.20, 14, 100, 0, 0, None, "J01;J02;J03;J20;H66;L02;L03;J18", (3, 7)),
    ("RX1002", "Amoxicillin 500 mg capsule (originator brand)", "AMOXICILLIN", "ANTIBIOTIC", 500, "CAPSULE", 2.60, 14, 100, 0, 0, None, "J01;J02;J03;J20;H66;L02;L03;J18", (3, 7)),
    ("RX1003", "Amoxicillin 250 mg/5 ml oral suspension, per ml", "AMOXICILLIN", "ANTIBIOTIC", 50, "SUSPENSION", 0.25, 14, 90, 0, 0, None, "J01;J02;J03;J20;H66;L02;L03;J18", None),
    ("RX1004", "Amoxicillin-clavulanate 625 mg tablet (generic)", "AMOXICILLIN_CLAVULANATE", "ANTIBIOTIC", 625, "TABLET", 3.50, 14, 90, 0, 0, None, "J01;J02;J03;J20;H66;L02;L03;J18", (2, 7)),
    ("RX1005", "Amoxicillin-clavulanate 625 mg tablet (originator brand)", "AMOXICILLIN_CLAVULANATE", "ANTIBIOTIC", 625, "TABLET", 6.80, 14, 90, 0, 0, None, "J01;J02;J03;J20;H66;L02;L03;J18", (2, 7)),
    ("RX1006", "Azithromycin 250 mg tablet", "AZITHROMYCIN", "ANTIBIOTIC", 250, "TABLET", 4.50, 5, 12, 0, 0, None, "J02;J03;J18;J20;A09", (2, 3)),
    ("RX1007", "Cefuroxime 500 mg tablet", "CEFUROXIME", "ANTIBIOTIC", 500, "TABLET", 5.00, 14, 30, 0, 0, None, "J01;J02;J03;J18;N39;L03", (2, 7)),
    ("RX1008", "Ciprofloxacin 500 mg tablet", "CIPROFLOXACIN", "ANTIBIOTIC", 500, "TABLET", 2.00, 14, 30, 0, 0, None, "N39;N10;A09", (2, 7)),
    ("RX1009", "Nitrofurantoin 100 mg capsule", "NITROFURANTOIN", "ANTIBIOTIC", 100, "CAPSULE", 1.80, 7, None, 0, 0, None, "N39", (2, 5)),
    ("RX1010", "Metformin 500 mg tablet", "METFORMIN", "BIGUANIDE", 500, "TABLET", 0.25, 90, None, 0, 0, None, "E11;O24;E28", (2, 30)),
    ("RX1011", "Metformin 850 mg tablet (generic)", "METFORMIN", "BIGUANIDE", 850, "TABLET", 0.35, 90, None, 0, 0, None, "E11;O24;E28", (2, 30)),
    ("RX1012", "Metformin 850 mg tablet (originator brand)", "METFORMIN", "BIGUANIDE", 850, "TABLET", 0.80, 90, None, 0, 0, None, "E11;O24;E28", (2, 30)),
    ("RX1013", "Gliclazide 60 mg modified-release tablet", "GLICLAZIDE", "SULFONYLUREA", 60, "TABLET", 0.90, 90, None, 0, 0, None, "E11", (1, 30)),
    ("RX1014", "Sitagliptin 100 mg tablet", "SITAGLIPTIN", "DPP4_INHIBITOR", 100, "TABLET", 7.50, 90, None, 0, 0, None, "E11", (1, 30)),
    ("RX1015", "Empagliflozin 10 mg tablet", "EMPAGLIFLOZIN", "SGLT2_INHIBITOR", 10, "TABLET", 6.50, 90, None, 0, 0, None, "E11;I50;N18", (1, 30)),
    ("RX1017", "Atorvastatin 20 mg tablet (generic)", "ATORVASTATIN", "STATIN", 20, "TABLET", 0.90, 90, None, 0, 0, None, "E78;I25;I63;I21;E11", (1, 30)),
    ("RX1018", "Atorvastatin 20 mg tablet (originator brand)", "ATORVASTATIN", "STATIN", 20, "TABLET", 2.90, 90, None, 0, 0, None, "E78;I25;I63;I21;E11", (1, 30)),
    ("RX1019", "Rosuvastatin 10 mg tablet", "ROSUVASTATIN", "STATIN", 10, "TABLET", 1.60, 90, None, 0, 0, None, "E78;I25;I63;E11", (1, 30)),
    ("RX1020", "Amlodipine 5 mg tablet (generic)", "AMLODIPINE", "CALCIUM_CHANNEL_BLOCKER", 5, "TABLET", 0.40, 90, None, 0, 0, None, "I10;I25;I20", (1, 30)),
    ("RX1021", "Amlodipine 5 mg tablet (originator brand)", "AMLODIPINE", "CALCIUM_CHANNEL_BLOCKER", 5, "TABLET", 1.50, 90, None, 0, 0, None, "I10;I25;I20", (1, 30)),
    ("RX1022", "Losartan 50 mg tablet", "LOSARTAN", "ARB", 50, "TABLET", 0.90, 90, None, 0, 0, None, "I10;N18;E11", (1, 30)),
    ("RX1023", "Valsartan 80 mg tablet", "VALSARTAN", "ARB", 80, "TABLET", 1.40, 90, None, 0, 0, None, "I10;I50", (1, 30)),
    ("RX1024", "Bisoprolol 5 mg tablet", "BISOPROLOL", "BETA_BLOCKER", 5, "TABLET", 0.80, 90, None, 0, 0, None, "I10;I25;I48;I50;I20", (1, 30)),
    ("RX1025", "Aspirin 81 mg enteric-coated tablet", "ASPIRIN", "ANTIPLATELET", 81, "TABLET", 0.10, 90, None, 0, 0, None, "I25;I63;I21;I20;R07", (1, 30)),
    ("RX1026", "Clopidogrel 75 mg tablet (generic)", "CLOPIDOGREL", "ANTIPLATELET", 75, "TABLET", 1.50, 90, None, 0, 0, None, "I25;I21;I63", (1, 30)),
    ("RX1027", "Clopidogrel 75 mg tablet (originator brand)", "CLOPIDOGREL", "ANTIPLATELET", 75, "TABLET", 5.50, 90, None, 0, 0, None, "I25;I21;I63", (1, 30)),
    ("RX1028", "Furosemide 40 mg tablet", "FUROSEMIDE", "LOOP_DIURETIC", 40, "TABLET", 0.20, 90, None, 0, 0, None, "I50;N18;I11", (1, 30)),
    ("RX1029", "Omeprazole 20 mg capsule", "OMEPRAZOLE", "PPI", 20, "CAPSULE", 0.60, 56, None, 0, 0, None, "K21;K29;K25;K30", (1, 28)),
    ("RX1030", "Esomeprazole 40 mg tablet (generic)", "ESOMEPRAZOLE", "PPI", 40, "TABLET", 1.20, 56, None, 0, 0, None, "K21;K29;K25", (1, 28)),
    ("RX1031", "Esomeprazole 40 mg tablet (originator brand)", "ESOMEPRAZOLE", "PPI", 40, "TABLET", 4.20, 56, None, 0, 0, None, "K21;K29;K25", (1, 28)),
    ("RX1032", "Pantoprazole 40 mg tablet", "PANTOPRAZOLE", "PPI", 40, "TABLET", 1.00, 56, None, 0, 0, None, "K21;K29;K25", (1, 28)),
    ("RX1033", "Paracetamol 500 mg tablet", "PARACETAMOL", "ANALGESIC", 500, "TABLET", 0.10, 30, 75, 0, 0, None, "", (4, 5)),
    ("RX1034", "Paracetamol 120 mg/5 ml oral syrup, per ml", "PARACETAMOL", "ANALGESIC", 24, "SYRUP", 0.08, 7, 75, 0, 0, None, "", None),
    ("RX1035", "Ibuprofen 400 mg tablet", "IBUPROFEN", "NSAID", 400, "TABLET", 0.30, 14, 40, 0, 0, None, "M;S;R51;G43;J02;J03;J06;N92;N20;R50;R10", (3, 5)),
    ("RX1036", "Ibuprofen 100 mg/5 ml oral suspension, per ml", "IBUPROFEN", "NSAID", 20, "SUSPENSION", 0.10, 7, 40, 0, 0, None, "M;S;R51;J02;J03;J06;H66;R50;N20", None),
    ("RX1037", "Diclofenac sodium 50 mg tablet", "DICLOFENAC", "NSAID", 50, "TABLET", 0.40, 14, 3, 0, 0, None, "M;S", (2, 7)),
    ("RX1038", "Salbutamol 100 mcg metered-dose inhaler, 200 doses", "SALBUTAMOL", "BRONCHODILATOR", 20, "INHALER", 18.0, 90, None, 0, 0, None, "J45;J44;J20", None),
    ("RX1039", "Fluticasone/salmeterol 250/50 mcg inhaler, 60 doses", "FLUTICASONE_SALMETEROL", "ICS_LABA", 18, "INHALER", 95.0, 90, None, 0, 0, None, "J45;J44", None),
    ("RX1040", "Montelukast 10 mg tablet", "MONTELUKAST", "LEUKOTRIENE_ANTAGONIST", 10, "TABLET", 1.20, 90, None, 0, 0, None, "J45;J30", (1, 30)),
    ("RX1041", "Cetirizine 10 mg tablet", "CETIRIZINE", "ANTIHISTAMINE", 10, "TABLET", 0.30, 30, None, 0, 0, None, "J30;L50;J06;L30;R05;J20", (1, 10)),
    ("RX1042", "Loratadine 10 mg tablet", "LORATADINE", "ANTIHISTAMINE", 10, "TABLET", 0.35, 30, None, 0, 0, None, "J30;L50;J06;L30", (1, 10)),
    ("RX1043", "Levothyroxine 50 mcg tablet", "LEVOTHYROXINE", "THYROID_HORMONE", 0.05, "TABLET", 0.30, 90, None, 0, 0, None, "E03;E89", (1, 30)),
    ("RX1044", "Colecalciferol 50,000 IU capsule", "COLECALCIFEROL", "VITAMIN", 1.25, "CAPSULE", 3.00, 84, None, 0, 0, None, "E55;M81", None),
    ("RX1045", "Ferrous sulfate 200 mg tablet", "FERROUS_SULFATE", "IRON", 200, "TABLET", 0.15, 90, None, 0, 0, None, "D50;O99;Z34", (1, 30)),
    ("RX1046", "Folic acid 5 mg tablet", "FOLIC_ACID", "VITAMIN", 5, "TABLET", 0.10, 90, None, 0, 0, None, "Z34;D52;O99;D50", (1, 30)),
    ("RX1047", "Tramadol 50 mg capsule", "TRAMADOL", "OPIOID_ANALGESIC", 50, "CAPSULE", 0.90, 7, 8, 0, 1, None, "M;S;K35;K80", (3, 5)),
    ("RX1048", "Pregabalin 75 mg capsule", "PREGABALIN", "GABAPENTINOID", 75, "CAPSULE", 2.50, 30, None, 0, 1, None, "M54;G62;M79", (2, 30)),
    ("RX1049", "Alprazolam 0.5 mg tablet", "ALPRAZOLAM", "BENZODIAZEPINE", 0.5, "TABLET", 0.50, 14, None, 0, 1, None, "F41", (2, 14)),
    ("RX1050", "Zolpidem 10 mg tablet", "ZOLPIDEM", "HYPNOTIC", 10, "TABLET", 1.10, 28, None, 0, 1, None, "G47;F51", (1, 28)),
    ("RX1051", "Methylphenidate 10 mg tablet", "METHYLPHENIDATE", "STIMULANT", 10, "TABLET", 2.20, 30, 2.0, 0, 1, None, "F90", (2, 30)),
    ("RX1052", "Sertraline 50 mg tablet", "SERTRALINE", "SSRI", 50, "TABLET", 1.00, 90, None, 0, 0, None, "F32;F41;F33", (1, 30)),
    ("RX1053", "Escitalopram 10 mg tablet", "ESCITALOPRAM", "SSRI", 10, "TABLET", 1.30, 90, None, 0, 0, None, "F32;F41;F33", (1, 30)),
    ("RX1054", "Prednisolone 5 mg tablet", "PREDNISOLONE", "CORTICOSTEROID", 5, "TABLET", 0.20, 14, 2, 0, 0, None, "J45;M05;L30;J44", (6, 5)),
    ("RX1055", "Hydrocortisone 1% cream, 30 g tube", "HYDROCORTISONE_TOPICAL", "TOPICAL_STEROID", 300, "CREAM", 12.0, 30, None, 0, 0, None, "L30;L20;L50", None),
    ("RX1056", "Mupirocin 2% ointment, 15 g tube", "MUPIROCIN", "TOPICAL_ANTIBIOTIC", 300, "OINTMENT", 22.0, 10, None, 0, 0, None, "L01;L02;L08;S61", None),
    ("RX1057", "Ondansetron 4 mg orally disintegrating tablet", "ONDANSETRON", "ANTIEMETIC", 4, "TABLET", 3.00, 5, 0.45, 0, 0, None, "A09;R11;K52", (2, 3)),
    ("RX1058", "Domperidone 10 mg tablet", "DOMPERIDONE", "ANTIEMETIC", 10, "TABLET", 0.40, 7, None, 0, 0, None, "R11;R10;K30;K21;K29;A09", (3, 5)),
    ("RX1059", "Oral rehydration salts, sachet", "ORAL_REHYDRATION", "ELECTROLYTE", 0, "SACHET", 1.00, 7, None, 0, 0, None, "A09;E86;K52", (3, 3)),
    ("RX1060", "Methotrexate 2.5 mg tablet", "METHOTREXATE", "DMARD", 2.5, "TABLET", 0.60, 90, None, 0, 0, None, "M05;L40", (6, 28)),
    ("RX1061", "Tamsulosin 0.4 mg capsule", "TAMSULOSIN", "ALPHA_BLOCKER", 0.4, "CAPSULE", 1.30, 90, None, 0, 0, None, "N40", (1, 30)),
    ("RX1062", "Isotretinoin 20 mg capsule", "ISOTRETINOIN", "RETINOID", 20, "CAPSULE", 4.50, 30, 1.0, 0, 0, None, "L70", (1, 30)),
    ("RX1063", "Nitroglycerin 0.4 mg sublingual tablet", "NITROGLYCERIN", "NITRATE", 0.4, "TABLET", 0.90, 90, None, 0, 0, None, "I20;I25", None),
    # injectables billed as HCPCS J-codes: strength_mg is per BILLING UNIT; vial_size_mg per vial/syringe
    ("J1745", "Infliximab, per 10 mg", "INFLIXIMAB", "TNF_INHIBITOR", 10, "VIAL", 160.0, 56, 10, 1, 0, 100, "K50;K51;M05;L40", None),
    ("J0135", "Adalimumab, per 20 mg", "ADALIMUMAB", "TNF_INHIBITOR", 20, "PREFILLED_SYRINGE", 1100.0, 28, None, 1, 0, 40, "M05;L40;K50", None),
    ("J9355", "Trastuzumab, per 10 mg", "TRASTUZUMAB", "HER2_ANTIBODY", 10, "VIAL", 250.0, 21, 8, 1, 0, 150, "C50", None),
    ("J9312", "Rituximab, per 10 mg", "RITUXIMAB", "CD20_ANTIBODY", 10, "VIAL", 220.0, 28, None, 1, 0, 500, "C83;C85;C91;M05", None),
    ("J2506", "Pegfilgrastim, per 0.5 mg", "PEGFILGRASTIM", "G_CSF", 0.5, "PREFILLED_SYRINGE", 300.0, 21, None, 1, 0, 6, "C50;C18;C34;D70", None),
    ("J0897", "Denosumab, per 1 mg", "DENOSUMAB", "RANKL_INHIBITOR", 1, "PREFILLED_SYRINGE", 18.0, 180, None, 1, 0, 60, "M81", None),
    ("J1650", "Enoxaparin sodium, per 10 mg", "ENOXAPARIN", "ANTICOAGULANT", 10, "PREFILLED_SYRINGE", 9.0, 35, 2, 0, 0, 40, "I26;I80;Z79;S72;M17;O82;K35;K80", None),
    ("J0696", "Ceftriaxone sodium, per 250 mg", "CEFTRIAXONE", "ANTIBIOTIC", 250, "VIAL", 8.0, 14, 100, 0, 0, 1000, "J18;N10;L03;A41;K35;J02;J03", None),
    ("J1100", "Dexamethasone sodium phosphate, per 1 mg", "DEXAMETHASONE", "CORTICOSTEROID", 1, "VIAL", 2.0, 5, 0.6, 0, 0, 4, "J45;J44;L50;T78;J38", None),
    ("J1885", "Ketorolac tromethamine, per 15 mg", "KETOROLAC", "NSAID", 15, "VIAL", 6.0, 5, 2, 0, 0, 30, "M;S;N20;R10", None),
    ("J2405", "Ondansetron hydrochloride, per 1 mg", "ONDANSETRON", "ANTIEMETIC", 1, "VIAL", 3.0, 5, 0.45, 0, 0, 4, "A09;R11;Z51;K52", None),
]

DRUGS: list[dict[str, Any]] = [
    {
        "product": r[0], "description": r[1], "equivalence_group": r[2], "therapeutic_class": r[3],
        "strength_mg": float(r[4]), "form": r[5], "unit_price": float(r[6]),
        # chronic maintenance products (90-day ceiling in the table above) carry no duration limit
        "max_duration_days": None if int(r[7]) >= 90 else int(r[7]),
        "max_mg_per_kg_day": None if r[8] is None else float(r[8]), "is_high_cost": bool(r[9]),
        "is_controlled": bool(r[10]), "vial_size_mg": None if r[11] is None else float(r[11]),
        "indication_prefixes": r[12], "regimen": r[13],
    }
    for r in _DRUG_ROWS
]

CODES: list[dict[str, Any]] = []  # filled lazily by _codes() (needs DRUGS and CASE_RATES)


# ------------------------------------------------------------------------ policies

_PANELS = {
    "80048": ["82947", "82565", "84132", "84295", "82435", "82374", "84520", "82310"],
    "80053": ["82947", "82565", "84132", "84295", "82435", "82374", "84520", "82310",
              "82040", "84155", "82247", "84075", "84460", "84450"],
    "80061": ["82465", "84478", "83718"],
    "80076": ["82040", "82247", "84075", "84460", "84450", "84155"],
    "85025": ["85048", "85018"],
}

#: (column_1 comprehensive, column_2 component, edit_type, modifier_allowed)
_BUNDLING = [
    ("80053", "80048", "COMPREHENSIVE_COMPONENT", False), ("80053", "82947", "COMPREHENSIVE_COMPONENT", False),
    ("80053", "82565", "COMPREHENSIVE_COMPONENT", False), ("80053", "84460", "COMPREHENSIVE_COMPONENT", False),
    ("80053", "84450", "COMPREHENSIVE_COMPONENT", False), ("80053", "80076", "COMPREHENSIVE_COMPONENT", False),
    ("80048", "82947", "COMPREHENSIVE_COMPONENT", False), ("80048", "82565", "COMPREHENSIVE_COMPONENT", False),
    ("80048", "84132", "COMPREHENSIVE_COMPONENT", False), ("80048", "84295", "COMPREHENSIVE_COMPONENT", False),
    ("80061", "82465", "COMPREHENSIVE_COMPONENT", False), ("80061", "84478", "COMPREHENSIVE_COMPONENT", False),
    ("80061", "83718", "COMPREHENSIVE_COMPONENT", False), ("80076", "84460", "COMPREHENSIVE_COMPONENT", False),
    ("85025", "85027", "COMPREHENSIVE_COMPONENT", False), ("85025", "85048", "COMPREHENSIVE_COMPONENT", False),
    ("85025", "85018", "COMPREHENSIVE_COMPONENT", False),
    ("93000", "93005", "COMPREHENSIVE_COMPONENT", False), ("93000", "93010", "COMPREHENSIVE_COMPONENT", False),
    ("45380", "45378", "COMPREHENSIVE_COMPONENT", False), ("43239", "43235", "COMPREHENSIVE_COMPONENT", False),
    ("47562", "49320", "COMPREHENSIVE_COMPONENT", False), ("44970", "49320", "COMPREHENSIVE_COMPONENT", False),
    ("27447", "20610", "COMPREHENSIVE_COMPONENT", True), ("29881", "29877", "COMPREHENSIVE_COMPONENT", True),
    ("59400", "59409", "COMPREHENSIVE_COMPONENT", False), ("59510", "59514", "COMPREHENSIVE_COMPONENT", False),
    ("96365", "96372", "COMPREHENSIVE_COMPONENT", True), ("96365", "96374", "COMPREHENSIVE_COMPONENT", True),
    ("96413", "96365", "COMPREHENSIVE_COMPONENT", True), ("96413", "96372", "COMPREHENSIVE_COMPONENT", True),
    ("10061", "10060", "COMPREHENSIVE_COMPONENT", False), ("12002", "12001", "COMPREHENSIVE_COMPONENT", False),
    ("11042", "97597", "COMPREHENSIVE_COMPONENT", True), ("20611", "20610", "COMPREHENSIVE_COMPONENT", False),
    ("71046", "71045", "MUTUALLY_EXCLUSIVE", False), ("70553", "70551", "MUTUALLY_EXCLUSIVE", False),
    ("70553", "70552", "MUTUALLY_EXCLUSIVE", False), ("74177", "74176", "MUTUALLY_EXCLUSIVE", False),
    ("71260", "71250", "MUTUALLY_EXCLUSIVE", False), ("90837", "90834", "MUTUALLY_EXCLUSIVE", False),
    ("90834", "90832", "MUTUALLY_EXCLUSIVE", False), ("99215", "99211", "MUTUALLY_EXCLUSIVE", False),
    ("97140", "97530", "MUTUALLY_EXCLUSIVE", True),
]

_UNIT_MAX = {
    "36415": (2, "Two venipunctures per day at most"),
    "96372": (4, "Four separate injections per day"),
    "96366": (6, "Infusion rarely exceeds seven hours"),
    "96415": (6, "Chemotherapy infusion rarely exceeds seven hours"),
    "97110": (4, "One hour of therapeutic exercise per day"), "97140": (2, "Thirty minutes of manual therapy per day"),
    "97530": (3, "Forty-five minutes of therapeutic activity per day"), "97035": (2, "Two ultrasound units per day"),
    "99292": (6, "Critical care beyond four hours is exceptional"),
    "J1745": (100, "Infliximab 10 mg/kg for a 100 kg patient"), "J0135": (2, "One 40 mg dose"),
    "J9355": (84, "Trastuzumab 8 mg/kg loading dose for a 105 kg patient"),
    "J9312": (100, "Rituximab 375 mg/m2 upper bound"), "J2506": (12, "One 6 mg syringe"),
    "J0897": (60, "One 60 mg syringe"), "J1650": (10, "Treatment dose 1 mg/kg twice daily"),
    "J0696": (8, "Ceftriaxone 2 g per day"), "J1100": (12, "Dexamethasone 12 mg per day"),
    "J1885": (8, "Ketorolac 120 mg per day"), "J2405": (16, "Ondansetron 16 mg per day"),
}

_INDICATIONS = {
    "83036": "E11;E10;E13;R73;O24", "84153": "N40;R97;C61;Z12.5", "77080": "M81;M85;E55;Z13.82",
    "73721": "M17;M23;M25.56;S83", "72148": "M54;M51;M48", "73221": "M75;M25.51;S43",
    "70551": "G40;G43;I63;R51;G35;R42", "70552": "G40;G43;I63;R51;G35;C71", "70553": "G40;G43;I63;R51;G35;C71",
    "70450": "I63;R51;S06;G40;R55", "93306": "I50;I25;I21;I48;R07;I10;I35;R06", "84484": "R07;I21;I20;I25",
    "83880": "I50;R06;I11", "43235": "K21;K29;K25;K92;R10;D50", "43239": "K21;K29;K25;K92;R10;D50",
    "45378": "K57;K92;D50;R19;K50;K51;Z12", "45380": "K57;K92;D50;R19;K50;K51;Z12",
    "20610": "M17;M25;M19;M06;M05;M10", "66984": "H25;H26", "27447": "M17", "27125": "S72",
    "29881": "M23;S83", "44970": "K35", "47562": "K80;K81", "49505": "K40", "59400": "O80;O60;O70;Z37",
    "59409": "O80;O60;O70;Z37", "59510": "O82;O32;O34;O64;O68", "59514": "O82;O32;O34;O64;O68",
    "92920": "I21;I20;I25", "93458": "I21;I20;I25;R07", "90791": "F", "90832": "F", "90834": "F", "90837": "F",
    "97110": "M;S;G81;I69;Z96;Z47", "97140": "M;S;Z96;Z47", "97530": "M;S;G81;I69;Z96", "97161": "M;S;Z96;Z47",
    "97162": "M;S;Z96;Z47;I69", "87880": "J02;J03", "87635": "U07;J06;R50;J12;J18;R05", "17110": "B07",
    "69210": "H61", "76801": "Z34;O", "76805": "Z34;O", "76856": "N92;N83;N94;R10;N85", "77067": "Z12;N63;N64",
    "84702": "Z34;O;N92;N91", "10060": "L02;L03", "10061": "L02;L03", "12001": "S01;S41;S51;S61;S71;S81;S91",
    "12002": "S01;S41;S51;S61;S71;S81;S91", "15823": "H02.83;H02.4;H53.4", "30520": "J34.2", "19318": "N62;M54.2",
    "94010": "J44;J45;R06;R05", "94640": "J45;J44;J20;J21", "95816": "G40;R56;R40", "96413": "C;Z51",
    "96415": "C;Z51", "78815": "C;R91", "88305": "K;C;D;N;L", "73562": "M17;M23;M25.56;S83;S89",
    "72100": "M54;M51;M48;S32", "73030": "M75;M25.51;S43;S42", "74177": "R10;K35;K57;C;N20;K80",
    "76700": "R10;K80;K76;N10;K35;R74;N20", "76705": "R10;K35;K80;N10;N20;N18;N40;K21;E11", "93000": "R07;I10;I20;I21;I25;I48;I50;R00;Z01.8",
}

_PROHIBITED = [
    ("O80", "59510", "Uncomplicated vaginal delivery diagnosis with a caesarean-delivery procedure"),
    ("O80", "59514", "Uncomplicated vaginal delivery diagnosis with a caesarean-delivery procedure"),
    ("O82", "59400", "Caesarean-delivery diagnosis with a vaginal-delivery procedure"),
    ("O82", "59409", "Caesarean-delivery diagnosis with a vaginal-delivery procedure"),
    ("Z34", "74177", "Contrast CT of abdomen/pelvis in routine pregnancy supervision"),
    ("Z34", "74176", "CT of abdomen/pelvis in routine pregnancy supervision"),
    ("Z34", "71260", "Contrast CT thorax in routine pregnancy supervision"),
    ("Z34", "84153", "Prostate antigen with a pregnancy diagnosis"),
    ("N40", "76856", "Female pelvic ultrasound with benign prostatic hyperplasia"),
    ("N40", "77067", "Mammography with benign prostatic hyperplasia"),
    ("N40", "84702", "Pregnancy hormone with benign prostatic hyperplasia"),
    ("N40", "76801", "Obstetric ultrasound with benign prostatic hyperplasia"),
    ("N92", "84153", "Prostate antigen with a menstrual disorder"),
    ("K35", "47562", "Cholecystectomy billed against acute appendicitis"),
    ("K80", "44970", "Appendectomy billed against gallstones"),
    ("J06", "70551", "MRI brain for an uncomplicated upper respiratory infection"),
    ("J06", "70553", "MRI brain for an uncomplicated upper respiratory infection"),
    ("J06", "71250", "CT thorax for an uncomplicated upper respiratory infection"),
    ("J06", "78815", "PET-CT for an uncomplicated upper respiratory infection"),
    ("J30", "93306", "Echocardiography for allergic rhinitis"),
    ("L70", "70553", "MRI brain for acne"),
    ("R51", "73721", "Knee MRI for headache"),
    ("H25", "27447", "Knee arthroplasty against a cataract diagnosis"),
    ("Z23", "99285", "Highest-level emergency visit for a vaccination encounter"),
    ("B07", "11042", "Surgical debridement for a viral wart"),
]

_REPEAT = [
    ("83036", 80, True), ("80061", 150, True), ("84443", 35, True), ("82306", 80, True),
    ("84153", 300, True), ("77067", 300, True), ("77080", 600, True), ("73721", 180, True),
    ("72148", 180, True), ("73221", 180, True), ("70551", 180, True), ("70553", 180, True),
    ("93306", 150, True), ("45378", 1000, True), ("45380", 1000, True), ("43239", 180, True),
    ("94010", 150, True), ("80053", 28, True), ("82728", 60, True), ("78815", 90, True),
]

_PATHWAYS = [
    ("73721", "73562", 180), ("72148", "72100", 365), ("73221", "73030", 180),
    ("27447", "73562", 365), ("92920", "93458", 1), ("29881", "73721", 180),
]

_PAYMENT_POLICY = [
    ("READMISSION_BUNDLING_WINDOW_DAYS", "INPATIENT", 30,
     "A readmission to the same facility within 30 days of discharge for a related condition is paid "
     "within the original case rate, not as a new admission."),
    ("CONTINUOUS_STAY_TRANSFER_HOURS", "INPATIENT", 24,
     "A discharge followed by an admission within 24 hours (same or different facility) is one "
     "continuous stay for payment purposes."),
    ("DAY_CASE_MAX_LOS_DAYS", "DAY_SURGERY", 0, "Day-case packages assume same-day discharge."),
    ("CLAIM_SUBMISSION_LIMIT_DAYS", "ALL", 90, "Claims must be submitted within 90 days of discharge."),
    ("RESUBMISSION_LIMIT_DAYS", "ALL", 60, "A resubmission must follow the remittance within 60 days."),
    ("EARLY_REFILL_THRESHOLD", "PHARMACY", 0.8,
     "A refill is payable once 80% of the previous fill's days of supply has elapsed."),
    ("NEW_PATIENT_LOOKBACK_DAYS", "CONSULTATION", 1095,
     "A new-patient E&M code requires no visit to the same facility within three years."),
    ("TELEHEALTH_MAX_PER_DAY", "TELEHEALTH", 1, "One telehealth consultation per member per day."),
    ("PACKAGE_COMPONENT_PRICE", "INPATIENT", 0,
     "Components included in a case rate are reported at zero price."),
]

_DOC_REQUIREMENTS = [
    ("INPATIENT", None, "DISCHARGE_SUMMARY"), ("DAY_SURGERY", None, "DISCHARGE_SUMMARY"),
    ("INPATIENT", "27447", "OPERATIVE_NOTE"), ("INPATIENT", "27125", "OPERATIVE_NOTE"),
    ("INPATIENT", "92920", "OPERATIVE_NOTE"), ("INPATIENT", "44970", "OPERATIVE_NOTE"),
    ("INPATIENT", "47562", "OPERATIVE_NOTE"), ("INPATIENT", "19303", "OPERATIVE_NOTE"),
    ("DAY_SURGERY", "66984", "OPERATIVE_NOTE"), ("DAY_SURGERY", "29881", "OPERATIVE_NOTE"),
    ("LAB", None, "LAB_REPORT"), ("IMAGING", None, "IMAGING_REPORT"),
    ("ADVANCED_IMAGING", None, "IMAGING_REPORT"), ("ADVANCED_IMAGING", None, "CLINICAL_NOTE"),
    ("SPECIALTY_DRUG", None, "CLINICAL_NOTE"),
]

#: activity code → required terms (all must appear, case-insensitive) in some document on the claim.
MEDICAL_NECESSITY: dict[str, list[str]] = {
    "73721": ["knee pain", "radiograph", "conservative management"],
    "72148": ["low back pain", "radiculopathy", "conservative management"],
    "73221": ["shoulder pain", "radiograph", "rotator cuff"],
    "70551": ["neurological", "mri indicated"], "70553": ["neurological", "mri indicated"],
    "78815": ["malignancy", "staging"],
    "27447": ["osteoarthritis", "radiographic", "conservative management"],
    "29881": ["meniscal tear", "mri"],
    "92920": ["troponin", "angiography", "stenosis"],
    "66984": ["cataract", "visual acuity"],
    "J1745": ["crohn", "inadequate response"],
    "J0135": ["rheumatoid", "methotrexate"],
    "J9355": ["her2", "positive"],
    "J0897": ["osteoporosis", "fracture risk"],
}

_DISGUISE = [
    ("15823", "Cosmetic upper-eyelid surgery (blepharoplasty)", "Z41.1;L57;L90"),
    ("30520", "Cosmetic rhinoplasty", "Z41.1;M95.0;Q30"),
    ("19318", "Cosmetic breast reduction", "Z41.1;N65"),
    ("11200", "Cosmetic skin tag removal", "Z41.1;L57;L81"),
    ("17110", "Cosmetic lesion destruction", "Z41.1;L81;L82"),
    ("15780", "Cosmetic facial resurfacing", "Z41.1;L57;L90;L70"),
]

_EQUIVALENCE_EXTRA = {
    "CHEST_XRAY": ["71045", "71046"], "CBC": ["85025", "85027"], "MRI_BRAIN": ["70551", "70552", "70553"],
    "CT_ABDOMEN_PELVIS": ["74176", "74177"], "CT_THORAX": ["71250", "71260"], "UPPER_GI_ENDOSCOPY": ["43235", "43239"],
    "COLONOSCOPY": ["45378", "45380"], "ECG": ["93000", "93005", "93010"], "VAGINAL_DELIVERY": ["59400", "59409"],
    "CAESAREAN_DELIVERY": ["59510", "59514"], "WOUND_REPAIR": ["12001", "12002"],
    "ABSCESS_DRAINAGE": ["10060", "10061"], "ARTHROCENTESIS_MAJOR": ["20610", "20611"],
}

_STEP = [("RX1014", "METFORMIN", 365), ("RX1015", "METFORMIN", 365), ("RX1019", "ATORVASTATIN", 365),
         ("J0135", "METHOTREXATE", 365)]

#: Secondary diagnosis → (is_cc, is_mcc, severity_weight). Non-listed codes are neither.
DRG_GROUPER: dict[str, tuple[bool, bool, float]] = {
    "A41.9": (False, True, 0.80), "R65.20": (False, True, 0.80), "J96.00": (False, True, 0.70),
    "I50.23": (False, True, 0.60), "N17.9": (False, True, 0.50), "J18.9": (False, True, 0.50),
    "E87.1": (True, False, 0.25), "E87.6": (True, False, 0.15), "D62": (True, False, 0.25),
    "I48.91": (True, False, 0.20), "N39.0": (True, False, 0.20), "E86.0": (True, False, 0.15),
    "E11.65": (True, False, 0.15), "J44.1": (True, False, 0.30), "I50.9": (True, False, 0.30),
    "O24.419": (True, False, 0.20), "O13.9": (True, False, 0.20),
    "I10": (False, False, 0.0), "E11.9": (False, False, 0.0), "E78.5": (False, False, 0.0),
    "E03.9": (False, False, 0.0), "K21.9": (False, False, 0.0), "E66.9": (False, False, 0.0),
    "F17.210": (False, False, 0.0), "Z87.891": (False, False, 0.0), "J45.909": (False, False, 0.0),
    "N18.30": (False, False, 0.05), "I25.10": (False, False, 0.05), "Z79.4": (False, False, 0.0),
    "O99.019": (False, False, 0.05), "E55.9": (False, False, 0.0), "J44.9": (False, False, 0.05),
}

#: Acute secondary conditions (CC/MCC) by principal chapter: code → prior prevalence among admissions.
#: Chronic comorbidities are NOT drawn from here — they come from each member's own conditions —
#: and ``base.py`` recalibrates the whole morbidity_model table to the clean population once built.
ACUTE_SECONDARY: dict[str, dict[str, float]] = {
    "IX_circulatory": {"I48.91": 0.06, "N17.9": 0.025, "E87.1": 0.02, "I50.23": 0.015, "J18.9": 0.01},
    "X_respiratory": {"J96.00": 0.02, "E87.1": 0.03, "A41.9": 0.012, "N17.9": 0.015, "J44.1": 0.03},
    "XI_digestive": {"E86.0": 0.05, "E87.6": 0.03, "D62": 0.02, "A41.9": 0.006},
    "XIII_musculoskeletal": {"D62": 0.04, "E87.1": 0.015, "N39.0": 0.02},
    "XIX_injury_poisoning": {"D62": 0.06, "E87.1": 0.03, "N39.0": 0.03, "N17.9": 0.015},
    "XIV_genitourinary": {"E86.0": 0.04, "N17.9": 0.03, "A41.9": 0.012},
    "I_infectious": {"E86.0": 0.15, "E87.6": 0.05, "N17.9": 0.015},
    "IV_endocrine_metabolic": {"E86.0": 0.08, "E87.1": 0.04, "N17.9": 0.02},
    "XII_skin_subcutaneous": {"A41.9": 0.01, "E86.0": 0.02},
    "XV_pregnancy_childbirth": {"O24.419": 0.06, "O99.019": 0.08, "O13.9": 0.04, "D62": 0.03},
    "XVIII_symptoms_signs": {"E87.6": 0.02},
    "VII_eye_adnexa": {},
    "II_neoplasms": {"D62": 0.03, "E87.1": 0.02},
    "VI_nervous_system": {"E87.1": 0.02},
}


def _tables() -> dict[str, pd.DataFrame]:
    global CODES
    CODES = _codes()
    codes = pd.DataFrame(CODES)
    t: dict[str, pd.DataFrame] = {}
    t["activity_code_reference"] = codes

    # code_system_version: every activity code, every drug product and every ICD code
    csv_rows = []
    for c in CODES:
        system = {"CPT": "CPT", "HCPCS": "HCPCS", "DRG": "LOCAL_CASE_RATE", "SERVICE": "LOCAL_SERVICE"}[c["activity_type"]]
        valid_to = _dt.date(2020, 12, 31) if c["activity_code"] == "99201" else OPEN_END
        csv_rows.append({"code": c["activity_code"], "code_system": system, "valid_from": EPOCH, "valid_to": valid_to})
    for d in DRUGS:
        if not d["product"].startswith("J"):
            csv_rows.append({"code": d["product"], "code_system": "DRUG_CODE", "valid_from": EPOCH, "valid_to": OPEN_END})
    for code in ICD10:
        csv_rows.append({"code": code, "code_system": "ICD-10-CM", "valid_from": EPOCH, "valid_to": OPEN_END})
    t["code_system_version"] = pd.DataFrame(csv_rows)

    # unit maxima: explicit ones, then 1/day for E&M, imaging, labs, case rates, surgery, rooms
    um = [{"activity_code": k, "max_units_per_day": float(v[0]), "rationale": v[1]} for k, v in _UNIT_MAX.items()]
    for c in CODES:
        if c["activity_code"] in _UNIT_MAX:
            continue
        fam = c["code_family"]
        if fam.startswith("EM_") or fam in ("CASE_RATE", "ROOM_DAY") or c["service_family"] in (
                "LAB", "IMAGING", "ADVANCED_IMAGING", "DAY_SURGERY") or fam.startswith("SURG") or fam in (
                "OBSTETRIC", "CARDIAC_INVASIVE", "PSYCHOTHERAPY", "TELEHEALTH", "IMPLANT"):
            um.append({"activity_code": c["activity_code"], "max_units_per_day": 1.0,
                       "rationale": "One per day by definition of the code"})
        elif fam == "ANAESTHESIA":
            um.append({"activity_code": c["activity_code"], "max_units_per_day": 32.0,
                       "rationale": "Eight hours of anaesthesia time (15-minute units)"})
    t["unit_maximum_policy"] = pd.DataFrame(um)

    t["bundling_edit_table"] = pd.DataFrame([
        {"column_1_code": a, "column_2_code": b, "edit_type": e, "modifier_allowed": m,
         "valid_from": _dt.date(2020, 1, 1), "valid_to": OPEN_END} for a, b, e, m in _BUNDLING])

    pkg = []
    for base, cr in CASE_RATES.items():
        comps = [x for x in (cr.get("procedure"), cr.get("pre_procedure"), cr.get("anaesthesia"), cr["room"])
                 if x]
        comps += ["99221", "99222", "99223", "99231", "99232", "99233", "99238", "99239"]
        comps += list(cr.get("labs", [])) + list(cr.get("imaging", [])) + ["36415"]
        comps += [d for d in cr.get("drugs", []) if not _drug(d)["is_high_cost"]]
        if cr.get("echo"):
            comps.append("93306")
        if cr.get("physio"):
            comps += ["97110", "97161"]
        if cr.get("hydration"):
            comps.append("96360")
        if cr.get("nebuliser"):
            comps.append("94640")
        if cr["room"] == "RM-ICU":
            comps.append("RM-WARD")
        for suffix in "ABC":
            for comp in dict.fromkeys(comps):
                pkg.append({"package_code": f"{base}{suffix}", "component_code": comp, "expected_zero_price": True})
    t["package_definition"] = pd.DataFrame(pkg)

    t["panel_definition"] = pd.DataFrame([{"panel_code": p, "component_code": c}
                                          for p, comps in _PANELS.items() for c in comps])
    t["dx_proc_prohibition_table"] = pd.DataFrame(
        [{"diagnosis_prefix": d, "activity_code": a, "reason": r} for d, a, r in _PROHIBITED])
    ind = [{"activity_code": k, "allowed_diagnosis_prefixes": v} for k, v in _INDICATIONS.items()]
    for base, cr in CASE_RATES.items():
        for suffix in "ABC":
            ind.append({"activity_code": f"{base}{suffix}",
                        "allowed_diagnosis_prefixes": ";".join(sorted({d.split(".")[0] for d in cr["dx"]}))})
    t["indication_policy"] = pd.DataFrame(ind)
    t["repeat_interval_policy"] = pd.DataFrame(
        [{"activity_code": a, "min_interval_days": d, "requires_result_before_repeat": r} for a, d, r in _REPEAT])
    t["care_pathway_policy"] = pd.DataFrame(
        [{"activity_code": a, "required_prior_code": p, "within_days": d} for a, p, d in _PATHWAYS])
    t["payment_policy"] = pd.DataFrame(
        [{"policy_key": k, "service_family": f, "value": float(v), "description": d} for k, f, v, d in _PAYMENT_POLICY])
    t["document_requirement_policy"] = pd.DataFrame(
        [{"service_family": f, "activity_code": a, "doc_type": d} for f, a, d in _DOC_REQUIREMENTS])
    t["medical_necessity_policy"] = pd.DataFrame(
        [{"activity_code": k, "required_terms": ";".join(v)} for k, v in MEDICAL_NECESSITY.items()])
    t["disguise_risk_policy"] = pd.DataFrame(
        [{"activity_code": a, "excluded_service": e, "risk_diagnosis_prefixes": r} for a, e, r in _DISGUISE])
    t["drug_policy"] = pd.DataFrame([{k: v for k, v in d.items() if k != "regimen"} for d in DRUGS])
    eq = [{"code": d["product"], "equivalence_group": d["equivalence_group"]} for d in DRUGS]
    eq += [{"code": c, "equivalence_group": g} for g, cs in _EQUIVALENCE_EXTRA.items() for c in cs]
    t["code_equivalence_map"] = pd.DataFrame(eq)
    t["step_therapy_policy"] = pd.DataFrame(
        [{"product": p, "required_prior_group": g, "lookback_days": d} for p, g, d in _STEP])
    t["drg_grouper"] = pd.DataFrame(
        [{"diagnosis_code": k, "is_cc": v[0], "is_mcc": v[1], "severity_weight": v[2]} for k, v in DRG_GROUPER.items()])
    mm = [{"principal_chapter": ch, "secondary_code": code, "expected_prevalence": p}
          for ch, d in ACUTE_SECONDARY.items() for code, p in d.items()]
    t["morbidity_model"] = pd.DataFrame(mm)
    return t


def _drug(product: str) -> dict[str, Any]:
    for d in DRUGS:
        if d["product"] == product:
            return d
    raise KeyError(product)


def build_reference(world) -> None:
    """Create every reference table and leave the catalogues in ``world.context``."""
    tables = _tables()
    for name, frame in tables.items():
        world.tables[name] = frame.reset_index(drop=True)
    world.context["codes"] = {c["activity_code"]: c for c in CODES}
    world.context["drugs"] = {d["product"]: d for d in DRUGS}
    world.context["icd"] = {code: (desc, icd_chapter(code)) for code, desc in ICD10.items()}
    world.context["case_rates"] = CASE_RATES
    world.context["panels"] = _PANELS
    world.context["medical_necessity"] = MEDICAL_NECESSITY
    world.context["drg_grouper"] = DRG_GROUPER
    world.context["repeat_interval"] = {a: d for a, d, _ in _REPEAT}
    world.context["pathways"] = _PATHWAYS
    world.context["indications"] = {k: v.split(";") for k, v in _INDICATIONS.items()}
    world.context["prohibited"] = [(d, a) for d, a, _ in _PROHIBITED]
    world.context["bundling_pairs"] = {(a, b) for a, b, _, _ in _BUNDLING} | {(b, a) for a, b, _, _ in _BUNDLING}
    world.context["step_therapy"] = {p: (g, d) for p, g, d in _STEP}
