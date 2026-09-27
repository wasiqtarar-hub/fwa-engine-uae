SYNTHETIC DOCUMENT CORPUS
============================================================

Every file in this directory is GENERATED. None of it is a real clinical record, and none of it describes a real person, provider or event.

the claim extract contains no documents of any kind. This corpus exists so that the  document pipeline — classification, OCR hook, language detection, clinical entity and negation extraction, structured comparison against the billed claim, and span-grounded evidence rendering — can be exercised and demonstrated.

Documents: 800
By language: {'en': 552, 'ar': 248}
Deliberately inconsistent: 95 ({'stay_length_conflict': 38, 'diagnosis_conflict': 29, 'date_conflict': 28})
Poor OCR quality: 142 (so that confidence degradation is observable,)

manifest.csv records exactly which claims received an injected inconsistency and what was injected, so that a detection can be distinguished from a coincidence.

No figure derived from this corpus is a measurement of clinical-NLP performance. It measures whether the pipeline's plumbing works.
