"""Presentation layer: plain-language wording for a non-technical reader.

Nothing in this package changes what the engine does. It translates internal
identifiers — enum values, field names, rule ids, model features — into words
at display time, so the identifiers that the contract, the audit log, the
reports and the tests depend on are never renamed.

The one rule every function here follows: plain wording must *carry* the
safety boundary, never drop it. "A reason to look, not proof of fraud" and
"amount at risk not yet established" travel with every finding they apply to.
"""

from .plain_language import (  # noqa: F401
    PlainVocabulary, Phrase, ControlText, vocabulary, reload_vocabulary,
    status, label, field_label, field_meaning, feature_phrase, feature_unit, control_text,
    control_title, table_label, missing_information, glossary, glossary_entry, term_help,
    page_info, standard_text, humanise, subject_word, model_label, parameter_text, plain_month,
    aed, pct, times_phrase, ratio_words, plain_date, plain_number, plain_value, yes_no,
    count_phrase, duration_days, friendly_frame,
)
