""""Why was this flagged?" — a plain explanation a reviewer can read in 30 seconds.

For every case (and every claim a pattern-finder flagged) this module builds:

1. **A one-sentence headline** — who, how much, how many claims, when, how many
   unusual patterns, and what policy suggests should happen next.
2. **Ranked reasons (at most five)** — each one a sentence with the actual fact
   and a comparison ("AED 2,649 per day … similar hospitals about AED 912 … roughly
   3 times as much"), never a rule id.
3. **How strong the evidence is** — Strong / Moderate / Weak, with one line on why,
   and an explicit note when several reasons rest on the same underlying fact
   and therefore count once. The label is computed with the engine's own
   evidence-capping rule (:func:`fwa.engine.signals.cap_evidence`: the maximum
   per fact, never the sum), so the words cannot drift from the arithmetic.
4. **What this does not mean** — always "a reason to look, not proof of fraud",
   plus the model-only wording ("No rule was broken … amount at risk not yet
   established") wherever it applies.
5. **What to check next** — two to four concrete actions from the control
   catalogue's plain text.
6. **Technical details** — rule ids, raw scores, thresholds, SHAP values and
   field names, for auditors and examiners, rendered collapsed by the UI.

Nothing here changes a disposition, a priority or an exposure. It reads the
signals the engine produced and says what they say.
"""

from __future__ import annotations

import html as _html
import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from ..engine.signals import cap_evidence
from .plain_language import (
    RAW_TOKEN_RE, RULE_ID_RE, SNAKE_TOKEN_RE, aed, control_text, count_phrase, feature_phrase,
    feature_unit,
    label, missing_information, model_label, pct, plain_date, plain_month, ratio_words,
    standard_text, status, subject_word, times_phrase, vocabulary,
)

__all__ = [
    "Reason", "EvidenceStrength", "CaseExplanation", "explain_case", "explain_model_claim",
    "reason_sentence", "is_plain", "MAX_REASONS", "diagnosis_names", "ModelClaimExplanation",
]

#: The brief: "Reasons, most important first (at most five)."
MAX_REASONS = 5

Masker = Callable[[Any], str]


def _identity(value: Any) -> str:
    return str(value)


# =============================================================================
# data classes
# =============================================================================


@dataclass
class Reason:
    sentence: str
    rule_id: str
    title: str
    fact_key: str
    strength: float
    domain: str
    signal_ids: list[str] = field(default_factory=list)
    more_like_it: int = 0              # further signals of the same check on this case
    simplified: bool = False           # the check ran on a stand-in (PARTIAL)
    model_only: bool = False


@dataclass
class EvidenceStrength:
    label: str                         # Strong | Moderate | Weak
    score: float                       # capped evidence, 0–1
    why: str
    shared_fact_note: str = ""
    tone: str = "neutral"


@dataclass
class CaseExplanation:
    case_id: str
    subject_label: str
    headline: str
    reasons: list[Reason]
    strength: EvidenceStrength
    caveats: list[str]
    next_steps: list[str]
    disposition_label: str
    disposition_meaning: str
    priority_label: str
    amount_line: str
    technical: list[dict[str, Any]] = field(default_factory=list)
    period_label: str = ""
    claim_count: int = 0

    # ---------------------------------------------------------------- export

    def to_markdown(self) -> str:
        lines = [
            f"# Case summary — {self.subject_label}",
            "",
            f"**{self.headline}**",
            "",
            f"- What policy suggests: **{self.disposition_label}** — {self.disposition_meaning}",
            f"- Priority: **{self.priority_label}**",
            f"- Amount: {self.amount_line}",
            "",
            "## Why it was flagged",
            "",
        ]
        for i, r in enumerate(self.reasons, start=1):
            extra = []
            if r.more_like_it:
                extra.append(f"{r.more_like_it} more like it")
            if r.simplified:
                extra.append("simplified check")
            tail = f" _({'; '.join(extra)})_" if extra else ""
            lines.append(f"{i}. {r.sentence}{tail}")
        lines += ["", "## How strong the evidence is", "",
                  f"**{self.strength.label}.** {self.strength.why}"]
        if self.strength.shared_fact_note:
            lines.append(self.strength.shared_fact_note)
        lines += ["", "## What this does not mean", ""]
        lines += [f"- {c}" for c in self.caveats]
        lines += ["", "## What to check next", ""]
        lines += [f"{i}. {s}" for i, s in enumerate(self.next_steps, start=1)]
        lines += ["", "## Technical details", "",
                  "| Check | Version | Score | Confidence | Evidence strength | Underlying fact |",
                  "|---|---|---|---|---|---|"]
        for t in self.technical:
            lines.append(
                f"| {t['rule_id']} | {t['rule_version']} | {t['score']:.0f} | "
                f"{t['confidence']:.2f} | {t['evidence_strength']:.2f} | `{t['fact_key']}` |"
            )
        lines += ["", f"_Case {self.case_id}. {standard_text('not_proof')}_", ""]
        return "\n".join(lines)

    def to_html(self, *, synthetic_notice: str = "") -> str:
        e = _html.escape
        reasons = "".join(
            f"<li>{e(r.sentence)}"
            + (f" <span class='muted'>({e(str(r.more_like_it))} more like it)</span>" if r.more_like_it else "")
            + (" <span class='muted'>(simplified check)</span>" if r.simplified else "")
            + "</li>"
            for r in self.reasons
        )
        caveats = "".join(f"<li>{e(c)}</li>" for c in self.caveats)
        steps = "".join(f"<li>{e(s)}</li>" for s in self.next_steps)
        rows = "".join(
            f"<tr><td>{e(t['rule_id'])}</td><td>{e(t['rule_version'])}</td>"
            f"<td>{t['score']:.0f}</td><td>{t['confidence']:.2f}</td>"
            f"<td>{t['evidence_strength']:.2f}</td><td><code>{e(t['fact_key'])}</code></td></tr>"
            for t in self.technical
        )
        banner = f"<p class='synthetic'>{e(synthetic_notice)}</p>" if synthetic_notice else ""
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Case summary — {e(self.subject_label)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:780px;margin:24px auto;color:#1d1a20;line-height:1.45;padding:0 16px}}
 h1{{font-size:1.35rem;margin-bottom:.2rem}} h2{{font-size:1.05rem;margin:1.2rem 0 .4rem;border-bottom:1px solid #ddd;padding-bottom:.2rem}}
 .headline{{font-size:1.05rem;font-weight:600}} .muted{{color:#6b6470}} .synthetic{{background:#fff4d6;border:1px solid #e0c060;padding:6px 10px;border-radius:6px}}
 .meta{{display:flex;gap:18px;flex-wrap:wrap;font-size:.92rem}} table{{border-collapse:collapse;font-size:.85rem;width:100%}} td,th{{border:1px solid #ddd;padding:4px 6px;text-align:left}}
 .caveat{{background:#f3f0f5;border-radius:6px;padding:6px 12px}}
 @media print{{body{{margin:0}} .noprint{{display:none}}}}
</style></head><body>
{banner}
<h1>Case summary — {e(self.subject_label)}</h1>
<p class="headline">{e(self.headline)}</p>
<div class="meta"><div><b>What policy suggests:</b> {e(self.disposition_label)}</div>
<div><b>Priority:</b> {e(self.priority_label)}</div><div><b>Amount:</b> {e(self.amount_line)}</div></div>
<h2>Why it was flagged</h2><ol>{reasons}</ol>
<h2>How strong the evidence is</h2><p><b>{e(self.strength.label)}.</b> {e(self.strength.why)}
{(' ' + e(self.strength.shared_fact_note)) if self.strength.shared_fact_note else ''}</p>
<h2>What this does not mean</h2><ul class="caveat">{caveats}</ul>
<h2>What to check next</h2><ol>{steps}</ol>
<h2>Technical details</h2><table><tr><th>Check</th><th>Version</th><th>Score</th><th>Confidence</th><th>Evidence strength</th><th>Underlying fact</th></tr>{rows}</table>
<p class="muted">Case {e(self.case_id)}.</p>
</body></html>"""


# =============================================================================
# plain-sentence guard
# =============================================================================


def is_plain(text: str) -> bool:
    """True when a sentence contains no code tokens, rule ids or bare nan/None."""
    if not text:
        return False
    if RAW_TOKEN_RE.search(text) or RULE_ID_RE.search(text):
        return False
    if SNAKE_TOKEN_RE.search(text):
        return False
    lowered = f" {text.lower()} "
    return not any(w in lowered for w in (" nan ", " none ", " nat ", "nan.", "none."))


def _mask_text(text: str, identities: Iterable[Any], mask: Masker) -> str:
    """Replace every raw identity in ``text`` with its masked form."""
    out = str(text)
    for ident in sorted({str(i) for i in identities if i not in (None, "")}, key=len, reverse=True):
        if ident and ident in out:
            out = out.replace(ident, mask(ident))
    return out


# =============================================================================
# per-control reason builders (the 37 controls that run on any claim file)
# =============================================================================


def _num(v: Any, default: float | None = None) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def _days(v: Any) -> str:
    n = _num(v, 0) or 0
    return "the same day" if n == 0 else f"{n:,.0f} day{'s' if round(n) != 1 else ''}"


def _duration(v: Any) -> str:
    """A length of time, where 0 means zero days (not "the same day")."""
    n = _num(v, 0) or 0
    return f"{n:,.0f} day{'s' if round(n) != 1 else ''}"


def _unit_value(value: Any, unit: str) -> str:
    v = _num(value)
    if v is None:
        return "not available"
    if unit in ("AED",):
        return aed(v)
    if unit == "AED/day":
        return f"{aed(v)} per day"
    if unit == "ratio":
        return pct(v)
    if unit == "days":
        return f"{v:,.1f} days" if not float(v).is_integer() else f"{v:,.0f} days"
    if unit == "flag":
        return "yes" if v >= 0.5 else "no"
    if unit == "count":
        return f"{v:,.1f}" if not float(v).is_integer() else f"{v:,.0f}"
    return f"{v:,.2f}"


class _Ctx:
    """What a reason builder may use: masking and diagnosis names."""

    def __init__(self, mask: Masker, claims: pd.DataFrame | None,
                 dx_names: dict[str, str] | None = None) -> None:
        self.mask = mask
        self._dx: dict[str, str] = dict(dx_names or {})
        if claims is not None and not claims.empty and "diagnosis_primary" in claims.columns:
            if "diagnosis_description" in claims.columns:
                pairs = claims[["diagnosis_primary", "diagnosis_description"]].dropna()
                for code, name in zip(pairs["diagnosis_primary"].astype(str),
                                      pairs["diagnosis_description"].astype(str)):
                    self._dx.setdefault(code, name)

    def dx(self, code: Any) -> str:
        """"primary gonarthrosis (M17.1)", or "diagnosis code M17.1" when unnamed."""
        if code in (None, "", "nan"):
            return "this diagnosis"
        code = str(code)
        name = self._dx.get(code, "")
        if name and name != code and is_plain(name):
            return f"{name[:1].lower() + name[1:]} ({code})"
        return f"diagnosis code {code}"


def _b_anl_01_r01(ev, c):
    contribs = [x for x in ev.get("per_feature_contributions") or [] if _num(x.get("contribution"), 0)]
    contribs = sorted(contribs, key=lambda x: -abs(_num(x.get("contribution"), 0) or 0))[:2]
    parts = []
    for x in contribs:
        unit = str(x.get("unit") or "")
        v, m = x.get("value"), x.get("peer_median")
        comp = times_phrase(v, m, "the typical level") if unit not in ("ratio",) else ""
        parts.append(
            f"its {str(x.get('label', 'measure')).lower()} is {_unit_value(v, unit)} against "
            f"{_unit_value(m, unit)} at similar providers" + (f" ({comp})" if comp else "")
        )
    peers = _num(ev.get("peer_n"))
    lead = f"Compared with {peers:,.0f} similar providers, " if peers else "Compared with similar providers, "
    if not parts:
        return None
    return lead + " and ".join(parts) + "."


def _b_anl_01_r02(ev, c):
    metric = str(ev.get("metric") or "billing")
    pre, post = ev.get("pre_change_mean"), ev.get("post_change_mean")
    unit = "ratio" if "ratio" in metric else ("AED" if "value" in metric or "amount" in metric else "count")
    when = plain_month(ev.get("change_date")) if ev.get("change_date") else "during the period"
    direction = "rise" if str(ev.get("direction")) == "increase" else "fall"
    months = _num(ev.get("periods_of_history"))
    return (f"Its {metric} changed from {_unit_value(pre, unit)} to {_unit_value(post, unit)} around "
            f"{when} — a sudden {direction} compared with its own history"
            + (f" of {months:,.0f} months." if months else "."))


def _model_reason(ev, c, subject: str = "provider"):
    model = model_label(str(ev.get("model_name") or "model"))
    p = _num(ev.get("score_percentile"))
    rank = "among the most unusual" if p is None else (
        "the most unusual" if p >= 0.999 else f"in the top {max(1, round((1 - p) * 100))}%")
    lead = f"An automated pattern-finder ({model}) ranked this {subject} {rank} of those it scored"
    top = [x for x in ev.get("shap_top_features_original_units") or []
           if x.get("feature") not in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED")]
    if top:
        x = top[0]
        name = str(x.get("feature") or "")
        unit = feature_unit(name) if name else str(x.get("unit") or "")
        if unit == "flag":
            lead += f", mainly because of {feature_phrase(name)}"
        else:
            comp = (f"{_unit_value(x.get('value'), unit)}, compared with a typical "
                    f"{_unit_value(x.get('peer_median'), unit)}")
            lead += f", mainly because of {feature_phrase(name)} ({comp})"
    return lead + ". No rule was broken."


def _b_anl_01_r04(ev, c):
    feats = [x for x in ev.get("top_residual_features") or []][:2]
    words = []
    for x in feats:
        name = str(x.get("feature") or "")
        direction = "above" if (_num(x.get("mean_residual"), 0) or 0) > 0 else "below"
        words.append(f"{feature_phrase(name) if name else str(x.get('label', '')).lower()} {direction} typical")
    size = _num(ev.get("cluster_size"))
    return (f"It belongs to a group of {size:,.0f} providers that share an unusual combination "
            f"({' and '.join(words) or 'several measures'}) that does not match any known pattern. "
            f"This is a new pattern to study, not a finding.") if size else None


def _b_cln_01_r01(ev, c):
    obs, peer = ev.get("observed_rate"), ev.get("peer_mean")
    n, k = _num(ev.get("claim_count")), _num(ev.get("mismatch_count"))
    return (f"{pct(obs)} of its {n:,.0f} claims ({k:,.0f}) have a diagnosis that the claim file marks as "
            f"not matching the procedure billed; for similar providers it is {pct(peer)}."
            if n and k is not None else None)


def _b_cln_01_r02(ev, c):
    per_day, med = ev.get("amount_per_los_day"), ev.get("peer_median")
    los = _num(ev.get("length_of_stay_days"))
    return (f"It charged {aed(per_day)} per day of hospital stay for {c.dx(ev.get('diagnosis_code'))}"
            + (f" ({los:,.0f}-day stay)" if los else "") +
            f". Similar claims charge about {aed(med)} per day, so this is "
            f"{ratio_words((_num(per_day, 0) or 0) / max(_num(med, 1) or 1, 1e-9))} as much.")


def _b_doc_conflict(ev, c):
    field_name = str(ev.get("conflicting_field") or ev.get("field") or "")
    doc, claim = ev.get("extracted_value"), ev.get("claim_value")
    if "length_of_stay" in field_name or ev.get("documented_days") is not None:
        return (f"The discharge summary says the stay was {_days(doc)}, but the claim bills "
                f"{_days(claim)}.")
    return f"The attached document says {doc}, but the claim says {claim}."


def _b_cln_03_r01(ev, c):
    return (f"The diagnosis on this claim ({c.dx(ev.get('diagnosis_code'))}) is marked in the claim "
            f"file as not matching the procedure billed, on a claim of {aed(ev.get('gross_amount_aed'))}.")


def _b_cln_04_r01(ev, c):
    return (f"The patient had the same kind of service ({c.dx(ev.get('diagnosis_code'))}) again after "
            f"{_days(ev.get('days_between'))}; the expected minimum gap is "
            f"{_days(ev.get('interval_threshold'))}.")


def _b_cln_04_r02(ev, c):
    providers = ev.get("providers_in_window") or []
    return (f"The patient had {_num(ev.get('claims_in_window'), 0):,.0f} claims within "
            f"{_num(ev.get('window_days'), 0):,.0f} days at {len(providers)} different providers "
            f"({aed(ev.get('total_gross_aed'))} in total); more than "
            f"{_num(ev.get('threshold'), 0):,.0f} in that window is unusual.")


def _b_cln_04_r04(ev, c):
    return (f"It averages {_num(ev.get('claims_per_member'), 0):.1f} claims per patient; similar "
            f"providers average {_num(ev.get('peer_median'), 0):.1f}.")


def _b_cln_05_r01(ev, c):
    return (f"The stay lasted {_duration(ev.get('length_of_stay_days'))} for "
            f"{c.dx(ev.get('diagnosis_code'))}; similar stays usually last about "
            f"{_duration(ev.get('peer_median_los'))}.")


def _b_cln_05_r02(ev, c):
    return (f"The same patient was admitted again {_days(ev.get('discharge_readmit_gap_days'))} after "
            f"being discharged, inside the {_num(ev.get('readmit_window_days'), 30):,.0f}-day "
            f"readmission window. A readmission is not by itself an error.")


def _b_cln_05_r03(ev, c):
    gaps = ev.get("gap_days") or []
    gap = gaps[0] if gaps else None
    return (f"Two admissions for the same patient at the same provider, {_days(gap)} apart, were "
            f"billed as separate claims ({aed(ev.get('combined_gross_aed'))} together); they may "
            f"belong to one episode of care.")


def _b_cln_06_r02(ev, c):
    limbs = ev.get("limb_fired") or []
    if "exclusion_listed" in limbs:
        return (f"The provider is marked in the file as excluded, yet billed "
                f"{_num(ev.get('claim_count'), 0):,.0f} claims.")
    return (f"It billed {_num(ev.get('claim_count'), 0):,.0f} claims on only "
            f"{_num(ev.get('active_days'), 0):,.0f} days, averaging {aed(ev.get('mean_gross_aed'))} a "
            f"claim against {aed(ev.get('peer_median_gross_aed'))} at similar providers.")


def _b_cln_06_r03(ev, c):
    return (f"{_num(ev.get('repeat_count'), 0):,.0f} claims for {_num(ev.get('distinct_members'), 0):,.0f} "
            f"different patients have an identical pattern (same diagnosis, length of stay and amount).")


def _b_cln_07_r02(ev, c):
    return (f"On {plain_date(ev.get('peak_day'))}, {_num(ev.get('claims_on_peak_day'), 0):,.0f} patients "
            f"were billed as in hospital at once, against an assumed capacity of about "
            f"{_num(ev.get('effective_limit'), 0):,.0f}.")


def _b_cln_07_r04(ev, c):
    return (f"Its monthly number of claims moved from about {_num(ev.get('pre_change_mean'), 0):,.1f} to "
            f"{_num(ev.get('post_change_mean'), 0):,.1f} around {plain_month(ev.get('change_date'))}, "
            f"a sudden change compared with its own history.")


def _b_doc_02_r01(ev, c):
    return (f"Discharge summaries for two different patients are {pct(ev.get('similarity'))} identical "
            f"once standard wording is removed.")


def _b_ent_02_r03(ev, c):
    adm = ev.get("admission_dates") or []
    return (f"The same patient was an inpatient at two different providers at the same time: the stays "
            f"overlapped by {_days(ev.get('overlap_days'))}"
            + (f" (admitted {plain_date(adm[0])} and {plain_date(adm[1])})." if len(adm) >= 2 else "."))


def _b_ent_03_r02(ev, c):
    return (f"The provider is marked as excluded in the file, yet billed "
            f"{_num(ev.get('claim_count'), 0):,.0f} claims worth {aed(ev.get('total_gross_aed'))}.")


def _b_net_01_r01(ev, c):
    return (f"This agent's customers used one provider for {pct(ev.get('top_provider_share'))} of "
            f"{_num(ev.get('claim_count'), 0):,.0f} claims; for similar agents the figure is "
            f"{pct(ev.get('peer_median_top_share'))}.")


def _b_net_02_r02(ev, c):
    ext = _num(ev.get("external_ratio"))
    return (f"A tightly connected group of {count_phrase(ev.get('member_count'), 'patient')}, "
            f"{count_phrase(ev.get('provider_count'), 'provider')} and "
            f"{count_phrase(ev.get('agent_count'), 'agent')} formed in the week of "
            f"{plain_date(ev.get('window_start'))}"
            + (f"; {pct(1 - ext)} of its links stay inside the group." if ext is not None else "."))


def _b_net_02_r03(ev, c):
    claims = ev.get("synchronised_claims") or []
    return (f"{len(claims)} claims for {_num(ev.get('distinct_members'), 0):,.0f} different patients were "
            f"billed on the same day ({plain_date(ev.get('shared_date'))}) for almost identical amounts.")


def _b_net_03_r01(ev, c):
    return (f"This patient and provider have {_num(ev.get('dyad_claim_count'), 0):,.0f} claims together "
            f"totalling {aed(ev.get('dyad_total_aed'))}; a typical patient–provider pair totals about "
            f"{aed(ev.get('dyad_value_median_aed'))}.")


def _b_net_04_r02(ev, c):
    return (f"This agent's {_num(ev.get('distinct_members'), 0):,.0f} customers used only "
            f"{_num(ev.get('distinct_providers'), 0):,.0f} providers where about "
            f"{_num(ev.get('expected_distinct_providers'), 0):,.0f} would be expected; "
            f"{pct(ev.get('top_provider_share'))} of claims went to one provider.")


def _b_net_04_r03(ev, c):
    return (f"{pct(ev.get('agent_rate'))} of this agent's claims are high-value claims in the first "
            f"{_num(ev.get('early_tenure_days'), 90):,.0f} days of a policy; across all agents it is "
            f"{pct(ev.get('overall_rate'))}.")


def _b_pay_01_r01(ev, c):
    return (f"This claim repeats an earlier claim ({ev.get('duplicate_of_claim_sk')}) exactly: same "
            f"patient, provider, diagnosis, dates and amount ({aed(ev.get('gross_amount_aed'))}).")


def _b_pay_01_r02(ev, c):
    return (f"A near-identical claim ({ev.get('matched_claim_sk')}) for the same patient, provider and "
            f"diagnosis was billed {_days(ev.get('days_apart'))} apart, with amounts differing by "
            f"{aed(ev.get('amount_difference_aed'))}.")


def _b_pay_01_r03(ev, c):
    return (f"One episode of care for {c.dx(ev.get('diagnosis_code'))} was billed on "
            f"{len(ev.get('claim_sks') or [])} separate claims ({aed(ev.get('episode_total_gross_aed'))} "
            f"in total).")


def _b_pay_06_r02(ev, c):
    return (f"The approved amount ({aed(ev.get('approved_amount_aed'))}) is higher than the amount "
            f"billed ({aed(ev.get('gross_amount_aed'))}).")


def _b_pay_06_r03(ev, c):
    g, m = ev.get("gross_amount_aed"), ev.get("peer_median_aed")
    return (f"It billed {aed(g)} for {c.dx(ev.get('diagnosis_code'))}; similar claims usually bill "
            f"about {aed(m)}, so this is {times_phrase(g, m, 'the usual amount')}.")


def _b_pay_06_r04(ev, c):
    limbs = ev.get("limb_fired") or []
    if any("low" in str(l) for l in limbs):
        return (f"The insurer approved {pct(ev.get('observed_approval_ratio'))} of what this provider "
                f"billed; for similar providers it is {pct(ev.get('peer_mean'))}.")
    return "Its approved amounts are almost identical on every claim, which similar providers do not show."


def _b_pay_10_r01(ev, c):
    return (f"{_num(ev.get('num_insurers_same_event'), 0):,.0f} insurers are recorded for the same event, "
            f"but there is no record of which insurer should pay first.")


def _b_phr_03_r03(ev, c):
    return (f"Medicines make up {pct(ev.get('pharmacy_bill_ratio'))} of this bill; similar claims for "
            f"{c.dx(ev.get('diagnosis_code'))} run about {pct(ev.get('peer_median_ratio'))}.")


def _b_pol_01_r04(ev, c):
    return (f"A high-value claim ({aed(ev.get('gross_amount_aed'))}) came "
            f"{_days(ev.get('days_since_policy_start'))} after the policy started; similar claims "
            f"usually bill about {aed(ev.get('peer_median_aed'))}. Early claims alone are not evidence "
            f"of wrongdoing.")


_BUILDERS: dict[str, Callable[[dict, _Ctx], str | None]] = {
    "ANL-01-R01": _b_anl_01_r01, "ANL-01-R02": _b_anl_01_r02,
    "ANL-01-R03": lambda ev, c: _model_reason(ev, c, "provider"), "ANL-01-R04": _b_anl_01_r04,
    "CLN-01-R01": _b_cln_01_r01, "CLN-01-R02": _b_cln_01_r02, "CLN-01-R03": _b_doc_conflict,
    "CLN-03-R01": _b_cln_03_r01, "CLN-04-R01": _b_cln_04_r01, "CLN-04-R02": _b_cln_04_r02,
    "CLN-04-R04": _b_cln_04_r04, "CLN-05-R01": _b_cln_05_r01, "CLN-05-R02": _b_cln_05_r02,
    "CLN-05-R03": _b_cln_05_r03, "CLN-06-R02": _b_cln_06_r02, "CLN-06-R03": _b_cln_06_r03,
    "CLN-07-R02": _b_cln_07_r02, "CLN-07-R04": _b_cln_07_r04, "DOC-01-R02": _b_doc_conflict,
    "DOC-02-R01": _b_doc_02_r01, "ENT-02-R03": _b_ent_02_r03, "ENT-03-R02": _b_ent_03_r02,
    "NET-01-R01": _b_net_01_r01, "NET-02-R02": _b_net_02_r02, "NET-02-R03": _b_net_02_r03,
    "NET-03-R01": _b_net_03_r01, "NET-04-R02": _b_net_04_r02, "NET-04-R03": _b_net_04_r03,
    "PAY-01-R01": _b_pay_01_r01, "PAY-01-R02": _b_pay_01_r02, "PAY-01-R03": _b_pay_01_r03,
    "PAY-06-R02": _b_pay_06_r02, "PAY-06-R03": _b_pay_06_r03, "PAY-06-R04": _b_pay_06_r04,
    "PAY-10-R01": _b_pay_10_r01, "PHR-03-R03": _b_phr_03_r03, "POL-01-R04": _b_pol_01_r04,
}


def _template(template: str, evidence: dict[str, Any]) -> str | None:
    """Fill a vocabulary template ``{key}`` / ``{key:aed}`` / ``{key:pct}`` / ``{key:date}``."""
    import re

    def repl(m: "re.Match[str]") -> str:
        key, _, fmt = m.group(1).partition(":")
        if key not in evidence or evidence[key] in (None, ""):
            raise KeyError(key)
        v = evidence[key]
        return {"aed": aed, "pct": pct, "date": plain_date}.get(fmt, lambda x: str(x))(v)

    try:
        return re.sub(r"\{([a-z0-9_]+(?::[a-z]+)?)\}", repl, template)
    except KeyError:
        return None


def _identities(signal) -> list[Any]:
    ev = signal.evidence or {}
    out: list[Any] = [signal.subject_id]
    for key in ("provider_sk", "member_sk", "agent_id", "clinician_id", "prior_provider_sk",
                "top_provider_sk", "pharmacy_id", "prescriber_id", "entity"):
        v = ev.get(key)
        if isinstance(v, (list, tuple)):
            out.extend(v)
        elif v:
            out.append(v)
    for key in ("providers", "member_sks", "provider_sks", "members"):
        v = ev.get(key)
        if isinstance(v, (list, tuple)):
            out.extend(v)
    return out


def reason_sentence(signal, *, mask: Masker = _identity, claims: pd.DataFrame | None = None,
                    ctx: _Ctx | None = None) -> str:
    """One plain sentence for one signal: the fact and a comparison, never a rule id."""
    ctx = ctx or _Ctx(mask, claims)
    ev = dict(signal.evidence or {})
    text = control_text(signal.rule_id)
    candidates: list[str | None] = []
    builder = _BUILDERS.get(signal.rule_id)
    if builder is not None:
        try:
            candidates.append(builder(ev, ctx))
        except Exception:  # a builder must never break the page
            candidates.append(None)
    if text.explanation:
        candidates.append(_template(text.explanation, ev))
    if "shap_top_features_original_units" in ev and signal.rule_id not in _BUILDERS:
        candidates.append(_model_reason(ev, ctx, subject_word(signal.subject_type)))
    candidates.append(ev.get("plain_language"))
    for cand in candidates:
        if not cand:
            continue
        sentence = _mask_text(str(cand).strip(), _identities(signal), mask)
        if is_plain(sentence):
            return sentence if sentence.endswith((".", "!", "?")) else sentence + "."
    # Last resort: the control's own plain finding, which always reads as words.
    return text.finding


# =============================================================================
# case explanation
# =============================================================================


def _strength_band(score: float) -> tuple[str, str]:
    bands = vocabulary().evidence_strength_bands or [
        {"min": 0.75, "label": "Strong", "tone": "bad"},
        {"min": 0.5, "label": "Moderate", "tone": "warn"},
        {"min": 0.0, "label": "Weak", "tone": "neutral"},
    ]
    for band in sorted(bands, key=lambda b: -float(b.get("min", 0))):
        if score >= float(band.get("min", 0)):
            return str(band.get("label")), str(band.get("tone", "neutral"))
    return "Weak", "neutral"


_DOMAIN_PHRASE = {
    "rule": "a hard or expert rule",
    "statistical": "a comparison with similar providers or with past behaviour",
    "network": "the connections between providers, agents and patients",
    "document": "what the documents say",
    "model": "an automated pattern-finder",
}


def _domain_words(domains: Iterable[str]) -> list[str]:
    return [_DOMAIN_PHRASE.get(d, label(d, "domain").lower()) for d in sorted(set(domains))]


def _join(words: Sequence[str]) -> str:
    words = [w for w in words if w]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _number_word(n: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine"}.get(n, f"{n:,}")


def explain_case(
    case,
    signals: Sequence[Any],
    *,
    registry=None,
    claims: pd.DataFrame | None = None,
    mask: Masker = _identity,
    subject_label: str | None = None,
    dx_names: dict[str, str] | None = None,
) -> CaseExplanation:
    """Build the plain explanation of one case from the signals behind it.

    ``dx_names`` maps diagnosis codes to descriptions (see :func:`diagnosis_names`)
    so reasons can say "primary gonarthrosis (M17.1)" rather than a bare code.
    """
    ctx = _Ctx(mask, claims, dx_names)
    subject_kind = subject_word(case.primary_subject_type)
    subject_label = subject_label or f"{subject_kind.capitalize()} {mask(case.primary_subject_id)}"

    # ---- one reason per check (rule), strongest signal first ---------------
    by_rule: dict[str, list[Any]] = {}
    for s in signals:
        by_rule.setdefault(s.rule_id, []).append(s)
    reasons: list[Reason] = []
    for rule_id, group in by_rule.items():
        group = sorted(group, key=lambda s: -float(s.evidence_strength))
        top = group[0]
        text = control_text(rule_id)
        support = str((top.evidence or {}).get("data_support", ""))
        reasons.append(Reason(
            sentence=reason_sentence(top, mask=mask, ctx=ctx),
            rule_id=rule_id,
            title=text.title,
            fact_key=top.fact_key or top.signal_id,
            strength=float(top.evidence_strength),
            domain=getattr(top.domain, "value", str(top.domain)),
            signal_ids=[s.signal_id for s in group],
            more_like_it=len(group) - 1,
            simplified=support == "PARTIAL",
            model_only=getattr(top.domain, "value", str(top.domain)) == "model",
        ))
    reasons.sort(key=lambda r: (-r.strength, r.rule_id))
    shown = reasons[:MAX_REASONS]

    # ---- evidence strength, with the engine's own capping rule -------------
    capped = float(cap_evidence(signals))
    band, tone = _strength_band(capped)
    distinct_facts = {(s.fact_key or s.signal_id) for s in signals}
    domains = {getattr(s.domain, "value", str(s.domain)) for s in signals}
    kinds = _domain_words(domains)
    if len(domains) > 1:
        why = (f"{_number_word(len(distinct_facts)).capitalize()} separate "
               f"{'fact' if len(distinct_facts) == 1 else 'facts'} from {len(domains)} independent kinds "
               f"of evidence point the same way: {_join(kinds)}.")
    else:
        why = (f"It rests on {_number_word(len(distinct_facts))} "
               f"{'fact' if len(distinct_facts) == 1 else 'separate facts'}, from one kind of "
               f"evidence: {_join(kinds)}.")
    if any(r.simplified for r in shown):
        why += " Some checks ran in a simplified form on this file, which weakens them."
    avg_conf = sum(float(s.confidence) for s in signals) / max(len(signals), 1)
    if avg_conf < 0.75:
        why += " The data behind it is incomplete, so confidence is reduced."

    shared_note = ""
    by_fact: dict[str, list[int]] = {}
    for i, r in enumerate(shown, start=1):
        by_fact.setdefault(r.fact_key, []).append(i)
    shared = [idx for idx in by_fact.values() if len(idx) > 1]
    if shared:
        notes = []
        for idx in shared:
            nums = _join([str(i) for i in idx])
            notes.append(f"Reasons {nums} come from the same underlying fact, so together they count "
                         f"as one piece of evidence.")
        shared_note = " ".join(notes)

    # ---- caveats -------------------------------------------------------------
    caveats = [standard_text("not_proof")]
    model_only = domains <= {"model"}
    if model_only:
        caveats.append(standard_text("model_only"))
    elif not getattr(case, "exposure_established", True):
        caveats.append("The amount at risk has not been established: no objective rule has been shown "
                       "to fail, so the figure is the gross amount, not money owed.")
    if getattr(case, "shadow_only", False):
        caveats.append("Every check behind this case runs in watch-only mode, so it does not change any "
                       "payment by itself.")
    caveats.append(standard_text("priority_not_decision"))
    caveats = [c for c in caveats if c]

    # ---- next steps ----------------------------------------------------------
    steps: list[str] = []
    for r in shown:
        for s in control_text(r.rule_id).next_steps:
            if s not in steps:
                steps.append(s)
            if len(steps) >= 4:
                break
        if len(steps) >= 4:
            break
    if len(steps) < 2:
        from ..enums import REVIEWER_ASK

        ask = REVIEWER_ASK.get(case.disposition)
        if ask:
            steps.append(ask)

    # ---- headline -------------------------------------------------------------
    disp = status(case.disposition)
    suggestion = vocabulary().suggestions.get(getattr(case.disposition, "value", str(case.disposition)),
                                              "a reviewer checking it")
    claim_ids = list(getattr(case, "claim_ids", []) or [])
    billed = None
    if claims is not None and not claims.empty and claim_ids and "claim_sk" in claims.columns:
        sub = claims[claims["claim_sk"].astype(str).isin([str(c) for c in claim_ids])]
        if "gross_amount_aed" in sub.columns and not sub.empty:
            billed = float(sub["gross_amount_aed"].sum())
    period = str(getattr(case, "period_bucket", "") or "")
    period_label = plain_month(period + "-01") if len(period) == 7 and period[4] == "-" else ""
    n_patterns = len({r.fact_key for r in shown})
    pattern_words = (f"{_number_word(n_patterns)} unusual pattern" + ("s" if n_patterns != 1 else ""))
    verb = {"provider": "billed", "agent": "sold policies whose claims total",
            "patient": "had claims totalling", "claim": "was billed at"}.get(
        subject_kind, "involves claims totalling")
    money = f" {verb} {aed(billed)}" if billed is not None else ""
    claims_part = (f" across {len(claim_ids):,} claim{'s' if len(claim_ids) != 1 else ''}"
                   if claim_ids and not (subject_kind == "claim" and len(claim_ids) == 1) else "")
    when = f" in {period_label}" if period_label else ""
    headline = (f"{subject_label}{money}{claims_part}{when}, with {pattern_words}. "
                f"We suggest {suggestion}.")
    if not money:
        headline = (f"{subject_label}: {pattern_words}{claims_part and ' in' + claims_part[7:] or ''}"
                    f"{when}. We suggest {suggestion}.")

    amount_line = (aed(case.exposure_aed) if getattr(case, "exposure_established", True)
                   else f"{aed(case.exposure_aed)} — {standard_text('not_established')}")

    priority = getattr(case, "priority", None)
    priority_label = "not scored" if priority is None else (
        f"{label(priority.band, 'band')} ({priority.value:.0f} out of 100)")

    technical = [{
        "rule_id": s.rule_id, "rule_version": s.rule_version, "reason_code": s.reason_code,
        "score": float(s.score), "confidence": float(s.confidence),
        "evidence_strength": float(s.evidence_strength), "fact_key": s.fact_key,
        "domain": getattr(s.domain, "value", str(s.domain)),
        "disposition": getattr(s.disposition, "value", str(s.disposition)),
        "peer_level_used": s.peer_level_used, "exposure_aed": float(s.exposure_aed),
        "exposure_established": bool(s.exposure_established), "exposure_basis": s.exposure_basis,
        "evidence": json.dumps(s.evidence, default=str)[:4000],
    } for s in sorted(signals, key=lambda s: -float(s.evidence_strength))]

    return CaseExplanation(
        case_id=case.case_id, subject_label=subject_label, headline=headline, reasons=shown,
        strength=EvidenceStrength(label=band, score=capped, why=why, shared_fact_note=shared_note,
                                  tone=tone),
        caveats=caveats, next_steps=steps[:4], disposition_label=disp.label,
        disposition_meaning=disp.meaning, priority_label=priority_label, amount_line=amount_line,
        technical=technical, period_label=period_label, claim_count=len(claim_ids),
    )


# =============================================================================
# a single claim flagged by a pattern-finding model
# =============================================================================


@dataclass
class ModelClaimExplanation:
    claim_sk: str
    model: str
    headline: str
    reasons: list[str]
    caveats: list[str]
    percentile: float | None
    technical: list[dict[str, Any]]
    exploratory: bool = False


def explain_model_claim(layer, model_name: str, claim_sk: str, *, top_n: int = 3,
                        exploratory: bool = False) -> ModelClaimExplanation | None:
    """Why a pattern-finder ranked one claim as unusual, in words.

    ``top_n`` contributions are rendered as "because <feature phrase> was <value>,
    compared with a typical <range>", where the typical range is the middle half
    of scored claims — a reader's sense of "usual", not a model internal.
    """
    if layer is None or getattr(layer, "anomaly", None) is None:
        return None
    scores = layer.anomaly.models.get(model_name)
    if scores is None:
        return None
    claim_sk = str(claim_sk)
    if claim_sk not in scores.scores.index:
        return None
    score_series = scores.scores
    percentile = float((score_series <= score_series[claim_sk]).mean())
    contributions = layer.explain_claim(model_name, claim_sk, top_n=top_n) or []
    raw = getattr(layer, "raw_features", None)
    reasons: list[str] = []
    for c in contributions:
        feat = str(c.get("feature") or "")
        if feat in ("EXPLANATION_UNAVAILABLE", "EXPLANATION_FAILED") or not feat:
            reasons.append("The model's reasons for this score could not be worked out.")
            continue
        if (c.get("shap_value") or 0) <= 0:
            continue
        unit = feature_unit(feat)
        value = _unit_value(c.get("value"), unit)
        if unit == "flag" and raw is not None and feat in raw.columns:
            share = pd.to_numeric(raw[feat], errors="coerce").dropna()
            rate = float((share >= 0.5).mean()) if not share.empty else None
            phrase = feature_phrase(feat)
            if (_num(c.get("value"), 0) or 0) >= 0.5:
                reasons.append(phrase[:1].upper() + phrase[1:]
                               + (f", which is true for only {pct(rate)} of claims." if rate is not None else "."))
            else:
                reasons.append(f"The absence of {phrase}"
                               + (f", which is present on {pct(rate)} of claims." if rate is not None else "."))
            continue
        typical = ""
        if raw is not None and feat in raw.columns and pd.api.types.is_numeric_dtype(raw[feat]):
            col = raw[feat].dropna()
            if not col.empty and unit not in ("flag",):
                q1, q3 = col.quantile(0.25), col.quantile(0.75)
                typical = (f"a typical {_unit_value(q1, unit)}" if abs(q3 - q1) < 1e-9
                           else f"a typical {_unit_value(q1, unit)}–{_unit_value(q3, unit)}")
        if not typical:
            typical = f"a typical {_unit_value(c.get('peer_median'), unit)}"
        reasons.append(f"{feature_phrase(feat).capitalize()} was {value}, compared with {typical}.")
    top_pct = max(1, round((1 - percentile) * 100))
    lead = (f"An automated pattern-finder ({model_label(model_name)}) ranked this claim among the top "
            f"{top_pct}% of unusual claims")
    if reasons:
        first = reasons[0].rstrip(".")
        lead += f". The main reason: {first[0].lower() + first[1:]}."
    else:
        lead += "."
    caveats = [standard_text("model_only"), standard_text("not_proof")]
    if exploratory:
        caveats.insert(0, "Exploratory score: the model has seen some of this provider's claims during "
                          "training, so this score is not used for the promotion gate.")
    technical = [{"feature": c.get("feature"), "value": c.get("value"), "peer_median": c.get("peer_median"),
                  "shap_value": c.get("shap_value")} for c in contributions]
    return ModelClaimExplanation(
        claim_sk=claim_sk, model=model_name, headline=lead, reasons=reasons,
        caveats=[c for c in caveats if c], percentile=percentile, technical=technical,
        exploratory=exploratory,
    )


def diagnosis_names(dataset) -> dict[str, str]:
    """Diagnosis code -> description, from the canonical ``diagnosis`` table."""
    try:
        dx = dataset.get("diagnosis")
    except Exception:
        return {}
    if dx is None or dx.empty or "code" not in dx.columns or "description" not in dx.columns:
        return {}
    pairs = dx[["code", "description"]].dropna().drop_duplicates("code")
    return {str(c): str(d) for c, d in zip(pairs["code"], pairs["description"]) if str(d) != str(c)}
