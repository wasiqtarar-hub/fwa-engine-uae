#!/usr/bin/env python3
"""Strip manuscript and build-brief citations from user-facing text.

The interface should explain itself without asking the reader to hold a
document open beside it. "Priority orders the queue only; it never sets or
overrides a disposition (§4.11)" says the same thing as the version without
the citation, to a reader who has the thesis and to a reader who does not.

What this touches and what it deliberately does not:

* **String literals that are not docstrings** — rewritten. These are what
  reaches the screen.
* **Docstrings** — left alone. They are the provenance record that ties each
  module to the design it implements, they are read by developers rather than
  users, and losing them would make the code harder to defend, not easier.
* **Comments** — left alone, for the same reason. ``tokenize`` gives them their
  own token type, so they are excluded by construction rather than by luck.

Run with ``--check`` to list what would change without writing anything.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import token as _token
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Phrases where the citation carries the only noun in the sentence, so
#: deleting it would leave "the formula" with no antecedent. Applied before the
#: general rules, longest first.
PHRASES: list[tuple[str, str]] = [
    ("the §4.11 formula", "the priority formula"),
    ("the §4.6 demonstration", "the shrinkage demonstration"),
    ("the §3.7 contract", "the atomic-control contract"),
    ("the §3.7 atomic-control contract", "the atomic-control contract"),
    ("the §3.3 safety boundary", "the safety boundary"),
    ("the §3.3 boundary", "the safety boundary"),
    ("the §4.12 dimensions", "the six correlation dimensions"),
    ("§4.12 correlation dimension", "correlation dimension"),
    ("the §4.8 calibration basis", "the calibration basis"),
    ("the §3.10 requirement", "the requirement"),
    ("the §14.1 requirement", "the requirement"),
    ("the §4.9 rule", "the rule"),
    ("Precision in the §6.5 sense", "Precision in the strict sense"),
    ("precision in the §6.5 sense", "precision in the strict sense"),
    ("which §4.11 and §7.8 treat as", "which this system treats as"),
    ("§4.11 and §7.8 treat", "this system treats"),
    ("§4.8 requires", "the promotion gate requires"),
    ("§4.8 treats", "the promotion gate treats"),
    ("§6.6 calls it", "the evaluation protocol calls it"),
    ("§14.1:", "The design bar:"),
    ("§3.3 makes", "the safety boundary makes"),
    ("§3.3 requires", "the safety boundary requires"),
    ("§3.3 forbids", "the safety boundary forbids"),
    ("§4.5 says", "the missingness rule says"),
    ("§4.6 requires", "the peer design requires"),
    ("§4.7 requires", "the change-detection design requires"),
    ("§3.6 requires", "the effective-dating rule requires"),
    ("§3.9 requires", "parameter governance requires"),
    ("§3.10 requires", "the platform requirements require"),
    ("§13.2 requires", "the access-control design requires"),
    ("§6.5 requires", "the metric definitions require"),
    ("§6.2 requires", "the test design requires"),
    ("§4.11 defines", "the exposure rules define"),
    ("§3.8 defines", "the signal contract defines"),
    ("the Table 6.1 hard-edit gate", "the hard-edit release gate"),
    ("the Table 6.1 statistical-rule", "the statistical-rule release gate's"),
    ("the six Table 6.1 gates", "the six release gates"),
    ("Table 6.1 gates", "release gates"),
    ("the Table 3.4 ceilings", "the stage ceilings"),
    ("Table 3.4 ceilings", "stage ceilings"),
    ("the §6.7 / Table 6.2 evidence bundle", "the evidence bundle"),
    ("Promotion gate (Table 6.1)", "Promotion gate"),
]

#: Phrases that appear in the governed YAML — rule catalogues, the parameter
#: registry, the peer hierarchy — rather than in Python. These files are data
#: the interface renders directly: a parameter's ``source`` and ``rationale``
#: are shown on the Parameters page beside its value, and a reader who has to
#: chase a section number to find out where a threshold came from has been told
#: nothing. Each replacement names the artefact instead of pointing at it.
YAML_PHRASES: list[tuple[str, str]] = [
    # Parentheticals whose whole content is a pointer, with enough words in
    # them that the general "a parenthetical of nothing but citations" rule
    # does not see them.
    (" (§13.3 of the build brief; manuscript §9.4 access control)", ""),
    ("(§4.6, explicit manuscript requirement)", "(an explicit design requirement)"),
    ("(§4.9, non-negotiable)", "(non-negotiable)"),
    ("(§3.10 NFR)", "(a platform requirement)"),
    ("(§3.10 reproducibility NFR)", "(the reproducibility requirement)"),
    ("(Definition of Done §17)", "(the Definition of Done)"),
    ("(§4.8, Table 6.1)", ""),
    ("(§4.8, §9.8)", ""),
    ("(§4.8, §6.6)", ""),
    ("(Appendix C, CLN-05-R02)", "(control catalogue, CLN-05-R02)"),
    ("(Appendix C CLN-07-R02:", "(control catalogue, CLN-07-R02:"),
    ("§4.6 permits \"median/MAD OR PERCENTILE-BASED robust statistics\"",
     "the design permits \"median/MAD OR PERCENTILE-BASED robust statistics\""),
    ("inside the §4.6 menu", "inside that menu"),
    ("§4.6 permits percentile-based robust statistics",
     "The design permits percentile-based robust statistics"),
    ("the §4.11 priority formula", "the priority formula"),
    ("the §4.6 shrinkage demonstration", "the shrinkage demonstration"),
    ("way §4.8 requires a threshold to be set", "way the design requires a threshold to be set"),
    ("per §4.8;", "to capacity;"),
    ("scenario-aligned per §4.8", "scenario-aligned"),
    ("the §3.3 boundary expressed as a number", "the safety boundary expressed as a number"),
    ("§4.10 prohibits automatic merging", "The design prohibits automatic merging"),
    ("the five detection domains in §4.1", "the five detection domains"),
    ("Manuscript Appendix C", "The control catalogue"),
    ("manuscript Appendix C", "the control catalogue"),
    ("Appendix C's", "the control catalogue's"),
    ("Appendix C", "the control catalogue"),
    ("Manuscript level", "Canonical level"),
    ("Manuscript Table 6.1.", "Release-gate acceptance criteria."),
    ("Manuscript Table 3.4.", "Stage latency ceilings."),
    ("Manuscript §4.6 requires", "The peer design requires"),
    ("Manuscript §4.7 requires", "The change-detection design requires"),
    ("Manuscript §4.10 explicit prohibition.", "Explicit prohibition on automatic merging."),
    ("Manuscript §4.10", "The graph design"),
    ("Manuscript §4.11.", "Design specification for the priority formula."),
    ("Manuscript §3.3 / ", ""),
    ("explicit manuscript requirement", "an explicit design requirement"),
    ("the manuscript says so explicitly", "the design says so explicitly"),
    ("the manuscript specifies the term, not its shape",
     "the design specifies the term, not its shape"),
    ("the manuscript requires a baseline, not a number",
     "the design requires a baseline, not a number"),
    ("the manuscript is explicit on this point", "the design is explicit on this point"),
    ("reproduced from manuscript", "reproduced from the design specification"),
    ("Reproduced EXACTLY from manuscript", "Reproduced EXACTLY from the design specification"),
    ("manuscript", "the design specification"),
    ("Manuscript", "The design specification"),
    ("Build brief §9.", "Design specification for the AI layer."),
    ("the build brief", "the design specification"),
    ("build brief", "design specification"),
    ("Build brief", "Design specification"),
    ("dissertation_author", "policy_owner"),
    ("dissertation artefact", "detection artefact"),
    ("dissertation", "design"),
    ("Definition of Done §17", "the Definition of Done"),
]

#: A citation token: §3.3, §9.7–9.9, plus the table numbers the interface
#: quoted. "Appendix C" survives on purpose — it names the control catalogue,
#: which the tool itself displays, so it is a thing on screen rather than a
#: pointer to a document the reader does not have.
_CITE = r"(?:§\d+(?:\.\d+)*(?:\s*[–-]\s*\d+(?:\.\d+)*)?|Table \d+\.\d+)"
_SOURCE = r"(?:build brief|manuscript|the (?:build )?brief)\s*"
#: One or more citations joined by a comma, "and" or a semicolon.
_CITES = rf"(?:{_SOURCE})?{_CITE}(?:\s*(?:,|and|;|/)\s*(?:{_SOURCE})?{_CITE})*"

#: The rewrite rules, as TEMPLATES. ``{ws}`` stands for "whitespace" so the
#: same table can be compiled twice: once for Python source, where a string
#: literal may legitimately span lines, and once for YAML, where a rule that
#: swallowed a newline would weld two lines together — which in an
#: indentation-sensitive format is not a formatting change but a different
#: document. Literal regex braces are doubled, because these are format
#: strings.
RULES: list[tuple[str, str]] = [
    # A parenthetical that is nothing but citations.
    (r"{ws}*\({cites}\)", ""),
    # A citation as one item inside a parenthetical that also says something
    # else: "(§13.2, PAY-11)" keeps PAY-11.
    (r"\({cites},{ws}*", "("),
    (r",{ws}*{cites}\)", ")"),
    # A trailing attribution at the end of the sentence or the literal.
    (r"{ws}*[—–-]+{ws}*{cites}(?=[{wsc}\"']*$)", ""),
    (r"{ws}*[—–-]+{ws}*{source}(?=[{wsc}\"']*$)", ""),
    # A prepositional phrase whose whole object is a citation:
    # "a direct requirement of §3.10 and §6.3." loses the phrase, not the noun.
    (r"{ws}+(?:of|in|per|under|from|by|to|against){ws}+{cites}(?=[{wsc}.,;:)\"']|$)", ""),
    # A citation used as a parenthetical aside between commas.
    (r",{ws}*{cites}(?=[{wsc}.,;:)])", ""),
    # Anything left.
    (r"{ws}*{cites}{ws}*", " "),
    # Tidy what the removals leave behind. The empty-parens rule must NOT fire
    # on a method call: inside an f-string, "{{frame['x'].nunique()}}" contains
    # a literal "()" that a naive rule happily deletes, turning a call into an
    # attribute reference that formats as "<built-in method nunique>". The
    # look-behind spares anything attached to an identifier, a closing bracket
    # or a quote, which is every call and no citation residue.
    (r"(?<![\w\]\)'\"])\({ws}*\)", ""),
    (r"{ws}+([.,;:!?])", r"\1"),
    (r"\({ws}+", "("),
    (r"{ws}+\)", ")"),
    (r"[ \t]{{2,}}", " "),
]


def _compile(ws: str, wsc: str) -> list[tuple[re.Pattern, str]]:
    """Compile the table. ``ws`` is used on its own; ``wsc`` inside a set.

    Two placeholders rather than one because a character class cannot be
    nested: ``[\s"\']`` is fine and ``[[ \t]"\']`` is a syntax error, so the
    two positions need different spellings of the same idea.
    """
    return [
        (re.compile(p.format(ws=ws, wsc=wsc, cites=_CITES, source=_SOURCE)), r)
        for p, r in RULES
    ]


_COMPILED = _compile(r"\s", r"\s")

#: Horizontal space only, for the line-oriented YAML pass.
_YAML_COMPILED = _compile(r"[ \t]", r" \t")

#: The rules that only REMOVE citations, with the three cosmetic tidy-ups left
#: out. Those tidy-ups are safe inside a Python string literal and destructive
#: in prose: collapsing runs of spaces flattens an aligned code block or a
#: directory listing, and pulling whitespace off the front of punctuation turns
#: "writing into ./reports" into "writing into./reports" and "pip install -e ."
#: into "pip install -e.". Markdown gets the removals and none of the tidying.
_TEXT_COMPILED = [
    (rx, repl) for rx, repl in _YAML_COMPILED
    if rx.pattern not in {
        r"[ \t]{2,}",
        r"[ \t]+([.,;:!?])",
        r"\([ \t]+",
        r"[ \t]+\)",
    }
]



_PREFIX_RE = re.compile(r"^([A-Za-z]*)(\"\"\"|'''|'|\")")


def _split_literal(literal: str) -> tuple[str, str, str] | None:
    """Split a source-level string literal into (opening, content, closing).

    Working on the CONTENT rather than the raw token matters for the rules that
    anchor to the end of the text: with the closing quote still attached, a
    trailing "— manuscript §4.6" never matches and leaves a dangling dash. The
    split keeps any prefix (``f``, ``r``, ``rb``) intact, so f-strings survive
    and their ``{...}`` holes are simply text the rules step over.
    """
    match = _PREFIX_RE.match(literal)
    if not match:
        return None
    opening = match.group(0)
    closing = match.group(2)
    if not literal.endswith(closing) or len(literal) < len(opening) + len(closing):
        return None
    return opening, literal[len(opening):-len(closing)], closing


def clean(text: str) -> str:
    """Apply the phrase table, then the general rules, to one piece of prose."""
    out = text
    for phrase, replacement in sorted(PHRASES, key=lambda p: -len(p[0])):
        out = out.replace(phrase, replacement)
    for pattern, replacement in _COMPILED:
        out = pattern.sub(replacement, out)
    return out


def clean_literal(literal: str) -> str:
    """Clean a source-level string literal, leaving its quoting untouched."""
    split = _split_literal(literal)
    if split is None:
        return clean(literal)
    opening, content, closing = split
    return opening + clean(content) + closing


def _docstring_spans(source: str) -> set[tuple[int, int]]:
    """Start positions of every docstring, so they can be skipped.

    A docstring is the first statement of a module, class or function. Walking
    the AST is the only reliable way to tell one from an ordinary string that
    happens to sit at the top of a block.
    """
    import ast

    spans: set[tuple[int, int]] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                and isinstance(first.value.value, str):
            spans.add((first.value.lineno, first.value.col_offset))
    return spans


def transform(path: Path) -> tuple[str, int]:
    """Return (new source, number of literals changed)."""
    source = path.read_text(encoding="utf-8")
    skip = _docstring_spans(source)
    lines = source.splitlines(keepends=True)

    edits: list[tuple[int, int, int, int, str]] = []
    reader = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type != _token.STRING:
            continue
        if (tok.start[0], tok.start[1]) in skip:
            continue
        cleaned = clean_literal(tok.string)
        if cleaned != tok.string:
            edits.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1], cleaned))

    if not edits:
        return source, 0

    # Apply back to front so earlier offsets stay valid.
    for start_row, start_col, end_row, end_col, replacement in reversed(edits):
        if start_row == end_row:
            line = lines[start_row - 1]
            lines[start_row - 1] = line[:start_col] + replacement + line[end_col:]
        else:
            head = lines[start_row - 1][:start_col]
            tail = lines[end_row - 1][end_col:]
            lines[start_row - 1:end_row] = [head + replacement + tail]
    return "".join(lines), len(edits)


def clean_yaml_line(text: str, rules=None) -> str:
    """Clean one line of YAML or markdown prose, leaving its structure alone."""
    out = text
    for phrase, replacement in YAML_PHRASES:
        out = out.replace(phrase, replacement)
    for phrase, replacement in sorted(PHRASES, key=lambda p: -len(p[0])):
        out = out.replace(phrase, replacement)
    for pattern, replacement in (rules if rules is not None else _YAML_COMPILED):
        out = pattern.sub(replacement, out)
    return out


_YAML_PREFIX = re.compile(r"^(\s*(?:#\s*(?:[*-]\s*)?)?)(.*?)(\s*)$")


def clean_yaml_text(text: str, rules=None) -> str:
    """Clean a YAML document line by line.

    Line by line rather than by parsing and re-emitting, because these files
    carry long explanatory comments that a round trip through a YAML loader
    would silently delete — and those comments are where the reasoning behind
    each threshold is written down.

    Each line is split into its structural prefix — indentation, a comment
    marker, a bullet — and its prose, and only the prose is rewritten. That is
    what keeps a key from being reindented into a different level of the
    document, which in YAML is not a formatting change but a different file.
    """
    out = []
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        newline = line[len(stripped):]
        match = _YAML_PREFIX.match(stripped)
        if match is None:                       # pragma: no cover - regex is total
            out.append(line)
            continue
        prefix, body, trailing = match.groups()
        out.append(prefix + clean_yaml_line(body, rules) + trailing + newline)
    return "".join(out)


def transform_yaml(path: Path) -> tuple[str, int]:
    source = path.read_text(encoding="utf-8")
    new = clean_yaml_text(source)
    if new == source:
        return source, 0
    changed = sum(1 for a, b in zip(source.splitlines(), new.splitlines()) if a != b)
    return new, changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=None,
                        help="Files or directories. Defaults to app/ and the UI-facing modules.")
    parser.add_argument("--check", action="store_true", help="Report without writing.")
    parser.add_argument("--yaml", action="store_true",
                        help="Treat the targets as YAML and clean them line by line.")
    args = parser.parse_args()

    suffix = "*.yaml" if args.yaml else "*.py"
    targets: list[Path] = []
    for raw in (args.paths or ["app"]):
        p = (ROOT / raw) if not Path(raw).is_absolute() else Path(raw)
        targets.extend(sorted(p.rglob(suffix)) if p.is_dir() else [p])

    total = 0
    unit = "line" if args.yaml else "literal"
    for path in targets:
        if "__pycache__" in path.parts:
            continue
        new_source, n = (transform_yaml if args.yaml else transform)(path)
        if not n:
            continue
        total += n
        print(f"{path.relative_to(ROOT)}: {n} {unit}(s)")
        if not args.check:
            path.write_text(new_source, encoding="utf-8")

    print(f"\n{'Would change' if args.check else 'Changed'} {total} {unit}(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def clean_markdown_text(text: str) -> str:
    """Strip citations from markdown without touching its whitespace.

    Markdown carries meaning in spacing — fenced code blocks, aligned comment
    columns, directory listings, two-space line breaks — so it gets the
    citation-removal rules and none of the cosmetic ones.
    """
    return clean_yaml_text(text, _TEXT_COMPILED)
