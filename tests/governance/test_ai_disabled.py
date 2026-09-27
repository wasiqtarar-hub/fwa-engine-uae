"""The tool is fully functional with every AI feature off (build brief §9, §18).

    "Every AI feature must be **independently switchable**, and the tool must be
     **fully functional with all of them off**."
    "Do not require an API key or network access to run the tool."

Two claims, tested separately. The first is about *feature parity*: detection,
correlation, disposition, priority and exposure are identical with the AI layer
on and off, because none of them is produced by it. The second is about
*deployment*: the default path constructs no hosted client, reads no key and
opens no socket.
"""

from __future__ import annotations

import os

import pytest

from fwa.ai import AILayer
from fwa.ai.provider import OfflineDeterministicProvider, get_provider
from fwa.cases.correlate import CaseCorrelator
from fwa.cases.priority import PriorityScorer

pytestmark = pytest.mark.governance

AI_FEATURES = ("grounded_narrative", "reviewer_copilot", "rule_authoring", "typology_triage")


# ------------------------------------------------------------- switchability


def test_the_layer_can_be_built_with_everything_off(config):
    layer = AILayer(config=config, provider=OfflineDeterministicProvider(), enabled=False)
    assert layer.enabled is False
    assert layer.narrator is None
    assert layer.copilot is None
    assert layer.rule_author is None
    assert layer.triage is None
    status = layer.status()
    assert status["ai_enabled"] is False
    assert all(v is False for v in status["features"].values())


@pytest.mark.parametrize("feature", AI_FEATURES)
def test_each_feature_switches_independently(config, registry, feature, monkeypatch):
    """One feature on, the other three off — for each of the four in turn."""
    flags = {name: (name == feature) for name in AI_FEATURES}
    monkeypatch.setattr(
        type(config), "get",
        lambda self, key, as_of=None: (
            True if key == "ai_enabled"
            else flags if key == "ai_features_enabled"
            else "offline_deterministic" if key == "ai_provider"
            else AILayer.__dict__  # unreachable for the keys build() reads
        ),
        raising=False,
    )
    layer = AILayer.build(config, registry=registry)
    built = layer.status()["features"]
    assert built[feature] is True
    for other in AI_FEATURES:
        if other != feature:
            assert built[other] is False


# ------------------------------------------------------------ feature parity


def test_signals_are_produced_without_any_ai(sample_signals):
    """Detection is a property of the control catalogue, not of the AI layer."""
    assert sample_signals, "The catalogue produced no signals at all."
    assert all(not getattr(s, "is_synthetic", False) or s.evidence for s in sample_signals)


def test_cases_dispositions_priorities_and_exposures_are_identical_with_ai_off(
    config, registry, sample_signals
):
    """Correlate twice — once in a world with an AI layer, once without one.

    The correlator never receives the AI layer in either run, which is the
    point: there is no argument to pass. This test states that fact as an
    executable claim rather than leaving it to inspection.
    """
    def correlate():
        return CaseCorrelator(config, PriorityScorer(config)).correlate(sample_signals, registry)

    without_ai = correlate()

    layer = AILayer.build(config, registry=registry)  # exists, is simply never consulted
    assert layer is not None
    with_ai_present = correlate()

    assert set(without_ai) == set(with_ai_present)
    for case_id, case in without_ai.items():
        other = with_ai_present[case_id]
        assert case.disposition == other.disposition
        assert case.exposure_aed == pytest.approx(other.exposure_aed)
        assert case.exposure_established == other.exposure_established
        assert case.signal_ids == other.signal_ids
        if case.priority is not None:
            assert case.priority.value == pytest.approx(other.priority.value)


def test_the_queue_renders_without_the_ai_layer(sample_cases):
    correlator, cases = sample_cases
    assert cases
    frame = correlator.queue(tenant_id="T001")
    assert not frame.empty
    assert "disposition" in frame.columns


# ---------------------------------------------------------------- no network


def test_the_default_provider_is_offline_and_deterministic(monkeypatch):
    monkeypatch.delenv("FWA_LLM_PROVIDER", raising=False)
    provider = get_provider()
    assert isinstance(provider, OfflineDeterministicProvider)
    assert provider.deterministic is True


def test_the_default_provider_needs_no_api_key(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("FWA_LLM_PROVIDER", raising=False)
    response = get_provider().generate("summarise", {"case_id": "C1"}, task="narrative")
    assert response.text
    assert response.deterministic is True


def test_a_hosted_provider_without_a_key_degrades_to_offline(monkeypatch):
    """A missing key must yield a working offline artefact, not a crash."""
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FWA_LLM_PROVIDER", "anthropic")
    assert isinstance(get_provider(), OfflineDeterministicProvider)


def test_the_offline_provider_is_reproducible():
    a = OfflineDeterministicProvider().generate("p", {"case_id": "C1"}, task="narrative")
    b = OfflineDeterministicProvider().generate("p", {"case_id": "C1"}, task="narrative")
    assert a.text == b.text
    assert a.prompt_hash == b.prompt_hash


def test_a_hosted_provider_is_only_reachable_through_an_environment_variable(project_root):
    """No hosted client is constructed on any default path.

    The sweep is on the source: an ``import anthropic`` or ``import openai`` at
    module scope would make the package unimportable without those libraries
    installed, which is the same failure as requiring a key.
    """
    import ast

    for path in sorted((project_root / "src" / "fwa").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module scope only
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import) else [node.module or ""]
                )
                for name in names:
                    assert name.split(".")[0] not in ("anthropic", "openai"), (
                        f"{path.name} imports {name} at module scope"
                    )


def test_no_module_opens_a_socket_or_calls_out(project_root):
    import ast

    banned = {"requests", "httpx", "urllib", "urllib3", "socket", "http", "aiohttp"}
    offenders = []
    for path in sorted((project_root / "src" / "fwa").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in banned:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, "Network imports found:\n  " + "\n  ".join(offenders)


def test_no_api_key_is_present_in_the_test_environment():
    """Belt and braces: the suite itself proves the artefact runs without one."""
    assert not os.environ.get("ANTHROPIC_API_KEY")
    assert not os.environ.get("OPENAI_API_KEY")
