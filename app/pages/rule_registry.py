"""Rule Registry — Role 2, the policy owner (§9.3, §9.6).

Build brief §14, item 7: "browse the catalogue by scenario; view a control's
YAML, version history, shadow-period volume and review yield; author a rule in
**natural language → validated YAML draft → shadow**; the shadow→active
approval flow with sign-off capture."

§9.6's claim is that a policy owner can add a rule **without a code
deployment**. This page is where that claim is either true or it is not: a rule
drafted here becomes configuration in the registry, in shadow, with its
ten-category test stubs generated alongside, and nobody touches Python.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import yaml

from common import (
    ai_panel, banner, bar, dataframe, disposition_chip, empty_state, mono, note, pipeline,
    require_page, session_banner, status_chip, tiles,
)
from fwa.auth import Permission
from fwa.engine.contract import DispositionLegalityError
from fwa.engine.registry import SeparationOfDutiesError
from fwa.enums import DataSupport, RuleStatus


def render() -> None:
    state = require_page("Rule Registry", Permission.VIEW_RULE_REGISTRY)
    session_banner(state)

    st.markdown("# Rule registry")
    banner()

    result = pipeline()
    registry = result.registry
    summary = registry.summary()

    tab_browse, tab_control, tab_author, tab_approve = st.tabs([
        "Browse the catalogue", "Inspect a control", "Author a rule", "Approve activation",
    ])

    run = result.evaluation.to_frame()
    volumes = dict(zip(run["rule_id"], run["signal_count"])) if not run.empty else {}
    outcomes = run.set_index("rule_id")["outcome"].to_dict() if not run.empty else {}

    # ======================================================================
    with tab_browse:
        tiles([
            ("Scenarios", f"{summary['total_scenarios']}", "the whole catalogue"),
            ("Atomic controls", f"{summary['total_controls']}", "the whole catalogue"),
            ("Executable here", f"{summary['by_data_support'].get('EXECUTABLE', 0)}", ""),
            ("Partial (proxy)", f"{summary['by_data_support'].get('PARTIAL', 0)}", ""),
            ("Not executable", f"{summary['by_data_support'].get('NOT_EXECUTABLE_ON_THIS_DATASET', 0)}",
             "each with a stated reason"),
            ("May deny or reprice", f"{summary['controls_that_may_deny']}",
             "every one is type H or H/E"),
        ])
        note(
            "Every control is in SHADOW. The registry refuses at load time to accept a "
            "statistical, network, document or model control that declares REJECT or REPRICE — "
            "so the engine cannot be started with a rule that would let a score deny a claim."
        )

        rows = pd.DataFrame(registry.coverage_rows())
        rows["signals_this_run"] = rows["rule_id"].map(volumes).fillna(0).astype(int)
        rows["run_outcome"] = rows["rule_id"].map(outcomes).fillna("")

        f1, f2, f3 = st.columns(3)
        with f1:
            families = st.multiselect("Family", sorted(rows["family"].unique()),
                                      default=sorted(rows["family"].unique()))
        with f2:
            supports = st.multiselect("Data support", sorted(rows["data_support"].unique()),
                                      default=sorted(rows["data_support"].unique()))
        with f3:
            only_firing = st.checkbox("Only controls that produced signals", value=False)

        view = rows[rows["family"].isin(families) & rows["data_support"].isin(supports)]
        if only_firing:
            view = view[view["signals_this_run"] > 0]
        dataframe(view[["rule_id", "name", "priority", "type", "stage", "disposition",
                        "data_support", "signals_this_run", "run_outcome", "owner"]], height=430)

        counts = rows.groupby(["family", "data_support"]).size().reset_index(name="controls")
        bar(counts, "family", "controls", title="Catalogue coverage by family",
            colour="data_support", height=320)

    # ======================================================================
    with tab_control:
        st.markdown("## Inspect a control")
        scenario = st.selectbox("Scenario", registry.scenarios())
        controls = registry.by_scenario(scenario)
        rule_id = st.selectbox("Control", [c.rule_id for c in controls])
        control = registry.get(rule_id)

        st.markdown(
            f"### {control.rule_id} — {control.name} "
            f"<span class='fwa-tag'>{control.qualified_id}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            disposition_chip(control.disposition) + " "
            + status_chip(control.status.value.upper()) + " "
            + status_chip(control.data_support.value),
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Type** `{control.type_label}` — {control.primary_type.label}")
        c2.markdown(f"**Stage** `{control.stage.value}`")
        c3.markdown(f"**Owner** `{control.owner}`")

        st.markdown("**From the control catalogue, verbatim**")
        dataframe(pd.DataFrame([
            {"column": "Implementable trigger", "value": control.catalogue_trigger},
            {"column": "Configuration and exclusions", "value": control.catalogue_config_exclusions},
            {"column": "Output/disposition", "value": control.catalogue_disposition},
        ]))
        if control.governance_note:
            note(control.governance_note)

        st.markdown("**Data support on this dataset**")
        st.markdown(f"> {control.data_support_reason}")
        if control.required_canonical_fields:
            st.markdown("**What would unlock it**")
            st.markdown("\n".join(f"- `{f}`" for f in control.required_canonical_fields))

        st.markdown("**The atomic-control contract**")
        mono(yaml.safe_dump(control.to_yaml_dict(), sort_keys=False, allow_unicode=True,
                            width=88))

        signals = result.signals.by_rule(rule_id, state.tenant_id)
        st.markdown("**Shadow-period volume and review yield**")
        tiles([
            ("Signals this run", f"{len(signals):,}", ""),
            ("Run outcome", outcomes.get(rule_id, "—"), ""),
            ("Review yield", "NOT_MEASURABLE",
             "needs reviewer dispositions — the evaluation protocol calls it the primary signal of whether a "
             "control is still earning its place"),
        ])
        history = registry.history(rule_id)
        if history:
            st.markdown("**Version history**")
            dataframe(pd.DataFrame([
                {"version": h.version, "status": h.status.value, "fingerprint": h.fingerprint()}
                for h in history
            ]))

    # ======================================================================
    with tab_author:
        if not state.can(Permission.AUTHOR_RULE) and not state.can(Permission.AUTHOR_EXPERT_RULE):
            note(f"The {state.role.value} role may browse the registry but not author rules.")
        elif result.ai.rule_author is None:
            empty_state("Rule authoring is switched off",
                        "`cfg.ai_features_enabled.rule_authoring` is false. Every AI feature is "
                        "independently switchable and the tool remains fully functional without "
                        "them.")
        else:
            st.markdown("## Author a rule in natural language")
            st.markdown(
                "The LLM **drafts**. The deterministic validator **decides**. A human who did "
                "not author it **approves**. A drafted rule enters the registry in `shadow` with "
                "its ten-category test stubs generated alongside, and no code is deployed."
            )
            example = st.selectbox(
                "Try an example, or write your own",
                [
                    "Pend any claim where the pharmacy share exceeds twice the diagnosis-peer "
                    "median and the stay is under two days.",
                    "Audit providers whose approval ratio is unusually low compared to peers.",
                    "Reject any claim where the billed amount is more than three times the peer "
                    "median.",
                ],
            )
            request = st.text_area("Policy request", value=example, height=90)
            scenario_id = st.selectbox("Attach to scenario", registry.scenarios(),
                                       index=registry.scenarios().index("ANL-01"))
            if st.button("Draft the control", type="primary"):
                draft = result.ai.rule_author.draft(
                    request, author=state.username, scenario_id=scenario_id)
                st.session_state["fwa_draft"] = draft

            draft = st.session_state.get("fwa_draft")
            if draft is not None:
                st.markdown("### The drafted contract")
                mono(draft.yaml_text)
                if draft.valid:
                    st.success(
                        "The deterministic validator accepted this draft. It satisfies the "
                        "contract and the disposition-legality rule."
                    )
                else:
                    st.error("**The deterministic validator REJECTED this draft.**")
                    for error in draft.errors:
                        st.markdown(f"> {error}")
                    note(
                        "This is the mechanism working, not failing. The LLM drafted what was "
                        "asked for — including, in the third example above, a statistical test "
                        "that would deny a claim — and the validator refused it by the same code "
                        "path that validates a hand-written rule. There is no LLM-specific "
                        "check, and no override."
                    )

                st.markdown("### Diff against the registry")
                mono(draft.diff)

                st.markdown("### Ten-category test stubs generated alongside")
                st.caption(
                    "the test design requires all ten categories for EVERY control regardless of type. The "
                    "stubs exist before activation is even proposed."
                )
                st.markdown("\n".join(f"- `{c}`" for c in sorted(draft.test_fixture_stubs)))
                with st.expander("Show a stub"):
                    mono(draft.test_fixture_stubs["positive_fixture"])

                if draft.valid and st.button("Enter into the registry (shadow only)"):
                    control = result.ai.rule_author.register(draft, author=state.username)
                    st.success(
                        f"{control.rule_id} entered the registry in **shadow**. "
                        f"You authored it, so you may not approve its activation — a different "
                        f"policy owner must."
                    )

    # ======================================================================
    with tab_approve:
        st.markdown("## Shadow → active")
        if not state.can(Permission.APPROVE_RULE_ACTIVATION):
            note(
                f"The {state.role.value} role may not approve an activation. Only a policy owner "
                f"may, and never one who authored the rule."
            )
            return
        shadow = registry.by_status(RuleStatus.SHADOW)
        st.caption(f"{len(shadow):,} control(s) in shadow.")
        authored = [c for c in shadow if c.authored_by]
        options = authored or shadow[:50]
        rule_id = st.selectbox("Control", [c.rule_id for c in options])
        control = registry.get(rule_id)
        st.markdown(
            f"**{control.rule_id}** — authored by `{control.authored_by or 'catalogue'}`, "
            f"in shadow since `{control.shadow_since or 'load'}`."
        )
        signals = result.signals.by_rule(rule_id, state.tenant_id)
        tiles([
            ("Shadow volume", f"{len(signals):,}", "signals this run"),
            ("Shadow period", f"{result.config.get('shadow_period_min_days')} days required",
             "cfg.shadow_period_min_days — a single retrospective run does not satisfy it"),
            ("Review yield", "NOT_MEASURABLE", "needs reviewer dispositions"),
        ])
        sign_off = st.text_area("Sign-off note (required)", height=90)
        if st.button("Approve activation", type="primary"):
            if len(sign_off.strip()) < 10:
                st.error("A sign-off note of at least 10 characters is required.")
            else:
                try:
                    activated = registry.activate(rule_id, actor=state.username, note=sign_off)
                except SeparationOfDutiesError as exc:
                    st.error(str(exc))
                else:
                    result.audit.record(
                        "RULE_ACTIVATED", actor=state.username, actor_role=state.role.value,
                        tenant_id=state.tenant_id, subject=rule_id, reason=sign_off,
                        after={"status": activated.status.value},
                    )
                    st.success(
                        f"{rule_id} is now `active`, approved by {state.username} and recorded in "
                        f"the audit log."
                    )
                    st.warning(
                        "Note what this run could NOT demonstrate: the statistical-rule release gate's "
                        "gate requires a PROSPECTIVE shadow period and a review yield above an "
                        "agreed baseline. Neither is satisfiable from a single retrospective run, "
                        "which is why the release-gate report marks that gate NOT_ASSESSABLE."
                    )
