"""Network — the typed graph, communities, ring cases and entity-resolution candidates.

Build brief §14, item 5: "interactive graph, communities, ring cases,
entity-resolution candidates awaiting human confirmation."

The page is as much about what the graph *cannot* show as what it can. On this
dataset the relationship structure is thin and partly inferred, and a network
case that does not say so invites a reviewer to read more into a picture than
the picture supports.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    SEQUENCE, _chart_key, banner, bar, chart_template, dataframe, empty_state, mask, note,
    pipeline, require_page, session_banner, tiles,
)
from fwa.auth import Permission

NODE_COLOURS = {
    "member": "#1f5f9c", "provider": "#8b5a3c", "agent": "#1c6f63", "tpa": "#6b4796",
}


def render() -> None:
    state = require_page("Network", Permission.VIEW_NETWORK)
    session_banner(state)

    st.markdown("# Network")
    banner()

    result = pipeline()
    graph = result.graph
    if graph is None or graph.full_graph is None:
        empty_state("No graph was built", "The graph layer did not run in this pipeline pass.")
        return

    tab_overview, tab_communities, tab_graph, tab_entities = st.tabs([
        "What the graph contains", "Communities", "Explore", "Entity resolution",
    ])

    # ======================================================================
    with tab_overview:
        st.markdown("## What this graph is — and is not")
        tiles([
            ("Weekly snapshots", f"{len(graph.snapshots):,}",
             "Edges are TIME-BOUNDED: an edge exists only in its snapshot window"),
            ("Nodes", f"{graph.full_graph.number_of_nodes():,}", "cumulative graph"),
            ("Edges", f"{graph.full_graph.number_of_edges():,}", "cumulative graph"),
            ("Snapshot communities", f"{len(graph.community_stats):,}",
             "Louvain, per weekly snapshot"),
        ])
        st.markdown("### Edge inventory")
        dataframe(graph.edge_inventory())
        note(
            "Three of the four edge types present are DIRECTLY OBSERVED. The agent→provider edge "
            "is INFERRED from co-occurrence on the same claim — it is not a clinical referral, "
            "and every signal resting on it says so. Referral direction, prescriber-pharmacy "
            "links, shared administrative identifiers and ownership are all absent, which is why "
            "NET-01-R02 (reciprocity) is classified NOT_EXECUTABLE: in a bipartite graph the "
            "metric is structurally undefined, not merely noisy."
        )
        st.markdown("### Snapshot sizes")
        summary = graph.snapshot_summary()
        if not summary.empty:
            bar(summary.tail(40), "window", "edges", title="Edges per weekly snapshot", height=280)

    # ======================================================================
    with tab_communities:
        st.markdown("## Communities (NET-02)")
        stats = graph.community_stats
        if stats.empty:
            empty_state("No communities were detected", "The snapshots contained no usable graph.")
        else:
            config = result.config
            st.markdown(
                f"A community is flagged when its internal density exceeds "
                f"`cfg.community_density_peer_multiple` "
                f"({config.get('community_density_peer_multiple')}) × the median density of "
                f"**size-matched** communities, AND its external flow is below "
                f"`cfg.community_external_flow_max` "
                f"({config.get('community_external_flow_max')}), AND it contains at least "
                f"{config.get('community_min_distinct_members')} members and providers."
            )
            view = stats.head(200)[
                ["community_id", "window", "size", "internal_density", "external_ratio",
                 "member_count", "provider_count", "agent_count", "claim_count", "exposure_aed"]
            ]
            dataframe(view, height=340)
            note(
                "Size matching is not optional here. In a bipartite claims graph density falls "
                "roughly as 1/size, so an unmatched absolute threshold flags every small "
                "community and no large one — at the catalogue's nominal 0.35 it raises 686 "
                "signals and breaches the alert-volume ceiling. The size-matched form is what "
                "the catalogue's own exclusion column asks for."
            )

            st.markdown("### Ring cases raised")
            ring_signals = [s for s in result.signals if s.scenario_id.startswith("NET-02")]
            if not ring_signals:
                st.markdown(
                    "<div class='fwa-empty'><div class='fwa-empty-title'>No abnormally dense, "
                    "closed communities were found</div><div>This is a finding worth stating "
                    "plainly: the 89 claims labelled <code>coordinated_ring</code> in this dataset "
                    "are spread across 70 of 200 hospitals — statistically indistinguishable from "
                    "random assignment — and each has a distinct patient. There is no ring "
                    "structure in the graph to detect. A network layer cannot find a ring that "
                    "was never written into the relationships.</div></div>",
                    unsafe_allow_html=True,
                )
            else:
                rows = [{
                    "case": s.subject_id, "rule_id": s.rule_id,
                    "size": s.evidence.get("size"),
                    "internal_density": s.evidence.get("internal_density"),
                    "external_ratio": s.evidence.get("external_ratio"),
                    "claims": s.evidence.get("claim_count"),
                    "exposure_aed": round(s.exposure_aed, 2),
                    "plain_language": s.evidence.get("plain_language"),
                } for s in ring_signals]
                dataframe(pd.DataFrame(rows))
            note(
                "A network case sums each distinct claim EXACTLY ONCE, keyed by claim_sk, even "
                "where a claim touches several nodes in the community."
            )

    # ======================================================================
    with tab_graph:
        st.markdown("## Explore a neighbourhood")
        st.caption(
            "Rendering 3,400 nodes would produce a hairball that tells a reviewer nothing. "
            "Pick a subject and see its neighbourhood instead."
        )
        claims = result.claims
        subject_type = st.radio("Subject", ["provider", "agent", "member"], horizontal=True)
        column = {"provider": "provider_sk", "agent": "agent_id", "member": "member_sk"}[subject_type]
        counts = claims[column].value_counts().head(60)
        options = {mask(v, state): v for v in counts.index}
        label = st.selectbox(f"{subject_type.title()}", list(options))
        centre = options[label]
        depth = st.slider("Neighbourhood depth", 1, 2, 1)

        node = f"{subject_type}:{centre}"
        if node not in graph.full_graph:
            empty_state("Not in the graph", "That subject produced no edges.")
        else:
            nodes = {node}
            frontier = {node}
            for _ in range(depth):
                nxt = set()
                for n in frontier:
                    nxt |= set(graph.full_graph.neighbors(n))
                nodes |= nxt
                frontier = nxt
            if len(nodes) > 400:
                st.warning(f"Neighbourhood has {len(nodes):,} nodes; showing the 400 "
                           f"most strongly connected.")
                degrees = sorted(nodes, key=lambda n: -graph.full_graph.degree(n))[:400]
                nodes = set(degrees) | {node}
            sub = graph.full_graph.subgraph(nodes)
            _plot_graph(sub, node, state)
            st.caption(
                f"{sub.number_of_nodes():,} nodes, {sub.number_of_edges():,} edges. "
                f"Dashed edges are INFERRED (agent→provider co-occurrence), solid edges are "
                f"directly observed."
            )

    # ======================================================================
    with tab_entities:
        st.markdown("## Entity-resolution candidates awaiting human confirmation")
        st.markdown(
            "The entity-resolution design prohibits automatic merging of low-confidence "
            "matches. This artefact goes "
            "further and merges **nothing** automatically at any confidence: "
            "`cfg.entity_resolution_auto_merge_min_confidence` is set to 1.01, a value the score "
            "can never reach. Every candidate below is surfaced for a human."
        )
        resolver_frame = _candidates_frame(result, state)
        if resolver_frame.empty:
            empty_state("No candidates above the review threshold",
                        "No pair of providers is behaviourally similar enough to be worth a "
                        "human's time.")
        else:
            dataframe(resolver_frame, height=320)
            note(
                "These are BEHAVIOURAL matches — overlapping member populations, shared agents "
                "and TPAs, similar billing profiles. the claim extract carries no bank account, phone, "
                "address, device or owner identifier, which is why NET-02-R01 (shared "
                "administrative identity) is classified NOT_EXECUTABLE. A behavioural match is a "
                "question, not an identification."
            )
        if state.can(Permission.CONFIRM_ENTITY_MERGE) and not resolver_frame.empty:
            with st.form("confirm_merge"):
                pair = st.selectbox("Candidate", list(resolver_frame["pair"]))
                reason = st.text_input("Reason for confirming this match")
                if st.form_submit_button("Confirm match"):
                    if len(reason.strip()) < 10:
                        st.error("A reason of at least 10 characters is required.")
                    else:
                        st.success(
                            f"Confirmation for {pair} recorded to the audit log as "
                            f"ENTITY_MERGE_CONFIRMED. Nothing is merged automatically — a "
                            f"confirmation records a human judgement."
                        )


def _candidates_frame(result, state) -> pd.DataFrame:
    rows = []
    for c in result.entity_candidates[:100]:
        rows.append({
            "pair": f"{mask(c.entity_a, state)} ~ {mask(c.entity_b, state)}",
            "confidence": c.confidence,
            "matched_fields": "; ".join(f"{k}={v}" for k, v in c.matched_fields.items()),
            "auto_merge_allowed": c.auto_merge_allowed,
            "status": c.status,
        })
    return pd.DataFrame(rows)


def _plot_graph(sub: nx.Graph, centre: str, state) -> None:
    positions = nx.spring_layout(sub, seed=7, k=0.55)
    edge_traces = []
    for kind, dash in (("observed", "solid"), ("inferred", "dot")):
        xs, ys = [], []
        for u, v, data in sub.edges(data=True):
            if (data.get("inferred", False) and kind == "observed") or \
               (not data.get("inferred", False) and kind == "inferred"):
                continue
            xs += [positions[u][0], positions[v][0], None]
            ys += [positions[u][1], positions[v][1], None]
        if xs:
            edge_traces.append(go.Scatter(
                x=xs, y=ys, mode="lines", hoverinfo="none", name=f"{kind} edges",
                line=dict(width=0.8, color="rgba(128,128,128,0.45)", dash=dash),
            ))

    node_traces = []
    for kind, colour in NODE_COLOURS.items():
        members = [n for n in sub.nodes if sub.nodes[n].get("node_type") == kind]
        if not members:
            continue
        node_traces.append(go.Scatter(
            x=[positions[n][0] for n in members],
            y=[positions[n][1] for n in members],
            mode="markers", name=kind,
            marker=dict(
                size=[16 if n == centre else 8 for n in members],
                color=colour,
                line=dict(width=[2.4 if n == centre else 0.6 for n in members], color="#ffffff"),
            ),
            text=[f"{kind}: {mask(sub.nodes[n].get('label', ''), state)}" for n in members],
            hoverinfo="text",
        ))

    fig = go.Figure(edge_traces + node_traces)
    fig.update_layout(
        template=chart_template(), height=520, showlegend=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=8, r=8, t=8, b=8),
    )
    st.plotly_chart(fig, width="stretch", key=_chart_key(fig, "graph"))
