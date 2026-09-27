"""Connections between providers, agents and patients — the graph, groups, explorer, duplicates.

Build brief §14, item 5: interactive graph, communities, ring cases and
entity-resolution candidates awaiting human confirmation.

The page is as much about what the connections *cannot* show as what they can.
The relationships in a claim file are thin and one of them is inferred, and a
picture that does not say so invites a reviewer to read more into it than it
supports. So every connection type is described in words, the inferred one is
drawn dashed and labelled, and possible duplicate records are presented as
questions awaiting a person, never as identifications.

Presentation only; identities are masked per role exactly as before.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import (
    _chart_key, audit_log, banner, bar, chart_note, chart_template, dataframe, empty_state,
    friendly_failure, friendly_table, headline_cards, help_text, mask, page_header, pipeline,
    require_page, session_banner, synthetic_banner,
)
from components.analysis_pages import edge_words, match_sentence, node_word
from fwa.auth import Permission
from fwa.presentation import aed, pct, plain_date, standard_text, status

NODE_COLOURS = {
    "member": "#1f5f9c", "provider": "#8b5a3c", "agent": "#1c6f63", "tpa": "#6b4796",
}
SUBJECT_COLUMNS = {"provider": "provider_sk", "agent": "agent_id", "member": "member_sk"}


def render() -> None:
    state = require_page("Network", Permission.VIEW_NETWORK)
    session_banner(state)
    page_header("Network")
    banner()
    synthetic_banner()

    result = pipeline()
    graph = result.graph
    if graph is None or getattr(graph, "full_graph", None) is None:
        empty_state("No connections map was built",
                    "The connections step did not run on this file. It needs claims that name the "
                    "patient, the provider and the sales agent.")
        return

    tab_overview, tab_groups, tab_explore, tab_dupes = st.tabs([
        "What the map shows", "Tightly connected groups", "Explore connections",
        "Possible duplicate records",
    ])
    with tab_overview:
        with friendly_failure("the connections overview"):
            _overview(graph)
    with tab_groups:
        with friendly_failure("the connected groups"):
            _groups(result, graph, state)
    with tab_explore:
        with friendly_failure("the connections explorer"):
            _explore(result, graph, state)
    with tab_dupes:
        with friendly_failure("the possible duplicate records"):
            _duplicates(result, state)


# =============================================================================
# Overview
# =============================================================================


def _overview(graph) -> None:
    g = graph.full_graph
    st.markdown(
        "This page draws the claim file as a **map of connections**. Each dot is a patient, a "
        "provider, a sales agent or a claims administrator; each line is a relationship that appears "
        "on the claims, such as a patient treated by a provider. Groups that are much more connected "
        "to each other than to anyone else can be worth a closer look. "
        + standard_text("not_proof"),
    )
    kinds = pd.Series([d.get("node_type") for _, d in g.nodes(data=True)]).value_counts()
    headline_cards([
        (f"{g.number_of_nodes():,}", "people and organisations on the map: "
         + ", ".join(f"{int(kinds.get(k, 0)):,} {node_word(k).lower()}s" for k in NODE_COLOURS)),
        (f"{g.number_of_edges():,}", "connections between them"),
        (f"{len(graph.snapshots):,}", "weekly snapshots: a connection only counts in the week it happened"),
    ])

    st.markdown("#### Kinds of connection")
    inventory = graph.edge_inventory()
    rows = []
    for r in inventory.itertuples(index=False):
        label, meaning = edge_words(r.edge_type)
        rows.append({
            "Connection": label,
            "In this file": "Yes" if r.present else "No",
            "Number": int(r.edges),
            "How we know": ("Seen on the claim" if r.observed and r.present else
                            "Inferred (not seen directly)" if r.present else "—"),
            "What it means": meaning,
            "Edge type": r.edge_type,
        })
    friendly_table(pd.DataFrame(rows), key="net_edges",
                   columns=["Connection", "In this file", "Number", "How we know", "What it means"],
                   keep_numeric=["Number"])
    st.caption("The link between an agent and a provider is inferred from appearing on the same "
               "claim; it is not a referral, and every flag that rests on it says so. Connection types "
               "marked No are missing from the file, so the checks that need them do not run.")

    summary = graph.snapshot_summary()
    if not summary.empty:
        recent = summary.tail(40).copy()
        recent["week"] = pd.to_datetime(recent["start"], errors="coerce")
        busiest = recent.loc[recent["edges"].idxmax()]
        bar(recent, "week", "edges",
            title=f"The busiest recent week began {plain_date(busiest['start'])}, with "
                  f"{int(busiest['edges']):,} connections",
            height=280, x_label="Week starting", y_label="Connections that week",
            how_to_read="Each bar is one week (the last 40 weeks in the file). Taller bars are weeks "
                        "with more claims and so more connections; a group is only judged within one "
                        "week.")

    with st.expander("Technical details: edge inventory and snapshots", expanded=False):
        dataframe(inventory)
        dataframe(summary)


# =============================================================================
# Communities
# =============================================================================


def _groups(result, graph, state) -> None:
    config = result.config
    stats = graph.community_stats
    st.markdown(
        "The engine splits each week's map into **tightly connected groups**: sets of patients, "
        "providers and agents that are linked to each other far more than to anyone else. A group "
        "is flagged only when all three hold: it is at least "
        f"{config.get('community_density_peer_multiple')} times as tightly linked as other groups of "
        f"the same size, fewer than {pct(float(config.get('community_external_flow_max')))} of its links "
        f"go to outsiders, and it has at least {config.get('community_min_distinct_members')} different "
        "patients and providers.",
        help=help_text("network_community"),
    )
    if stats is None or stats.empty:
        empty_state("No connected groups were found",
                    "The weekly maps had too few connections to form groups.")
        return

    view = stats.head(200)
    rows = []
    for r in view.itertuples(index=False):
        rows.append({
            "Group": str(r.community_id).split("#")[-1],
            "Week of": plain_date(getattr(r, "window_start", None), missing="—"),
            "Size": int(r.size),
            "How tightly linked": pct(r.internal_density),
            "Links to outsiders": pct(r.external_ratio),
            "Patients": int(r.member_count),
            "Providers": int(r.provider_count),
            "Agents": int(r.agent_count),
            "Claims": int(r.claim_count),
            "Amount billed": aed(r.exposure_aed),
            "Group id": r.community_id,
        })
    st.markdown("#### The most tightly linked groups")
    friendly_table(pd.DataFrame(rows), key="net_groups",
                   columns=["Week of", "Size", "How tightly linked", "Links to outsiders",
                            "Patients", "Providers", "Agents", "Claims"],
                   keep_numeric=["Size", "Patients", "Providers", "Agents", "Claims"], height=340)
    st.caption("Small groups are naturally tighter than big ones, so each group is compared only with "
               "groups of a similar size. Each claim's amount is counted once per group, however many "
               "people in the group it touches.")

    st.markdown("#### Groups flagged for a closer look")
    ring_signals = [s for s in result.signals if str(s.scenario_id).startswith("NET-02")]
    if not ring_signals:
        empty_state(
            "No unusually tight, closed groups were found",
            "That is a finding in itself: a connections check can only find a ring if the ring "
            "shows in the relationships recorded in the file. It cannot find links that were never "
            "written down.")
    else:
        rows = [{
            "Group": s.subject_id,
            "Size": s.evidence.get("size"),
            "How tightly linked": pct(s.evidence.get("internal_density")),
            "Links to outsiders": pct(s.evidence.get("external_ratio")),
            "Claims": s.evidence.get("claim_count"),
            "Amount billed": aed(s.exposure_aed),
            "In words": s.evidence.get("plain_language") or "",
            "Check id": s.rule_id,
        } for s in ring_signals]
        friendly_table(pd.DataFrame(rows), key="net_rings",
                       columns=["Group", "Size", "How tightly linked", "Links to outsiders", "Claims",
                                "Amount billed", "In words"])
        st.caption(standard_text("not_savings"))

    with st.expander("Technical details: community statistics (Louvain, per weekly snapshot)",
                     expanded=False):
        dataframe(view.drop(columns=["claim_ids", "nodes"], errors="ignore"), height=340)
        st.caption("Communities are detected with the Louvain method on each weekly snapshot. "
                   "Density is internal edges ÷ possible edges; the flag uses "
                   "community_density_peer_multiple against the median density of size-matched "
                   "communities and community_external_flow_max.")


# =============================================================================
# Explorer
# =============================================================================


def _explore(result, graph, state) -> None:
    st.markdown("Pick a provider, agent or patient to see who they are connected to. Showing the "
                "whole map at once would be an unreadable tangle, so the view starts from one person "
                "or organisation.")
    claims = result.claims
    choices = [k for k, col in SUBJECT_COLUMNS.items() if col in claims.columns]
    if not choices:
        empty_state("Nothing to explore", "The file does not name providers, agents or patients.")
        return
    subject_type = st.radio(
        "Start from a", choices, horizontal=True, key="net_subject_type",
        format_func=lambda k: node_word(k).lower(),
        help="Choose what kind of dot to start from. Start from a provider to see its patients and "
             "agents; start from an agent to see whose policies they sold and where those patients "
             "were treated.",
    )
    column = SUBJECT_COLUMNS[subject_type]
    counts = claims[column].dropna().value_counts().head(60)
    if counts.empty:
        empty_state("Nothing to explore", f"No {node_word(subject_type).lower()} appears on the claims.")
        return
    options = {mask(v, state): v for v in counts.index}
    label = st.selectbox(
        f"Which {node_word(subject_type).lower()}", list(options), key="net_subject",
        help="The 60 with the most claims are listed, busiest first. Identities are hidden unless "
             "your role may see them.",
    )
    centre = options[label]
    depth = st.slider(
        "How far to follow connections", 1, 2, 1, key="net_depth",
        help="How far to follow connections: 1 = direct links only, 2 = links of links. Use 2 to "
             "see, for example, the other providers that share this provider's patients; the view "
             "gets busy quickly.",
    )

    node = f"{subject_type}:{centre}"
    g = graph.full_graph
    if node not in g:
        empty_state("Not on the map", "This one has no connections on the claims.")
        return
    nodes, frontier = {node}, {node}
    for _ in range(depth):
        nxt = set()
        for n in frontier:
            nxt |= set(g.neighbors(n))
        nodes |= nxt
        frontier = nxt
    trimmed = False
    if len(nodes) > 400:
        trimmed = True
        nodes = set(sorted(nodes, key=lambda n: -g.degree(n))[:400]) | {node}
    sub = g.subgraph(nodes)
    direct = len(set(g.neighbors(node)))
    st.markdown(
        f"**{label}** has {direct:,} direct connections."
        + (f" Following links of links reaches {sub.number_of_nodes() - 1:,} in total." if depth == 2 else "")
        + (" Only the 400 most connected are drawn." if trimmed else ""))
    _plot_graph(sub, node, state)
    chart_note("The large dot is the one you picked. Colours show the kind of dot (see the legend). "
               "Solid lines are connections seen on the claims; dotted lines are inferred from an "
               "agent and a provider appearing on the same claim. Hover over a dot to see who it is.")


def _plot_graph(sub: nx.Graph, centre: str, state) -> None:
    positions = nx.spring_layout(sub, seed=7, k=0.55)
    edge_traces = []
    for kind, dash, name in (("observed", "solid", "Seen on the claims"),
                             ("inferred", "dot", "Inferred (same claim)")):
        xs, ys = [], []
        for u, v, data in sub.edges(data=True):
            if bool(data.get("inferred", False)) != (kind == "inferred"):
                continue
            xs += [positions[u][0], positions[v][0], None]
            ys += [positions[u][1], positions[v][1], None]
        if xs:
            edge_traces.append(go.Scatter(
                x=xs, y=ys, mode="lines", hoverinfo="none", name=name,
                line=dict(width=0.8, color="rgba(128,128,128,0.45)", dash=dash),
            ))

    node_traces = []
    for kind, colour in NODE_COLOURS.items():
        members = [n for n in sub.nodes if sub.nodes[n].get("node_type") == kind]
        if not members:
            continue
        word = node_word(kind)
        node_traces.append(go.Scatter(
            x=[positions[n][0] for n in members],
            y=[positions[n][1] for n in members],
            mode="markers", name=word,
            marker=dict(
                size=[16 if n == centre else 8 for n in members],
                color=colour,
                line=dict(width=[2.4 if n == centre else 0.6 for n in members], color="#ffffff"),
            ),
            text=[f"{word}: {mask(sub.nodes[n].get('label', ''), state)}" for n in members],
            hoverinfo="text",
        ))

    fig = go.Figure(edge_traces + node_traces)
    fig.update_layout(
        template=chart_template(), height=520, showlegend=True,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=8, r=8, t=8, b=8),
    )
    st.plotly_chart(fig, width="stretch", key=_chart_key(fig, "graph"))


# =============================================================================
# Entity resolution
# =============================================================================


def _duplicates(result, state) -> None:
    st.markdown(
        "Sometimes one real organisation appears under two provider records. The engine lists "
        "**possible duplicate records awaiting a person's confirmation**: pairs of providers that "
        "share many of the same patients and agents and bill in a similar way. Nothing is ever "
        "merged automatically, at any level of similarity. A match here is a question, not an "
        "identification: the file has no bank account, phone, address or owner details to prove "
        "two records are the same.",
        help=help_text("entity_resolution"),
    )
    candidates = list(result.entity_candidates[:100])
    if not candidates:
        empty_state("No possible duplicates",
                    "No pair of providers is similar enough to be worth a person's time.")
        return
    rows, pairs = [], {}
    for c in candidates:
        pair = f"{mask(c.entity_a, state)} and {mask(c.entity_b, state)}"
        pairs[pair] = c
        rows.append({
            "Pair": pair,
            "How similar": pct(c.confidence),
            "What they share": match_sentence(c.matched_fields),
            "Merged automatically?": "No, never",
            "Status": status(c.status).label,
            "Similarity score": c.confidence,
        })
    friendly_table(pd.DataFrame(rows), key="net_dupes",
                   columns=["Pair", "How similar", "What they share", "Merged automatically?", "Status"],
                   height=320)

    if state.can(Permission.CONFIRM_ENTITY_MERGE):
        st.markdown("#### Confirm a match")
        st.caption("Confirming records your judgement, with your name and reason, in the audit log. "
                   "It does not merge any records.")
        with st.form("confirm_merge"):
            pair = st.selectbox(
                "Which pair", list(pairs),
                help="The pair you have checked and believe are the same organisation.")
            reason = st.text_input(
                "Why you believe they are the same (at least 10 characters)",
                help="For example the documents or registry entries you checked. The reason is "
                     "saved in the audit log with your name.")
            if st.form_submit_button("Confirm this match"):
                if len(reason.strip()) < 10:
                    st.error("Please give a reason of at least 10 characters.")
                else:
                    c = pairs[pair]
                    try:
                        from fwa.audit.log import AuditEventType

                        audit_log().record(
                            AuditEventType.ENTITY_MERGE_CONFIRMED, actor=state.username,
                            actor_role=state.role.value, tenant_id=state.tenant_id,
                            subject=f"{c.entity_a}~{c.entity_b}", reason=reason.strip(),
                            after={"confidence": c.confidence, "matched_fields": c.matched_fields},
                        )
                        st.success(f"Your confirmation for {pair} was saved to the audit log. Nothing "
                                   "is merged automatically: a confirmation records a person's "
                                   "judgement.")
                    except Exception as exc:  # noqa: BLE001
                        from common import friendly_error

                        friendly_error("Couldn't save the confirmation.",
                                       "Nothing was recorded. Try again, or ask an administrator to "
                                       "check the audit log.", exc)

    with st.expander("Technical details: match candidates", expanded=False):
        dataframe(pd.DataFrame([c.to_row() for c in candidates]).assign(
            entity_a=lambda f: f["entity_a"].map(lambda v: mask(v, state)),
            entity_b=lambda f: f["entity_b"].map(lambda v: mask(v, state))))
        st.caption("Behavioural matching: overlapping patients (weight 0.40), shared agents (0.20), "
                   "shared administrators (0.10), shared diagnoses (0.10) and billing-profile "
                   "closeness (0.20). The automatic-merge threshold is set above 1.0, so it can never "
                   "be reached.")
