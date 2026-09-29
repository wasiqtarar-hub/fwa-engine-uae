"""Typed, time-bounded relationship graph (manuscript §4.10).

    "The graph service materialises TYPED, TIME-BOUNDED relationship edges
     (referral, prescriber-pharmacy, shared administrative identifier,
     ownership) into WEEKLY SNAPSHOTS, computing concentration, reciprocity,
     connected components, community membership and centrality ONLY WITHIN
     COMPARABLE NODE TYPES."                                — manuscript §4.10

What this dataset actually supports, stated plainly because the difference
matters for how much weight a network case can bear:

=========================  ==========================================================
Manuscript edge type       Status here
=========================  ==========================================================
patient—hospital (claim)   **Real.** Every claim is one such edge.
agent—patient (policy)     **Real.** Every claim carries its originating agent.
agent—hospital             **Inferred from co-occurrence**, not observed. Labelled
                           ``inferred=True`` on the edge and on every signal.
hospital—tpa               **Real.** Every claim carries its administering TPA.
referral (A→B)             **Absent.** No referral records, so reciprocity
                           (NET-01-R02) is identically undefined, not merely noisy.
prescriber—pharmacy        **Absent.** No prescriptions or pharmacies as entities.
shared admin identifier    **Absent.** No bank, phone, address or device tokens.
ownership                  **Absent.** No ownership records.
=========================  ==========================================================

Edges are **time-bounded**: an edge exists only inside the weekly snapshot
window of the claim that created it. That matters for NET-02 — a community that
looks dense across three years may be nothing more than a busy hospital, while a
community that is dense *within one week* is a different observation.

Because the observed graph is **bipartite by construction** (members and agents
connect to providers, not to each other), directed metrics such as reciprocity
cannot exist here. Rather than compute a metric that is structurally zero, those
controls are classified ``NOT_EXECUTABLE_ON_THIS_DATASET``.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

__all__ = ["GraphSnapshot", "GraphService", "EDGE_TYPES", "NODE_TYPES"]

NODE_TYPES = ("member", "provider", "agent", "tpa")

EDGE_TYPES = {
    "member_provider": {"observed": True, "note": "One edge per claim. Directly observed."},
    "agent_member": {"observed": True, "note": "Policy origination. Directly observed."},
    "agent_provider": {
        "observed": False,
        "note": "INFERRED from co-occurrence on the same claim, not an observed referral. "
                "Every signal resting on this edge says so.",
    },
    "provider_tpa": {"observed": True, "note": "Claim administration. Directly observed."},
}


@dataclass
class GraphSnapshot:
    """One weekly, time-bounded graph."""

    window: str
    start: _dt.date
    end: _dt.date
    graph: nx.Graph
    claim_ids: set[str] = field(default_factory=set)
    communities: dict[str, int] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.graph.number_of_edges()


class GraphService:
    """Builds weekly snapshots and computes community structure."""

    def __init__(self, config, seed: int | None = None) -> None:
        self.config = config
        self.seed = seed if seed is not None else int(config.get("random_seed"))
        self.snapshots: list[GraphSnapshot] = []
        self.full_graph: nx.Graph | None = None
        self.full_communities: dict[str, int] = {}
        self.community_stats: pd.DataFrame = pd.DataFrame()
        self.cumulative_community_stats: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------ build

    @staticmethod
    def _node(kind: str, value: Any) -> str:
        return f"{kind}:{value}"

    def _add_claim_edges(self, g: nx.Graph, row) -> None:
        m = self._node("member", row.member_sk)
        p = self._node("provider", row.provider_sk)
        a = self._node("agent", row.agent_id)
        t = self._node("tpa", row.tpa)
        for node, kind in ((m, "member"), (p, "provider"), (a, "agent"), (t, "tpa")):
            if node not in g:
                g.add_node(node, node_type=kind, label=str(node.split(":", 1)[1]))
        amount = float(row.gross_amount_aed or 0.0)
        for u, v, etype in ((m, p, "member_provider"), (a, m, "agent_member"),
                            (a, p, "agent_provider"), (p, t, "provider_tpa")):
            if g.has_edge(u, v):
                g[u][v]["weight"] += 1
                g[u][v]["amount_aed"] += amount
                g[u][v]["claim_ids"].add(str(row.claim_sk))
            else:
                g.add_edge(
                    u, v, edge_type=etype, weight=1, amount_aed=amount,
                    claim_ids={str(row.claim_sk)},
                    observed=EDGE_TYPES[etype]["observed"],
                    inferred=not EDGE_TYPES[etype]["observed"],
                )

    def build(self, claims: pd.DataFrame) -> "GraphService":
        work = claims.copy()
        work["service_date"] = pd.to_datetime(work["service_date"])
        work["window"] = work["service_date"].dt.to_period("W").astype(str)

        # weekly, time-bounded snapshots (§4.10)
        for window, sub in work.groupby("window", sort=True):
            g = nx.Graph()
            for row in sub.itertuples(index=False):
                self._add_claim_edges(g, row)
            snap = GraphSnapshot(
                window=str(window),
                start=sub["service_date"].min().date(),
                end=sub["service_date"].max().date(),
                graph=g,
                claim_ids=set(sub["claim_sk"].astype(str)),
            )
            snap.communities = self._detect_communities(g)
            self.snapshots.append(snap)

        # The cumulative graph is kept for VISUALISATION and for entity-level
        # lookups, but community detection for NET-02 runs on the WEEKLY
        # SNAPSHOTS, because §4.10 specifies time-bounded edges and because a
        # community that is dense across three years is usually just a busy
        # hospital, whereas one that is dense inside a single week is a
        # different observation entirely.
        full = nx.Graph()
        for row in work.itertuples(index=False):
            self._add_claim_edges(full, row)
        self.full_graph = full
        self.full_communities = self._detect_communities(full)
        self.cumulative_community_stats = self._community_stats(full, self.full_communities, work)

        frames = []
        for snap in self.snapshots:
            stats = self._community_stats(snap.graph, snap.communities, work)
            if stats.empty:
                continue
            stats = stats.copy()
            stats["window"] = snap.window
            stats["window_start"] = snap.start.isoformat()
            stats["window_end"] = snap.end.isoformat()
            stats["community_id"] = stats["community_id"].map(lambda c: f"{snap.window}#{c}")
            frames.append(stats)
        self.community_stats = (
            pd.concat(frames, ignore_index=True).sort_values("internal_density", ascending=False)
            if frames else pd.DataFrame()
        )
        return self

    # ------------------------------------------------------------- community

    def _detect_communities(self, g: nx.Graph) -> dict[str, int]:
        if g.number_of_nodes() == 0:
            return {}
        communities = nx.community.louvain_communities(g, weight="weight", seed=self.seed)
        return {node: i for i, comm in enumerate(communities) for node in comm}

    def _community_stats(self, g: nx.Graph, assignment: dict[str, int], claims: pd.DataFrame) -> pd.DataFrame:
        if not assignment:
            return pd.DataFrame()
        by_comm: dict[int, list[str]] = {}
        for node, cid in assignment.items():
            by_comm.setdefault(cid, []).append(node)

        claim_lookup = claims.set_index("claim_sk")
        rows = []
        for cid, nodes in by_comm.items():
            sub = g.subgraph(nodes)
            n = sub.number_of_nodes()
            e = sub.number_of_edges()
            possible = n * (n - 1) / 2
            density = (e / possible) if possible > 0 else 0.0

            # external flow: edges leaving the community
            external = sum(
                1 for u in nodes for v in g.neighbors(u) if assignment.get(v) != cid
            )
            internal_claims: set[str] = set()
            for u, v, data in sub.edges(data=True):
                internal_claims |= set(data.get("claim_ids", ()))
            valid = [c for c in internal_claims if c in claim_lookup.index]
            exposure = float(claim_lookup.loc[valid, "gross_amount_aed"].sum()) if valid else 0.0
            kinds = {k: 0 for k in NODE_TYPES}
            for node in nodes:
                kinds[g.nodes[node]["node_type"]] = kinds.get(g.nodes[node]["node_type"], 0) + 1
            rows.append(
                {
                    "community_id": cid,
                    "size": n,
                    "internal_edges": e,
                    "internal_density": round(density, 4),
                    "external_edges": external,
                    "external_ratio": round(external / max(e + external, 1), 4),
                    "member_count": kinds.get("member", 0),
                    "provider_count": kinds.get("provider", 0),
                    "agent_count": kinds.get("agent", 0),
                    "tpa_count": kinds.get("tpa", 0),
                    "claim_count": len(valid),
                    # each distinct claim counted exactly ONCE (§4.11, §6.3)
                    "exposure_aed": round(exposure, 2),
                    "claim_ids": sorted(valid),
                    "nodes": nodes,
                }
            )
        return pd.DataFrame(rows).sort_values("internal_density", ascending=False)

    # ---------------------------------------------------------------- views

    def edge_inventory(self, dataset: Any = None) -> pd.DataFrame:
        """What edge types exist here, and which are inferred rather than observed.

        ``dataset`` (the canonical dataset) lets the notes for the edge types the
        graph does not build say whether the FILE lacks the data or only the
        graph does: "no referral records" is false of a file that has them.
        """
        rows = []
        for etype, meta in EDGE_TYPES.items():
            count = 0
            if self.full_graph is not None:
                count = sum(1 for _, _, d in self.full_graph.edges(data=True) if d.get("edge_type") == etype)
            rows.append(
                {
                    "edge_type": etype,
                    "present": count > 0,
                    "edges": count,
                    "observed": meta["observed"],
                    "note": meta["note"],
                }
            )
        in_file = _unbuilt_edge_sources(dataset)
        for missing, key, note in (
            ("referral (directed A→B)", "referral",
             "ABSENT — no referral records. NET-01-R02 reciprocity is "
             "structurally undefined, not merely unmeasured."),
            ("prescriber—pharmacy", "prescriber_pharmacy",
             "ABSENT — no prescriptions or pharmacy entities."),
            ("shared administrative identifier", "shared_identifier",
             "ABSENT — no bank, phone, address or device tokens."),
            ("ownership", "ownership", "ABSENT — no ownership records."),
        ):
            if in_file.get(key):
                note = (f"NOT BUILT INTO THE GRAPH — the file has {in_file[key]}, but this graph "
                        f"does not yet build the edge. The controls that need it read those "
                        f"records directly.")
            rows.append({"edge_type": missing, "present": False, "edges": 0, "observed": False, "note": note})
        return pd.DataFrame(rows)

    def snapshot_summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "window": s.window, "start": s.start.isoformat(), "end": s.end.isoformat(),
                    "nodes": s.n_nodes, "edges": s.n_edges, "claims": len(s.claim_ids),
                }
                for s in self.snapshots
            ]
        )

    def community_of(self, node_kind: str, value: Any) -> int | None:
        return self.full_communities.get(self._node(node_kind, value))

    def subgraph_for_community(self, community_id: int) -> nx.Graph:
        nodes = [n for n, c in self.full_communities.items() if c == community_id]
        return self.full_graph.subgraph(nodes).copy() if self.full_graph else nx.Graph()


def _unbuilt_edge_sources(dataset: Any) -> dict[str, str]:
    """For each edge type the graph does not build, what the file holds (if anything)."""
    if dataset is None:
        return {}

    def table(name: str) -> pd.DataFrame:
        try:
            frame = dataset.get(name)
        except Exception:  # an adapter without this table
            return pd.DataFrame()
        return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    def filled(frame: pd.DataFrame, columns: tuple[str, ...]) -> int:
        cols = [c for c in columns if c in frame.columns]
        if frame.empty or not cols:
            return 0
        return int(frame[cols].notna().any(axis=1).sum())

    out: dict[str, str] = {}
    referral = table("referral")
    if not referral.empty:
        out["referral"] = f"{len(referral):,} referral records"
    dispense = table("prescription_dispense")
    if not dispense.empty:
        out["prescriber_pharmacy"] = f"{len(dispense):,} prescription and dispensing records"
    provider = table("provider")
    shared = filled(provider, ("bank_account_token", "phone_token"))
    if shared:
        out["shared_identifier"] = f"bank or phone tokens for {shared:,} providers"
    owned = filled(provider, ("owner_entity_id", "ownership_changed_on"))
    if owned:
        out["ownership"] = f"ownership details for {owned:,} providers"
    return out
