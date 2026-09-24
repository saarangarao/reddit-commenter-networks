"""
r/ApplyingToCollege reply-network dashboard.

Lets you retune the same parameters explored in main.ipynb — Louvain resolution,
how many communities get drawn, how many members per community, TF-IDF legend
sensitivity — and see the graph and its keyword legend update live.

Run locally:   streamlit run app.py
Deploy live:   push this repo to GitHub, then deploy at streamlit.io/cloud
               (GitHub Pages can't run this — it's a live Python app, not a
               static file. See static_site/ for a GitHub-Pages-compatible
               precomputed version of the same visualization.)
"""

import collections
import math
import os
import pickle

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="A2C Reply Network", layout="wide", page_icon="🕸️")


# ── Data loading (once per server process) ──────────────────────────
@st.cache_resource
def load_data():
    with open(os.path.join(DATA_DIR, "graph.pkl"), "rb") as f:
        G = pickle.load(f)
    with open(os.path.join(DATA_DIR, "metrics.pkl"), "rb") as f:
        metrics_df = pickle.load(f)
    word_cache_path = os.path.join(DATA_DIR, "user_word_counts.pkl")
    if os.path.exists(word_cache_path):
        with open(word_cache_path, "rb") as f:
            user_word_counts = pickle.load(f)
    else:
        user_word_counts = {}
    return G, metrics_df, user_word_counts


G, metrics_df, user_word_counts = load_data()
UG = G.to_undirected()
HAS_VOCAB = len(user_word_counts) > 0


# ── Cached computation (keyed only on the actual parameters, not on G/UG —
# those are closed-over module-level objects loaded once, so Streamlit never
# re-hashes the 9,605-node graph on every slider tick) ──────────────────────
@st.cache_data(show_spinner=False)
def louvain_partition(resolution, seed=42):
    comms = nx.community.louvain_communities(UG, weight="weight", seed=seed, resolution=resolution)
    # Sort deterministically — Louvain returns sets, and Python randomises set
    # iteration order per process, so without this the same (resolution, seed)
    # can silently hand back communities in a different order on every rerun.
    return sorted((sorted(c) for c in comms if len(c) >= 30), key=lambda c: (-len(c), c[0]))


@st.cache_data(show_spinner=False)
def build_subgraph(resolution, n_communities, peel_to, seed=42):
    communities = louvain_partition(resolution, seed)
    node_comm = {}
    for ci, comm in enumerate(communities[:n_communities]):
        sub = UG.subgraph(comm).copy()
        while sub.number_of_nodes() > peel_to:
            sub.remove_node(min(sub.nodes(), key=lambda n: (sub.degree(n, weight="weight"), n)))
        for n in sub.nodes():
            node_comm[n] = ci

    # Build explicitly rather than G.subgraph(...).copy(): that returns a VIEW
    # whose node filter holds the induced nodes in a set, and for a small
    # induced set the view iterates that set — hash-randomised per process.
    view = G.subgraph(node_comm.keys())
    H = nx.DiGraph()
    H.add_nodes_from((n, G.nodes[n]) for n in sorted(node_comm))
    for u, v in sorted(view.edges()):
        H.add_edge(u, v, **view.edges[u, v])
    H.remove_nodes_from(sorted(nx.isolates(H)))
    return H, node_comm, len(communities)


@st.cache_data(show_spinner=False)
def global_stopwords(top_n=180):
    freq = collections.Counter()
    for c in user_word_counts.values():
        freq.update(c)
    return {w for w, _ in freq.most_common(top_n)}


@st.cache_data(show_spinner=False)
def cluster_keywords(resolution, n_communities, seed, min_users, min_corpus, k):
    """TF-IDF keywords per cluster, pooled from each community's FULL
    membership (not the peeled/drawn subset — a 45-node sample is thin
    enough that one heavy poster can dominate a cluster's apparent topic).
    Smoothed idf (sklearn convention) so a word used in every cluster is
    down-weighted rather than deleted outright."""
    if not HAS_VOCAB:
        return {}
    communities = louvain_partition(resolution, seed)[:n_communities]
    stop = global_stopwords()

    counts = [collections.Counter() for _ in communities]
    userdf = [collections.Counter() for _ in communities]
    for ci, comm in enumerate(communities):
        for n in comm:
            c = user_word_counts.get(n)
            if not c:
                continue
            filtered = {w: v for w, v in c.items() if w not in stop}
            counts[ci].update(filtered)
            userdf[ci].update(filtered.keys())

    n = len(counts)
    totals = [sum(c.values()) for c in counts]
    vocab = set()
    for c in counts:
        vocab.update(c)
    corpus_freq = {w: sum(c[w] for c in counts) for w in vocab}
    doc_freq = {w: sum(1 for c in counts if c[w] > 0) for w in vocab}

    def top(ci):
        scored = []
        for w, cnt in counts[ci].items():
            if userdf[ci][w] < min_users or corpus_freq[w] < min_corpus:
                continue
            tf = cnt / totals[ci] if totals[ci] else 0
            idf = math.log((1 + n) / (1 + doc_freq[w])) + 1
            scored.append((tf * idf, w))
        scored.sort(reverse=True)
        return [w for _, w in scored[:k]]

    return {ci: top(ci) for ci in range(n)}


@st.cache_data(show_spinner=False)
def stability_ari(resolution, n_seeds=6):
    """Mean Adjusted Rand Index across n_seeds independent Louvain runs at
    this resolution — how much the partition would change if you'd happened
    to run it with a different random seed. High = trustworthy grouping,
    low = mostly an artifact of that particular run."""
    nodes = sorted(UG.nodes())
    idx = {n: i for i, n in enumerate(nodes)}

    def labels(seed):
        parts = nx.community.louvain_communities(UG, weight="weight", seed=seed, resolution=resolution)
        lab = [0] * len(nodes)
        for ci, c in enumerate(parts):
            for n in c:
                lab[idx[n]] = ci
        return lab

    def ari(a, b):
        n = len(a)
        ct = collections.Counter(zip(a, b))
        s = sum(v * (v - 1) / 2 for v in ct.values())
        ca = collections.Counter(a)
        cb = collections.Counter(b)
        sa = sum(v * (v - 1) / 2 for v in ca.values())
        sb = sum(v * (v - 1) / 2 for v in cb.values())
        tot = n * (n - 1) / 2
        exp = sa * sb / tot
        mx = (sa + sb) / 2
        return (s - exp) / (mx - exp) if mx != exp else 1.0

    import itertools
    labs = [labels(s) for s in range(1, n_seeds + 1)]
    scores = [ari(a, b) for a, b in itertools.combinations(labs, 2)]
    return sum(scores) / len(scores), min(scores)


COMM_COLORS = [
    "#7c6af7", "#00d4aa", "#e84393", "#ffa94d", "#4dabf7",
    "#c0eb75", "#ffd43b", "#ff8787", "#39a57a", "#eb88db",
    "#ff6b6b", "#4ecdc4", "#f7b731", "#a29bfe", "#fd79a8",
]
ROLE_COLORS = {
    "Advisor": "#7c6af7",
    "Debater": "#e84393",
    "Bridge": "#00d4aa",
    "Bridge-Debater": "#ffffff",
}

PHYSICS = """
{
  "physics": {
    "stabilization": {"enabled": true, "iterations": 800, "updateInterval": 50},
    "minVelocity": 0.75,
    "maxVelocity": 50,
    "barnesHut": {
      "gravity": -2500, "centralGravity": 0.1, "springLength": 150,
      "springConstant": 0.06, "damping": 0.9, "avoidOverlap": 0.2
    }
  }
}
"""


def render_html(H, node_comm, color_by, keywords):
    net = Network(height="760px", width="100%", directed=False, bgcolor="#0a0a0f", font_color="#e8e8f0")

    pr = metrics_df.set_index("user")["pagerank"].to_dict()
    roles = metrics_df.set_index("user")["role"].to_dict()
    max_pr = max((pr.get(n, 0) for n in H.nodes()), default=1) or 1

    for node in H.nodes():
        ci = node_comm[node]
        role = roles.get(node, "Advisor")
        p = pr.get(node, 0)
        nd = G.nodes[node]
        color = COMM_COLORS[ci % len(COMM_COLORS)] if color_by == "Community" else ROLE_COLORS.get(role, "#7c6af7")
        tip = (
            f"{node},  Cluster: {ci},  Role: {role},  Influence: {p:.4f},  "
            f"Comments: {nd.get('comment_count', 0)},  Score: {nd.get('total_score', 0)},  "
            f"Controversial: {nd.get('controversial_count', 0)}"
        )
        net.add_node(node, label=" ", size=10 + (p / max_pr) * 25, color=color, title=tip)

    weights = [d["weight"] for _, _, d in H.edges(data=True)]
    max_w = max(weights) if weights else 1
    log_max = math.log1p(max_w) or 1
    for src, tgt, data in H.edges(data=True):
        w = data["weight"]
        strength = math.log1p(w) / log_max
        alpha = int(25 + strength * 128)
        net.add_edge(src, tgt, value=w, length=300 - strength * 250, color=f"#ffffff{alpha:02x}")

    net.set_options(PHYSICS)
    html = net.generate_html(notebook=False)

    if color_by == "Community" and keywords:
        rows = "".join(
            f'<div style="display:flex;align-items:baseline;gap:8px;margin:4px 0;">'
            f'<span style="width:9px;height:9px;border-radius:50%;flex:none;'
            f'background:{COMM_COLORS[ci % len(COMM_COLORS)]};"></span>'
            f'<span><b style="opacity:.9">{ci}</b> — {", ".join(kws) if kws else "no distinctive terms"}</span>'
            f"</div>"
            for ci, kws in sorted(keywords.items())
        )
        legend = (
            '<div style="position:fixed;top:14px;right:14px;max-width:360px;'
            'background:rgba(10,10,15,.90);border:1px solid #322f47;border-radius:8px;'
            'padding:12px 15px;font:12px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;'
            'color:#e8e8f0;z-index:1000;">'
            '<div style="font-size:10px;font-weight:600;letter-spacing:.05em;'
            'text-transform:uppercase;opacity:.55;margin-bottom:7px;">Cluster keywords (TF-IDF)</div>'
            f"{rows}</div>"
        )
        html = html.replace("<body>", "<body>\n" + legend, 1)

    return html


# ── Sidebar controls ─────────────────────────────────────────────────
st.sidebar.title("Parameters")

resolution = st.sidebar.slider(
    "Resolution (γ)", 0.5, 10.0, 6.0, step=0.5,
    help="Higher = stricter about grouping hubs together, so communities split apart more. "
         "gamma=6 tested best on this graph (see main.ipynb) across stability, community "
         "count, and whether the resulting groups map to real topics.",
)
n_communities = st.sidebar.slider(
    "Communities to draw", 2, 20, 10,
    help="Only the N largest communities (of however many Louvain finds at this "
         "resolution) get drawn — the rest exist in the data but aren't shown.",
)
peel_to = st.sidebar.slider(
    "Max members per community", 20, 300, 100, step=10,
    help="Each drawn community is trimmed to its most-connected members, dropping the "
         "weakest-tied ones first, to keep the picture legible.",
)
color_by = st.sidebar.radio("Color nodes by", ["Community", "Role"])

with st.sidebar.expander("TF-IDF legend tuning", expanded=False):
    kw_k = st.slider("Keywords per cluster", 3, 10, 6)
    min_users = st.slider("Min distinct users using a word", 1, 15, 4)
    min_corpus = st.slider("Min total uses across all clusters", 1, 100, 15)

with st.sidebar.expander("Partition stability (slow)", expanded=False):
    st.caption("Reruns Louvain 6 times with different random seeds at the current "
               "resolution and measures how much they agree (Adjusted Rand Index). "
               "Not run automatically — it takes a few seconds.")
    if st.button("Compute ARI at current resolution"):
        with st.spinner("Running 6 Louvain passes..."):
            mean_ari, min_ari = stability_ari(resolution)
        st.metric("Mean ARI (6 seeds)", f"{mean_ari:.3f}")
        st.metric("Min pairwise ARI", f"{min_ari:.3f}")
        st.caption("Above ~0.75 is generally trustworthy; below ~0.6 means the grouping "
                   "is fairly sensitive to which random seed happened to run.")

if not HAS_VOCAB:
    st.sidebar.warning(
        "user_word_counts.pkl not found — TF-IDF legend disabled. "
        "Build it by running the notebook's cache cell once."
    )


# ── Main panel ────────────────────────────────────────────────────────
st.title("r/ApplyingToCollege Reply Network")
st.caption(
    "Every user who repeatedly replies to another user is an edge, weighted by how "
    "often it happened. Communities are found with Louvain at the resolution set in "
    "the sidebar — never treat one resolution's groups as *the* communities."
)

H, node_comm, n_communities_found = build_subgraph(resolution, n_communities, peel_to)
UH = H.to_undirected()
intra = sum(1 for a, b in UH.edges() if node_comm[a] == node_comm[b])
edge_total = UH.number_of_edges()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Communities found (≥30 members)", n_communities_found)
c2.metric("Nodes drawn", H.number_of_nodes())
c3.metric("Edges drawn", edge_total)
c4.metric("Intra-community edges", f"{intra/edge_total:.0%}" if edge_total else "—")
c5.metric("Share of graph shown", f"{H.number_of_nodes()/G.number_of_nodes():.1%}")

keywords = cluster_keywords(resolution, n_communities, 42, min_users, min_corpus, kw_k) if HAS_VOCAB else {}

html = render_html(H, node_comm, color_by, keywords)
components.html(html, height=780, scrolling=False)

if HAS_VOCAB and keywords:
    st.subheader("Cluster keywords")
    rows = [{"Cluster": ci, "Members (full community)": None, "TF-IDF keywords": ", ".join(kws) if kws else "—"}
            for ci, kws in sorted(keywords.items())]
    communities_full = louvain_partition(resolution, 42)[:n_communities]
    for row, comm in zip(rows, communities_full):
        row["Members (full community)"] = len(comm)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

with st.expander("What am I looking at?"):
    st.markdown(
        """
- **Node size** — PageRank within the drawn subgraph (bigger = more central).
- **Edge thickness and length** — both driven by the same number, reply weight
  (how many times two people replied to each other). A thick, short edge is a
  strong tie; a thin, long edge is a weak one — vis-network ignores edge
  thickness for its physics, so length was set explicitly to make weight
  actually affect the layout.
- **This is a slice, not the whole graph.** At most settings you're seeing a
  few percent of the ~9,600 users in the network — only the largest
  communities Louvain finds at this resolution, trimmed to their most
  connected members. Raise "Communities to draw" and "Max members per
  community" to see more, at the cost of a busier picture.
- **Position (left/right, near/far from center) is not fixed or reproducible**
  in this live-physics view — it comes from an unseeded random starting
  layout in the browser and will differ between reloads. Only edge length
  between two *connected* dots carries real meaning.
        """
    )
