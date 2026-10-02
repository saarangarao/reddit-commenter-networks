"""
Build an r/ApplyingToCollege reply-network graph with a TF-IDF cluster legend.

Reads graph.pkl, metrics.pkl and user_word_counts.pkl (produced by main.ipynb),
partitions the graph with Louvain at the resolution you choose, draws the dense
core of the largest communities, and labels each community with its most
distinctive words.

Examples:
    python make_graph.py                                  # gamma=6, 10 communities, 100 per community
    python make_graph.py -r 3 -c 8 -p 100                 # looser communities, smaller picture
    python make_graph.py -r 6 --color-by both -o out/g6   # out/g6_clusters.html + out/g6_roles.html
    python make_graph.py -r 6 --json g6.json              # also dump stats + keywords (docs/manifest.json format)

Choosing --resolution: modularity has a known resolution limit (Fortunato &
Barthelemy 2007), so gamma=1 is NOT a neutral default — it merges genuinely
distinct small groups. Swept on this graph, gamma=6 scored best on partition
stability (ARI across seeds), number of communities found, and whether the
TF-IDF keywords read as real topics. Whatever you pick, report it: these are
"communities at resolution gamma", never "the" communities.
"""

import argparse
import collections
import json
import math
import os
import pickle
import sys

import networkx as nx
from pyvis.network import Network

# one hue per community, tuned for the dark ground (cycles past 10)
COMM_COLORS = ["#7c6af7", "#00d4aa", "#e84393", "#ffa94d",
               "#4dabf7", "#c0eb75", "#ffd43b", "#ff8787",
               "#39a57a", "#eb88db"]

ROLE_COLORS = {
    "Advisor":        "#7c6af7",
    "Debater":        "#e84393",
    "Bridge":         "#00d4aa",
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a community-coloured reply-network graph with a TF-IDF keyword legend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Graph shape ──────────────────────────────────────────────────

    # -r / --resolution (float > 0, default 6.0)
    # The gamma in Louvain's modularity: how much denser than random chance a
    # group must be to count as a community. Higher gamma penalises big groups
    # harder, so communities split into more, smaller, tighter ones.
    #   ~1    few large, loose communities; least reproducible across seeds
    #   3-5   mid-sized groups
    #   6     best on this graph (stability, community count, topic keywords)
    #   8-10  most stable, but groups get rigid and keywords turn generic
    # Changes WHICH users are grouped together, so it changes everything below.
    p.add_argument("-r", "--resolution", type=float, default=6.0,
                   help="Louvain resolution (gamma). Higher = smaller, tighter communities.")

    # -c / --communities (int >= 1, default 10)
    # How many communities to draw, taken largest first from those that pass
    # --min-size. The rest still exist (and count toward "communities found")
    # but aren't drawn and get no legend entry. Only 10 colours exist, so past
    # 10 they repeat (cluster 10 shares cluster 0's colour).
    p.add_argument("-c", "--communities", type=int, default=10,
                   help="How many of the largest communities to draw.")

    # -p / --peel-to (int >= 1, default 100 — same as the website and app.py)
    # Max members drawn per community. Bigger communities are "peeled": the
    # member with the weakest ties inside the community is removed one at a
    # time until this many remain, leaving the dense core. Smaller communities
    # are drawn whole. Purely a drawing choice — the TF-IDF keywords always use
    # the full community. Total nodes drawn <= communities x peel-to, a bit
    # less because nodes left with no edges are dropped.
    p.add_argument("-p", "--peel-to", type=int, default=100,
                   help="Max members drawn per community (weakest-tied members dropped first).")

    # --min-size (int, default 30)
    # Communities with fewer members than this are discarded before anything
    # else, so tiny cliques don't count as communities. Affects "communities
    # found" and which communities are eligible for --communities.
    p.add_argument("--min-size", type=int, default=30,
                   help="Ignore communities smaller than this.")

    # --seed (int, default 42)
    # Louvain's random seed. Same seed + same settings = same picture every
    # run. Changing it shows how much the partition depends on chance: at a
    # well-chosen resolution the groups should look mostly the same.
    p.add_argument("--seed", type=int, default=42, help="Louvain random seed.")

    # ── Output ───────────────────────────────────────────────────────

    # --color-by (community | role | both, default community)
    #   community  one colour per community, with the TF-IDF keyword legend
    #   role       colour by user role (Advisor / Debater / Bridge /
    #              Bridge-Debater, from metrics.pkl), with a role legend
    #   both       writes both files from the same set of drawn nodes
    p.add_argument("--color-by", choices=["community", "role", "both"], default="community",
                   help="Node colouring. 'both' writes two files over the same subgraph.")

    # -o / --out (path prefix, default graph_res<gamma>, e.g. graph_res6)
    # "_clusters.html" / "_roles.html" is appended. May include folders
    # (out/g6 -> out/g6_clusters.html); missing folders are created. Existing
    # files with the same name are overwritten.
    p.add_argument("-o", "--out", default=None,
                   help="Output path prefix (default: graph_res<gamma>). "
                        "Writes <out>_clusters.html and/or <out>_roles.html.")

    # --json (path, default off)
    # Also save the run's stats and keywords as JSON, in the same shape as one
    # entry of docs/manifest.json: gamma, file, communities_found, nodes_drawn,
    # edges_drawn, intra_pct, coverage_pct, keywords.
    p.add_argument("--json", default=None,
                   help="Also write stats + keywords to this JSON file.")

    # --data-dir (folder, default: the folder this script is in)
    # Where to find graph.pkl and metrics.pkl (required) and
    # user_word_counts.pkl (optional — without it there's no keyword legend).
    p.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Folder containing graph.pkl, metrics.pkl, user_word_counts.pkl.")

    # ── TF-IDF legend ────────────────────────────────────────────────
    # These only change the keyword legend, never the graph itself.
    tf = p.add_argument_group("TF-IDF legend")

    # -k / --keywords (int >= 1, default 6)
    # Words listed per cluster, highest TF-IDF first. A cluster can show fewer
    # if not enough words pass --min-users and --min-corpus.
    tf.add_argument("-k", "--keywords", type=int, default=6, help="Keywords per cluster.")

    # --min-users (int, default 4)
    # A word only counts for a cluster if at least this many DIFFERENT members
    # used it, so one prolific poster's pet phrase can't define the cluster.
    # Raise it if keywords look like one person's vocabulary; lower it if
    # small clusters show "no distinctive terms".
    tf.add_argument("--min-users", type=int, default=4,
                    help="A word must be used by at least this many distinct members of the cluster.")

    # --min-corpus (int, default 15)
    # A word must appear at least this many times across all drawn clusters
    # combined. Filters typos and one-off tokens that would otherwise score
    # very high just for being rare.
    tf.add_argument("--min-corpus", type=int, default=15,
                    help="A word must appear at least this many times across all drawn clusters.")

    # --stopwords (int, default 180)
    # Remove the N most-used words across every user in the graph before
    # scoring. Words like "school", "college", "gpa" are used by everyone on
    # A2C, so they say nothing about which cluster someone belongs to. Raise it
    # if the legend is full of generic words; 0 turns this filter off.
    tf.add_argument("--stopwords", type=int, default=180,
                    help="Drop the N most common words across ALL users (subreddit-specific stopwords).")

    args = p.parse_args()
    for name in ("communities", "peel_to", "keywords"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.resolution <= 0:
        p.error("--resolution must be positive")
    if args.out is None:
        args.out = f"graph_res{args.resolution:g}"
    return args


# ── Data loading ─────────────────────────────────────────────────────
def load_data(data_dir):
    def load(name):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    G = load("graph.pkl")
    metrics_df = load("metrics.pkl")
    if G is None or metrics_df is None:
        sys.exit(f"graph.pkl and metrics.pkl must exist in {data_dir} — run main.ipynb first.")

    user_word_counts = load("user_word_counts.pkl")
    if user_word_counts is None:
        print("warning: user_word_counts.pkl not found — TF-IDF legend disabled. "
              "Build it with the notebook's comment-text cache cell.", file=sys.stderr)
        return G, metrics_df, {}

    # If the cache was built from a different graph.pkl (or with a different
    # anonymize()), its user IDs won't match and every legend comes out empty
    # with no error — so say so loudly.
    matched = sum(1 for n in G if n in user_word_counts)
    if matched < 0.5 * G.number_of_nodes():
        print(f"warning: only {matched:,}/{G.number_of_nodes():,} graph users have word counts — "
              f"user_word_counts.pkl doesn't match graph.pkl; delete it and rebuild it "
              f"with the notebook's comment-text cache cell.", file=sys.stderr)
    return G, metrics_df, user_word_counts


# ── Communities and the drawn subgraph ───────────────────────────────
def find_communities(UG, resolution, seed, min_size):
    comms = nx.community.louvain_communities(UG, weight="weight", seed=seed, resolution=resolution)
    # Louvain returns SETS, and Python randomises string hashes per process, so
    # set iteration order changes every run. Sort explicitly or the peel below
    # takes a different path (and draws a different sample) each time.
    return sorted((sorted(c) for c in comms if len(c) >= min_size),
                  key=lambda c: (-len(c), c[0]))


def build_subgraph(G, UG, communities, peel_to):
    # Select by COMMUNITY, not by influence rank: hubs span communities rather
    # than sit inside one, so a top-N-by-influence slice has no cluster
    # structure left to draw. Keep each community's dense core instead by
    # repeatedly dropping its weakest-connected member.
    node_comm = {}
    for ci, comm in enumerate(communities):
        sub = UG.subgraph(comm).copy()
        while sub.number_of_nodes() > peel_to:
            # tie-break on the name so the victim is well-defined
            sub.remove_node(min(sub.nodes(), key=lambda n: (sub.degree(n, weight="weight"), n)))
        for n in sub.nodes():
            node_comm[n] = ci

    # Build explicitly rather than G.subgraph(...).copy(): that returns a VIEW
    # whose node order is hash-randomised per process.
    view = G.subgraph(node_comm.keys())
    H = nx.DiGraph()
    H.add_nodes_from((n, G.nodes[n]) for n in sorted(node_comm))
    for u, v in sorted(view.edges()):
        H.add_edge(u, v, **view.edges[u, v])
    H.remove_nodes_from(sorted(nx.isolates(H)))
    return H, node_comm


# ── Cluster keywords (TF-IDF) ────────────────────────────────────────
def cluster_keywords(communities, user_word_counts, k, min_users, min_corpus, n_stop):
    """Each cluster's pooled vocabulary is one "document":
      tf  = share of the cluster's words that are w
      idf = log((1+N)/(1+df)) + 1, smoothed (sklearn convention) so a word
            used by every cluster is down-weighted rather than zeroed out.
    Pooled from each community's FULL membership, not the peeled sample — a
    small sample lets one heavy poster dominate a cluster's apparent topic.
    The N most common words across all users are dropped first: "school",
    "college", "gpa" say nothing about WHICH cluster someone is in."""
    if not user_word_counts:
        return {}

    global_freq = collections.Counter()
    for c in user_word_counts.values():
        global_freq.update(c)
    stop = {w for w, _ in global_freq.most_common(n_stop)}

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

    n_clusters = len(counts)
    totals = [sum(c.values()) for c in counts]
    vocab = set().union(*counts)
    corpus_freq = {w: sum(c[w] for c in counts) for w in vocab}
    doc_freq = {w: sum(1 for c in counts if c[w] > 0) for w in vocab}

    def top(ci):
        scored = []
        for w, cnt in counts[ci].items():
            if userdf[ci][w] < min_users or corpus_freq[w] < min_corpus:
                continue
            tf = cnt / totals[ci]
            idf = math.log((1 + n_clusters) / (1 + doc_freq[w])) + 1
            scored.append((tf * idf, w))
        scored.sort(reverse=True)
        return [w for _, w in scored[:k]]

    return {ci: top(ci) for ci in range(n_clusters)}


# ── Rendering ────────────────────────────────────────────────────────
def legend_html(title, rows):
    """rows: list of (colour, bold label, text)."""
    body = "".join(
        f'<div style="display:flex;align-items:baseline;gap:8px;margin:4px 0;">'
        f'<span style="width:9px;height:9px;border-radius:50%;flex:none;background:{color};"></span>'
        f'<span><b style="opacity:.9">{label}</b>{" — " + text if text else ""}</span>'
        f'</div>'
        for color, label, text in rows
    )
    return (
        '<div style="position:fixed;top:14px;right:14px;max-width:360px;'
        'background:rgba(10,10,15,.90);border:1px solid #322f47;border-radius:8px;'
        'padding:12px 15px;font:12px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;'
        'color:#e8e8f0;z-index:1000;">'
        '<div style="font-size:10px;font-weight:600;letter-spacing:.05em;'
        f'text-transform:uppercase;opacity:.55;margin-bottom:7px;">{title}</div>'
        f"{body}</div>"
    )


def render(G, H, node_comm, metrics_df, color_by, legend, filename,
           cdn_resources="in_line", height="800px"):
    """legend: HTML from legend_html(), or None for no legend.
    cdn_resources: "in_line" embeds vis-network so the file works on its own
    without the lib/ folder next to it; "remote" loads it from a CDN instead
    (smaller files — used for the website's snapshots)."""
    net = Network(height=height, width="100%", directed=False,
                  bgcolor="#0a0a0f", font_color="#e8e8f0", cdn_resources=cdn_resources)

    pr = metrics_df.set_index("user")["pagerank"].to_dict()
    roles = metrics_df.set_index("user")["role"].to_dict()
    max_pr = max((pr.get(n, 0) for n in H.nodes()), default=0) or 1

    for node in H.nodes():
        ci = node_comm[node]
        role = roles.get(node, "Advisor")
        p = pr.get(node, 0)
        nd = G.nodes[node]
        if color_by == "community":
            color = COMM_COLORS[ci % len(COMM_COLORS)]
        else:
            color = ROLE_COLORS.get(role, "#7c6af7")
        tip = (
            f"{node},  Cluster: {ci},  Role: {role},  Influence: {p:.4f},  "
            f"Comments: {nd.get('comment_count', 0)},  Score: {nd.get('total_score', 0)},  "
            f"Controversial: {nd.get('controversial_count', 0)}"
        )
        net.add_node(node, label=" ", size=10 + (p / max_pr) * 25, color=color, title=tip)

    # Tie strength drives the LAYOUT, not just line width: vis-network ignores
    # `value` for physics, and `length` is the spring rest length, so a strong
    # tie maps to a SHORT spring. Weights are heavily skewed, so log-compress.
    max_w = max((d["weight"] for _, _, d in H.edges(data=True)), default=1)
    log_max = math.log1p(max_w) or 1
    for src, tgt, data in H.edges(data=True):
        w = data["weight"]
        strength = math.log1p(w) / log_max                  # 0..1
        alpha = int(25 + strength * 128)
        net.add_edge(src, tgt, value=w,
                     length=300 - strength * 250,           # weak 300 → strong 50
                     color=f"#ffffff{alpha:02x}")

    net.set_options(PHYSICS)
    html = net.generate_html(notebook=False)
    if legend:
        html = html.replace("<body>", "<body>\n" + legend, 1)

    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {filename}  (coloured by {color_by})")


def summarize(G, H, node_comm, all_comms, resolution, keywords, file):
    """Stats for one run, in the shape of a docs/manifest.json entry. Edges are
    counted undirected (a reply pair in both directions is one line on screen)."""
    UH = H.to_undirected()
    n_edges = UH.number_of_edges()
    intra = sum(1 for a, b in UH.edges() if node_comm[a] == node_comm[b])
    return {
        "gamma": resolution,
        "file": file,
        "communities_found": len(all_comms),
        "nodes_drawn": H.number_of_nodes(),
        "edges_drawn": n_edges,
        "intra_pct": round(100 * intra / n_edges, 1) if n_edges else 0,
        "coverage_pct": round(100 * H.number_of_nodes() / G.number_of_nodes(), 2),
        "keywords": {str(ci): kws for ci, kws in keywords.items()},
    }


def main():
    args = parse_args()
    G, metrics_df, user_word_counts = load_data(args.data_dir)
    UG = G.to_undirected()

    print(f"graph: {G.number_of_nodes():,} users, {G.number_of_edges():,} directed edges "
          f"({UG.number_of_edges():,} reply pairs)")
    print(f"louvain: resolution={args.resolution:g}, seed={args.seed} ...")
    all_comms = find_communities(UG, args.resolution, args.seed, args.min_size)
    if not all_comms:
        sys.exit(f"no communities with >= {args.min_size} members at resolution {args.resolution:g}")
    drawn = all_comms[:args.communities]

    H, node_comm = build_subgraph(G, UG, drawn, args.peel_to)
    keywords = cluster_keywords(drawn, user_word_counts, args.keywords,
                                args.min_users, args.min_corpus, args.stopwords)
    entry = summarize(G, H, node_comm, all_comms, args.resolution, keywords,
                      os.path.basename(f"{args.out}_clusters.html"))
    print(f"{entry['communities_found']} communities found (>= {args.min_size} members), "
          f"drawing {len(drawn)} | {entry['nodes_drawn']} nodes | {entry['edges_drawn']} edges | "
          f"{entry['intra_pct']}% of edges intra-community")
    for ci, comm in enumerate(drawn):
        kws = keywords.get(ci)
        print(f"  cluster {ci} ({len(comm)} members): "
              f"{', '.join(kws) if kws else '(no distinctive terms above threshold)'}")

    if args.color_by in ("community", "both"):
        legend = legend_html(
            f"Cluster keywords (TF-IDF) · γ = {args.resolution:g}",
            [(COMM_COLORS[ci % len(COMM_COLORS)], ci, ", ".join(keywords.get(ci, [])) or "no distinctive terms")
             for ci in range(len(drawn))],
        )
        render(G, H, node_comm, metrics_df, "community", legend, f"{args.out}_clusters.html")
    if args.color_by in ("role", "both"):
        legend = legend_html("Role", [(c, role, "") for role, c in ROLE_COLORS.items()])
        render(G, H, node_comm, metrics_df, "role", legend, f"{args.out}_roles.html")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
