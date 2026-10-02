"""
Regenerate the GitHub Pages site's graph snapshots and manifest from the
current graph.pkl / metrics.pkl / user_word_counts.pkl.

    python build_site.py

Writes docs/graphs/res_<gamma>.html for each gamma in RESOLUTIONS and rewrites
docs/manifest.json. docs/index.html reads the manifest, so it needs no changes
unless you change how many snapshots there are (its intro text mentions 9).

Uses exactly the same pipeline as make_graph.py (imported from it), with the
same defaults: 10 communities of at most 100 members, seed 42, default TF-IDF
settings. The snapshots have no legend of their own — index.html draws it
from the manifest — and load vis-network from a CDN to keep them small.
"""

import json
import os

from make_graph import (build_subgraph, cluster_keywords, find_communities,
                        load_data, render, summarize)

RESOLUTIONS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
N_COMMUNITIES = 10
PEEL_TO = 100
MIN_SIZE = 30
SEED = 42

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")


def main():
    G, metrics_df, user_word_counts = load_data(ROOT)
    UG = G.to_undirected()
    print(f"graph: {G.number_of_nodes():,} users, {G.number_of_edges():,} directed edges "
          f"({UG.number_of_edges():,} reply pairs)")

    entries = []
    for gamma in RESOLUTIONS:
        all_comms = find_communities(UG, gamma, SEED, MIN_SIZE)
        drawn = all_comms[:N_COMMUNITIES]
        H, node_comm = build_subgraph(G, UG, drawn, PEEL_TO)
        keywords = cluster_keywords(drawn, user_word_counts, k=6, min_users=4, min_corpus=15, n_stop=180)

        rel = f"graphs/res_{gamma:.1f}.html"
        render(G, H, node_comm, metrics_df, "community", None, os.path.join(DOCS, rel),
               cdn_resources="remote", height="760px")
        entry = summarize(G, H, node_comm, all_comms, gamma, keywords, rel)
        entries.append(entry)
        print(f"  gamma={gamma:g}: {entry['communities_found']} communities, "
              f"{entry['nodes_drawn']} nodes, {entry['edges_drawn']} edges, {entry['intra_pct']}% intra")

    with open(os.path.join(DOCS, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"resolutions": entries}, f, indent=2)
        f.write("\n")
    print("wrote docs/manifest.json")


if __name__ == "__main__":
    main()
