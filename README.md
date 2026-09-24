# r/ApplyingToCollege Reply Network

A graph of who replies to whom on r/ApplyingToCollege, with users grouped into communities, ranked by influence, and labeled by the words their community uses most distinctively.

- **`docs/`** holds the static site served by GitHub Pages: a slider that switches between precomputed graphs at different community resolutions.
- **`app.py`** is a Streamlit dashboard that recomputes everything live, with more controls. It's the same visualization, but it needs a Python server, so GitHub Pages can't host it.
- **`main.ipynb`** is the pipeline that turns the raw comment dump into the graph and metrics.

---

## The math

### 1. From comments to a weighted, directed graph

Source: `r_ApplyingToCollege_comments.jsonl` (1.94M comments; gitignored because of its size).

Filters, in order:

| Step | Rule |
|---|---|
| Bots | Drop `[deleted]`, `AutoModerator`, `RemindMeBot`, `sneakpeek_bot`, `ApplyingToCollege-ModTeam` |
| Anonymize | Username → `user_` + first 8 hex characters of its MD5 hash |
| Time window | `2024-01-01 ≤ created_utc < 2025-09-01` |
| Active users | Keep only authors with **≥ 10** comments in the window |

**Edges.** When comment *c* by user *u* is a reply to another comment (`parent_id` starts with `t1_`) written by a different user *v*, the directed edge *u → v* gets +1 weight:

$$w_{uv} = \#\{\text{comments by } u \text{ that reply directly to } v\}$$

Edges with $w_{uv} < 3$ are dropped, so one-off exchanges don't count as ties. The result has **9,605 nodes and 16,702 edges**.

Each node also stores `comment_count`, `total_score` (sum of upvote scores), `controversial_count` (comments Reddit flagged `controversiality = 1`), and `posts_active` (distinct threads commented in).

### 2. Per-user metrics

**PageRank** (`nx.pagerank`, weight = reply count, damping 0.85). A user ranks highly when many people, especially highly ranked people, reply to them:

$$PR(v) = \frac{1-d}{N} + d \sum_{u \to v} PR(u)\,\frac{w_{uv}}{\sum_{x} w_{ux}}$$

**Betweenness centrality** is the share of shortest paths that pass through a user. Shortest-path algorithms treat weight as a *cost*, while reply counts measure *closeness*, so weights are inverted first:

$$\text{distance}_{uv} = 1 / w_{uv}$$

This makes paths prefer strong ties. It's approximated from 300 sampled source nodes (`k=300, seed=42`) because the exact version is too slow on 9.6k nodes.

**Weighted degree:** `replies_received` = weighted in-degree, `replies_given` = weighted out-degree.

**Influence score** is a hand-weighted blend. The multipliers roughly equalize the scales of the four terms:

$$\text{influence} = 0.35\,(1000\cdot PR) + 0.25\,(\text{replies received}) + 0.20\,(100\cdot\text{betweenness}) + 0.20\,(0.001\cdot\max(\text{total score},0))$$

**Roles** (checked in this order):

| Role | Rule |
|---|---|
| Bridge-Debater | Both conditions below |
| Debater | controversial rate > 1% **and** > 10 controversial comments |
| Bridge | betweenness above the 92nd percentile |
| Advisor | everyone else |

Results are saved to `graph.pkl` (the NetworkX graph) and `metrics.pkl` (a DataFrame with one row per user).

### 3. Communities: Louvain with a resolution parameter

The directed graph is converted to undirected, and communities are found with Louvain (`nx.community.louvain_communities`, `seed=42`), which maximizes **modularity**:

$$Q_\gamma = \frac{1}{2m}\sum_{i,j}\left[A_{ij} - \gamma\,\frac{k_i k_j}{2m}\right]\delta(c_i, c_j)$$

Here $A_{ij}$ is the edge weight, $k_i$ is node *i*'s weighted degree, $m$ is the total edge weight, and $\delta(c_i,c_j)=1$ when *i* and *j* are in the same community.

**γ (resolution)** sets how much a community has to beat random chance to count. Higher γ penalizes large groups more heavily, so communities split into smaller, tighter ones. γ = 1 is the textbook default, but modularity has a known *resolution limit* (Fortunato & Barthélemy, 2007): at γ = 1 it merges groups that really are distinct. So γ is treated as a parameter to choose and report, and the groups are always "communities at resolution γ", never "the" communities.

Only communities with **≥ 30 members** are kept. They are sorted by size (ties broken by the first member's ID) so the ordering is deterministic across runs.

**Choosing γ.** The notebook swept γ = 3 to 10 and checked three things:

1. **Stability.** Louvain is run with 6 different seeds, and the **Adjusted Rand Index** is averaged over every pair of runs:
   $$ARI = \frac{\sum_{ij}\binom{n_{ij}}{2} - \left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}{\tfrac12\left[\sum_i\binom{a_i}{2}+\sum_j\binom{b_j}{2}\right] - \left[\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}\right]/\binom{n}{2}}$$
   1 means identical partitions and 0 means chance-level agreement.
2. **Community count.** The number of groups with ≥ 30 members peaks around γ ≈ 6 to 6.5 (139), then falls.
3. **Topic recovery.** This checks whether the TF-IDF keywords below read as real topics. It's the strongest test, because it uses comment text the clustering never saw.

| γ | Mean ARI | Communities (≥30) | Clusters with a clear topic (of 8) |
|---|---|---|---|
| 1.0 | 0.48 | — | — |
| 3.0 | 0.74 | 104 | 2 |
| 5.0 | 0.81 | 138 | 4 |
| **6.0** | **0.811** | **139** | **5** |
| 8.0 | 0.848 | 128 | 2 |
| 10.0 | 0.860 | 115 | 2 |

ARI keeps rising past γ = 6, but community count and topic recovery both drop. Beyond that point the extra stability comes from locking a shrinking set of groups into place, not from finding more structure. **γ = 6** is the default.

### 4. Choosing which nodes to draw

Drawing all 9,605 nodes would be unreadable, so the picture is a sample:

1. Take the **10 largest communities**.
2. **Peel** each one down to its dense core by repeatedly removing the member with the lowest weighted degree inside the community (ties broken by name) until the cap is reached (100 per community on the site).
3. Draw the subgraph those nodes induce, then drop any node left with no edges.

The sample is picked *by community* rather than by top influence on purpose. High-influence users are the hubs that connect communities, so a top-N-by-influence sample has almost no cluster structure (Q ≈ 0.23 at N = 300).

### 5. Cluster keywords (TF-IDF)

`user_word_counts.pkl` stores each user's word counts from their in-window comments: lowercase tokens matching `[a-z][a-z']{2,}`, minus a small list of English stopwords.

For the keyword legend, each community's **full** membership (not just the drawn sample) is pooled into one "document," which gives 10 documents in total:

$$\text{tf}(w,c) = \frac{\text{count of } w \text{ in } c}{\text{total words in } c} \qquad \text{idf}(w) = \ln\frac{1+N}{1+\text{df}(w)} + 1$$

where *N* is the number of clusters and df(*w*) is how many clusters use *w*. The idf is smoothed (sklearn's convention), so a word used by every cluster is down-weighted instead of dropped. Without smoothing, idf would be 0 for any word all 10 clusters use, which would erase words like "international" that some clusters use far more than others.

Filters that keep junk out of the legend:
- **Subreddit-specific stopwords:** the 180 most common words across all users ("school", "college", "gpa", …) are removed, because everyone on A2C uses them.
- A word must be used by **≥ 4 distinct members** of the cluster, so one prolific poster can't define a cluster's topic.
- A word must appear **≥ 15 times** across all clusters.

Each cluster's top 6 words by tf × idf go in the legend.

### 6. Visual encoding

| Visual | Data |
|---|---|
| Node color | Community (on the site) or role (in the app, if selected) |
| Node size | $10 + 25 \cdot PR(v) / \max PR$ over drawn nodes (PageRank from the full graph) |
| Edge thickness | reply weight $w$ |
| Edge length | $300 - 250\,s$ where $s = \ln(1+w)/\ln(1+w_{\max})$: strong ties pull nodes together |
| Edge opacity | $25 + 128\,s$ (alpha out of 255) |

Weights are very skewed (median 3, max 270), so strength is log-compressed. Without that, most edges would look the same. Edge length is set explicitly because vis-network's physics ignores `value` (thickness). Without explicit lengths, tie strength wouldn't affect the layout at all.

Node positions come from a Barnes-Hut force simulation with an unseeded random start. **Absolute positions mean nothing and change on every reload.** Only the distance between two *connected* nodes carries meaning.

---

## How `docs/index.html` works

The site is fully static. It doesn't compute anything; it switches between 9 precomputed snapshots.

```
docs/
├── index.html          # page shell: slider, stats row, iframe, keyword legend
├── manifest.json       # one entry per γ with its stats and keywords
└── graphs/
    ├── res_1.0.html … res_10.0.html   # pyvis/vis-network graph, one per γ
    └── lib/                            # vis-network + tom-select, loaded by the graph pages
```

**`manifest.json`** has a `resolutions` array. Each entry looks like this:

```json
{
  "gamma": 6.0,
  "file": "graphs/res_6.0.html",
  "communities_found": 139,
  "nodes_drawn": 976,
  "edges_drawn": 1308,
  "intra_pct": 77.6,
  "coverage_pct": 10.16,
  "keywords": { "0": ["word", "…"], "1": ["…"] }
}
```

- `communities_found`: communities with ≥ 30 members at this γ (all of them, not just the 10 drawn)
- `intra_pct`: share of drawn edges whose endpoints are in the same community
- `coverage_pct`: `nodes_drawn` / 9,605

**What the page does:**

1. `boot()` fetches `manifest.json` and sets the slider's max to `resolutions.length - 1`. The slider moves through **positions in the array**, not γ values, so the gap between 8 and 10 is one step like any other. The starting position is 5, which is γ = 6.
2. On every `input` event, `render(idx)`:
   - shows `gamma` next to the slider,
   - points the `<iframe>` at that entry's `file`, which loads that snapshot's graph,
   - fills in the five stats,
   - builds the keyword legend from `keywords`. Dot colors come from the `COMM_COLORS` array, which must stay in the same order as the palette used to draw the graphs, or the legend colors won't match the nodes.

Each graph runs its own physics simulation inside the iframe, so the layout settles again every time you move the slider.

**Viewing locally:** browsers block `fetch()` on `file://` pages, so opening `index.html` directly shows a "could not load manifest.json" error. Serve the folder over HTTP instead:

```bash
cd docs && python3 -m http.server
# open http://localhost:8000
```

**Publishing:** in the repo's Settings → Pages, choose "Deploy from a branch", then `main` and `/docs`.

**Adding or changing a snapshot:** create the graph HTML with the pipeline from sections 3 to 6 at the new γ (100 members per community, 10 communities), put it in `docs/graphs/`, and add a matching entry to `manifest.json`. `index.html` doesn't need any changes. The script that produced the current snapshots isn't in this repo; `app.py`'s `build_subgraph`, `cluster_keywords` and `render_html` have the same logic.

---

## Running the live app

```bash
pip install -r requirements.txt
streamlit run app.py
```

`app.py` loads `graph.pkl`, `metrics.pkl` and `user_word_counts.pkl`, and lets you change γ (0.5 to 10), the number of communities drawn, members per community, color mode (community or role), and the TF-IDF thresholds. It also has an on-demand 6-seed ARI stability check. Results are cached per parameter set, so returning to a setting you've already viewed is instant.

## Reproducing from raw data

Put `r_ApplyingToCollege_comments.jsonl` in the repo root and run `main.ipynb`. Things to watch for:

- The notebook's word-count cell hashes usernames **twice** (`anonymize(anonymize(u))`), because the kernel that created the current `graph.pkl` ran the anonymize cell twice. If you rebuild `graph.pkl` with a clean top-to-bottom run, remove the second call, or no comment text will match a graph node.
- Cells 4 and 5 use a `posts_df` that is never loaded. They aren't needed for the graph and can be skipped.
- `user_word_counts.pkl` is built once from the raw file. After that, re-runs at a different γ reuse it.
