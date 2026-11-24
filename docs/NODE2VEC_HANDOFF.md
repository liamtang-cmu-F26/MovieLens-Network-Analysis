# Reference movie graph and node2vec handoff

## Reproduce the community analysis

Install the updated requirements, including NetworkX, and run from the repo root:

```bash
python3 run_milestone2.py communities --snapshot validation --data-dir ml-1m
```

Files are written under `outputs/milestone2/communities/validation/`.
An optional `--output-dir` changes the parent directory. The snapshot name is
always appended, so validation and test graph exports have separate destinations.

| File | Purpose |
|---|---|
| `movie_graph_nodes.csv` | Every candidate movie, original MovieLens ID, title, genres, positive reference degree, graph degree/strength, community assignment, and edge availability. |
| `movie_graph_edges.csv` | Each undirected edge once, with `source`, `target`, cosine `weight`, and `shared_positive_users`. |
| `community_summary.csv` | All communities, sizes, reference popularity, internal strength share, top genres, and representative movies. |
| `community_genres.csv` | Multi-label genre counts and fractions within every community. |
| `community_stability.csv` | Louvain community counts, modularity, and label-independent agreement across three seeds. |
| `figure_movie_communities.png` | Community sizes and genre composition for up to the 12 largest communities. |
| `graph_config.json` | Cutoff, thresholds, software versions, source-reference fingerprint, graph-file hashes, and graph statistics. |

## Construction rules

The existing deterministic 70/30 user split with seed 755 is reused.
Only reference users' ratings strictly before the selected global cutoff
contribute to the graph. Four- and five-star ratings are binary positive edges.
Target users' ratings and evaluation labels do not enter graph construction.
Movie titles and genres enter interpretation after graph construction.

For movies i and j, let c(i,j) be the number of reference users who liked both,
and d(i), d(j) their positive reference degrees. The edge similarity is
c(i,j) / sqrt(d(i) * d(j)). Self-loops are removed.

Each movie selects its 20 strongest neighbours with at least three shared
positive users. Ties use ascending movie ID. Keep the union of these selections
as an undirected graph. Final node degrees can exceed 20. These fixed thresholds
keep the projection manageable and filter edges supported by one or two users.
They were chosen as starting graph settings, without optimizing recommendation
accuracy. This pruning changes the graph used by node2vec relative to the full
cosine recommender, which should be disclosed in the report.

Weighted Louvain uses resolution 1, threshold 1e-7, and seeds 755, 756, 757.
The reported partition always uses 755, rather than selecting the seed with the
largest modularity. IDs C0, C1, ... follow decreasing community size, with
minimum movie ID breaking ties. Labels are local to a snapshot and seed.

All candidate movies are retained in the node table. Isolates have
`community_id=-1`, graph degree zero, and no informative walk context.
They are excluded from community detection and partition agreement.
`internal_strength_share` is twice the community's internal edge weight
divided by its total node strength. Representative titles are the three movies
with greatest weighted graph strength, with ascending movie-ID ties.

## Validation findings

The graph has 3,588 candidate movies, 49,129 undirected edges, and one connected
component of 3,072 movies. The remaining 516 movies are isolates. Seed 755
finds eight communities with weighted modularity 0.568.

The groups include a 299-movie community with 74% horror, a 286-movie community
with 64% children's movies, and a 257-movie community with 53% action and 51%
science fiction. Genre labels can overlap and these are within-group prevalence
figures. Broad genres such as drama appear across multiple communities.

All three seeds find eight communities with modularity 0.568-0.570.
Adjusted Rand agreement with seed 755 is 0.565 for seed 756 and 0.638 for seed
757. This indicates partial agreement with meaningful boundary variation.
The graph suggests interpretable audience groups, while exact memberships
depend on the optimization seed. The seed comparison assesses algorithm
variation on one graph. A degree-preserving null-model significance test and
graph-threshold sensitivity analysis have not been performed.

## Karthik's node2vec input

Load original movie IDs consistently as strings, and use an undirected weighted
graph. The explicit node list retains movies absent from the edge list.

```python
from pathlib import Path

import networkx as nx
import pandas as pd

root = Path("outputs/milestone2/communities/validation")
nodes = pd.read_csv(root / "movie_graph_nodes.csv")
edges = pd.read_csv(root / "movie_graph_edges.csv")

graph = nx.Graph()
graph.add_nodes_from(nodes["movie_id"].astype(str))
graph.add_weighted_edges_from(
    (str(int(row.source)), str(int(row.target)), row.weight)
    for row in edges.itertuples(index=False)
)
walk_graph = graph.subgraph(
    [node for node, degree in graph.degree() if degree > 0]
).copy()
```

Preserve the cosine weights in random-walk transition probabilities. Use the
same graph for all 1, 5, 10, and full-history conditions. Community labels,
genres, target histories, and evaluation labels are excluded from embedding
training. Fit on `walk_graph` and record which IDs receive usable finite vectors.
A node2vec implementation may otherwise return arbitrary isolated-node vectors.

For a simple recommendation baseline, normalize movie vectors to unit length,
average vectors of the target user's visible liked movies with usable vectors,
then use cosine similarity to rank movies. If the average has zero norm or no
usable vectors, use the shared popularity fallback. If there is no positively
scored unseen movie, apply the existing no-positive-candidate fallback.

Keep the common candidate catalog. Unembedded movies have score negative
infinity for learned ranking and remain eligible for the shared popularity
fallback. Record this method's embedding availability, fallback reasons, and
candidate support explicitly, since pruning excludes some movies that the
binary methods can use. Extend the export schema and validator to recognize any
new embedding-specific fallback reasons.

Use a small validation grid for node2vec settings, including a p=q=1 reference,
and at least three embedding seeds for each reported setting. Record dimensions,
walk length/count, context window, p/q, epochs, weighting behavior, training
workers, seeds, graph hash, and implementation version. Use a single training
worker when the chosen implementation needs it for reproducibility.

Retain the existing history, seen-mask, candidate and metric rules. For the
controlled exclusion experiment, reuse embeddings while changing only the mask.
Export per-user movie IDs, hits, relevant counts, precision, recall, and fallback
reasons. Extend the analysis command for the new method, embedding seeds, and
final test output.

## Final test graph

After choosing settings on validation, rebuild using the later reference cutoff:

```bash
python3 run_milestone2.py communities --snapshot test --data-dir ml-1m
```

This uses reference ratings strictly before December 29, 2000 at 23:43:34 UTC.
The validation graph uses the earlier December 2 cutoff. The test command
exports a graph and descriptive communities, without scoring recommendations.
Refit embeddings on the test reference graph using the chosen settings and
seeds. Community IDs across snapshots have no automatic correspondence.

## Method sources

The implementation uses [NetworkX weighted Louvain](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html).
The embedding handoff follows the graph-learning framework described by
[Grover and Leskovec's node2vec project](https://snap.stanford.edu/node2vec/).
