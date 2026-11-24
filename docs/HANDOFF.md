# Milestone 2 experiment handoff


## Karthik's complete workflow

Use the README's evaluation protocol and this document together. The workflow
below covers weighted diffusion, node2vec, the controlled exclusion experiment,
and final test evaluation. These are implementation tasks following the completed
community analysis.

1. Reproduce the existing audit, validation baseline, and analysis with
   `run_milestone2.py audit`, `evaluate`, and `report`.
2. Add the two weighted diffusion variants defined below.
3. Add node2vec using the reference movie graph and instructions below.
4. Evaluate all six methods at 1, 5, 10, and full history under both exclusion
   conditions. The methods are popularity, item cosine, binary diffusion,
   weighted-seed diffusion, weighted-seed-and-transition diffusion, and node2vec.
5. Choose settings on validation and record method definitions, seeds,
   graph settings, and the primary metric before final test scoring.
6. Rebuild the reference graph and refit methods using data before the test
   cutoff. Evaluate the frozen configurations and export the results described
   under "Final test evaluation and deliverables" below.

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


## Weighted diffusion experiments

Both variants use the original reference user-movie ratings, with the same
reference/target split and cutoff as binary diffusion. The pruned movie graph
export is for community analysis and node2vec.

Assign rating weight 1 to four stars, 2 to five stars, and 0 to other ratings.
Use those weights only on the visible target history when forming weighted
seeds. Low ratings still count toward history length and the seen mask.

| Method | Target seeds | Reference transitions |
|---|---|---|
| Binary diffusion baseline | 1 per visible usable liked movie | Binary positive edges |
| Weighted-seed diffusion | Four stars = 1, five stars = 2 | Binary positive edges |
| Weighted-seed-and-transition diffusion | Four stars = 1, five stars = 2 | Four-star reference edges = 1, five-star reference edges = 2 |

For the first variant, reuse the binary diffusion transitions and change only
initial seed resource. For the second, divide each movie's outgoing resource
by its reference weight sum, then divide each intermediate user's resource
by their outgoing reference weight sum. This preserves resource during each
transition from a supported seed.

With reference users as rows and movies as columns, B is the binary reference
matrix, W is its rating-weighted counterpart, and z is a row vector of weighted
target seeds. The respective score rows are:

```text
weighted seeds only       z D_movie_degree^-1 B.T D_user_degree^-1 B
weighted seeds/transitions z D_movie_strength^-1 W.T D_user_strength^-1 W
```

Degrees count positive reference edges. Strengths sum reference edge weights.
Set inverse zero degrees/strengths to zero. Apply the existing usable-seed,
mask, ranking, tie-breaking, and popularity fallback rules. Keep the original
popularity and cosine baselines fixed.

Compare weighted seeds against binary diffusion, then weighted transitions
against weighted seeds. This separates the effect of target rating strength
from weighting the reference network. Run the paired precision/recall analysis,
fallback counts, popularity, and coverage for each variant.

## Controlled exclusion experiment

Run both conditions on the same snapshot users, later positive labels,
reference training graph, candidate catalog, and scoring histories.

| Condition ID | Ratings used for scoring | Movies excluded |
|---|---|---|
| `visible_only` | First 1, 5, 10, or all pre-cutoff ratings | Every movie in that visible history |
| `full_pre_cutoff` | First 1, 5, 10, or all pre-cutoff ratings | Every movie the target user rated before the cutoff, including hidden low ratings |

The full exclusion list is used only for masking. Hidden earlier ratings
remain excluded from score construction and reference training. The controlled
condition gives the evaluator additional seen-movie information, which should
be disclosed when interpreting it.

Reuse fitted models and node2vec embeddings between conditions. Apply each
condition's mask to the popularity fallback as well. Keep held-out relevant-item
counts fixed. At full history, rankings, metrics, and fallback reasons must match
between conditions for each method and embedding seed.

Export condition IDs explicitly and compare the conditions on paired users.
This experiment helps separate changes caused by candidate exclusions from
changes caused by additional scoring history. Restricting scoring to the first
ratings still omits newer preference evidence.

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


## Final test evaluation and deliverables

Record validation choices before inspecting final test accuracy. Preserve the
existing validation exports as the baseline reproduction reference and save
expanded experiments in separate files or directories with their configuration.
The current evaluator and report command need extending for the new methods,
conditions, embedding seeds, and test snapshot.

The graph rebuild above uses only earlier reference ratings. Final test scoring
uses the existing test cohort and later eligible likes from the data pipeline.
The audit currently reports 321 users and 13,590 eligible likes. Refit learned
methods on the appropriate test reference data using the chosen settings.
Run the agreed six methods, four histories, and two exclusion conditions.

Push the following deliverables:

- Updated evaluation and reporting code with reproducible validation/test commands.
- A configuration recording cutoffs, model and graph settings, library versions,
  seeds, graph hashes, and the primary Precision@10 comparison.
- Aggregate validation and test metrics by method, history, exclusion condition,
  and embedding seed, with user counts, macro precision/recall, and fallback use.
- Per-user results with snapshot, user ID, method, history length, visible rating
  count, exclusion condition, embedding seed (blank for deterministic methods),
  top-10 IDs, hits, eligible relevant-item count, fallback reason, precision,
  and recall.
- Paired method differences and intervals, seed-availability/fallback summaries,
  and recommendation popularity/coverage tables and figures for the final test.

Report variation across node2vec training seeds separately from user-bootstrap
uncertainty. For a single node2vec paired comparison, average each user's metric
over the predeclared embedding seeds before resampling users. Pooling seed runs
as additional independent users would overstate the sample size.

Check cohort and relevance counts, exclusion compliance, recomputed hit metrics,
and fallback rankings. Verify full-history equality between exclusion conditions.
Use the resulting outputs to document both improvements and null/negative
findings under the fixed evaluation protocol.

## Method sources

The implementation uses [NetworkX weighted Louvain](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html).
The embedding handoff follows the graph-learning framework described by
[Grover and Leskovec's node2vec project](https://snap.stanford.edu/node2vec/).
