# MovieLens Network Analysis

Python analysis of the MovieLens 1M dataset as a weighted user–movie bipartite network.

## Structure

```text
main.py                    # command-line entry point
movielens_network/
├── data.py                # dataset loading and validation
├── analysis.py            # network summary statistics
├── milestone2_data.py    # global-time snapshots and target cohorts
├── milestone2_evaluation.py # validation recommenders and metrics
├── plots.py               # Figures 1 and 2
├── reporting.py           # Table 1 export
└── utils.py               # shared helpers
outputs/                   # generated tables and figures
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

By default, the script reads `ml-1m/` and writes to `outputs/`. Use `python main.py --help` for other paths.

The dataset is ignored by Git and should not be redistributed. Please cite Harper and Konstan (2015), *The MovieLens Datasets: History and Context*.

## Milestone 2 validation

The audit command writes cohort counts for both the validation and test windows:

```bash
python run_milestone2.py audit --data-dir ml-1m
```

Run the baseline recommenders on validation only:

```bash
python run_milestone2.py evaluate --data-dir ml-1m
```

The validation evaluator compares positive-edge popularity, item cosine, and
binary three-step diffusion for 1, 5, 10, and full pre-cutoff histories. It
uses the same reference-rated catalog for every method and masks every movie
visible in a target user's selected history, including ratings below four
stars. A personalized method falls back to the same popularity ranking when it
has no usable positive seed or no positive-scoring unseen candidate. Ties are
broken by ascending movie ID. Precision@10 uses 10 as its denominator and
Recall@10 uses each user's eligible held-out positive count; metrics are macro
averages across eligible users.

The command writes `outputs/milestone2/validation_metrics.csv` and
`outputs/milestone2/validation_per_user.csv`. Per-user rows include the requested
history length, number of visible ratings, JSON-encoded top-10 movie IDs, hit
count, eligible relevant-item count, and fallback reason. This command does not
run the test window; use it to tune and lock settings before a separate final
test evaluation.


## Milestone 2 analysis

Generate tables and figures from the saved validation exports:

```bash
python run_milestone2.py report --data-dir ml-1m
```

The report command reconstructs the validation snapshot and checks the exported
cohort, recommendation masks, hits, fallback rankings, and summary metrics before
writing to `outputs/milestone2/analysis/`. Use `--results-dir` to select another
directory containing the validation exports, and `--output-dir` for a different
destination.

- `figure_1_history_length.png` shows macro Precision@10 and Recall@10.
- `figure_2_paired_differences.png` compares every method pair on the same users.
- `figure_3_fallback_subgroups.png` compares accuracy within common seed-availability
  groups and shows each method's recorded fallback rate.
- `figure_4_recommendation_popularity.png` shows mean recommendation degree,
  catalog coverage, and recommendations rated earlier but hidden by truncation.

Each figure has a supporting CSV. Differences are in percentage points, with
95% percentile intervals from 10,000 paired user bootstrap resamples using seed
755. Intervals describe uncertainty conditional on this validation snapshot.
The comparisons are exploratory and have no multiple-comparison adjustment.

A usable seed is a visible rating of four or five stars whose movie has a
positive reference edge. Every method is evaluated on the same users within
each seed group. Group membership changes across history lengths. Empty groups
have zero users and blank accuracy values.

Recommendation popularity counts positive ratings from reference users strictly
before the cutoff. Mean degree is averaged over recommendation slots, counting
repeated exposure to a movie across users. Catalog coverage divides the number
of distinct recommended movies by the full reference-rated candidate catalog.
The hidden-rating diagnostic counts recommendations already rated before the
cutoff but hidden in the selected history. Full history changes both scoring
information and the seen-movie exclusions.

`analysis_config.json` records source export hashes and analysis settings.

## Milestone 2 evaluation protocol

This protocol describes the current binary baseline and the validation handoff.
The implementation is in `milestone2_data.py` and `milestone2_evaluation.py`.

| Rule | Specification |
|---|---|
| Dataset | MovieLens 1M. Each user-movie pair is unique. Ratings are whole stars from 1 to 5. |
| User assignment | Sort all dataset user IDs, permute with NumPy `default_rng(755)`, assign the first `floor(0.70 * users)` to reference and the remainder to target. This assigns 4,228 reference and 1,812 target users and is reused across snapshots. Only users with earlier ratings contribute to a snapshot. |
| Validation cutoff | December 2, 2000 at 14:52:18 UTC. Reference and target histories use timestamps strictly before this cutoff. |
| Validation labels | Target ratings at or after the validation cutoff and strictly before December 29, 2000 at 23:43:34 UTC. |
| Test cutoff | December 29, 2000 at 23:43:34 UTC. Test histories use earlier ratings. Test labels use timestamps at or after this cutoff through the dataset end. |
| Boundary ties | All ratings with a timestamp equal to the cutoff enter the later window. |
| Reference graph | Reference users' ratings strictly before the snapshot cutoff. Target users contribute no training edges, including target users outside the eligible evaluation cohort. |
| Candidate catalog | Movies with any earlier reference rating, including movies without positive reference ratings. No sampled negatives. |
| Eligible target cohort | At least 10 earlier ratings and at least one four- or five-star rating in the evaluation window whose movie is in the candidate catalog. The cohort and eligible labels are fixed across history lengths within a snapshot. |
| History conditions | First 1, 5, 10, or all earlier ratings, sorted by timestamp then ascending movie ID. A timestamp tie does not reveal the actual within-timestamp rating order. Low ratings count toward history length. |
| Hidden earlier history | Unrevealed earlier target ratings contribute no scores, seen masks, or relevance labels. Only later eligible four- or five-star ratings are relevance labels. |
| Seen mask | Exclude every visible rated movie, including ratings below four stars. |
| Positive edges | Four- and five-star reference ratings have weight 1. Other ratings contribute no positive edge. |
| Usable seed | A visible four- or five-star rating whose movie has at least one positive reference edge. Each usable movie contributes initial weight 1. |
| Popularity | Positive reference degree, with the target user's visible seen mask applied. |
| Item cosine | Cosine similarity between positive reference movie columns, summed across the target user's usable seed movies. |
| Binary diffusion | Each seed movie divides its resource among positive reference users, who divide that resource among their positive movies. Normalize by positive movie and user degrees. |
| Fallback | For a personalized scorer, use popularity if there are no usable seeds (`no_usable_seed`). Otherwise use popularity if no unseen candidate has a positive score (`no_positive_unseen_candidate`). Popularity rows have a blank fallback reason. |
| Ranking | Descending score, then ascending movie ID for ties. Return up to 10 movies. When 1-9 unseen movies have positive personalized scores, fill remaining positions with zero-score candidates in ascending ID order. |
| Metrics | Per-user Precision@10 = hits / 10. Recall@10 = hits / eligible later positives. Average each metric over users with equal user weight. The current validation export has exactly 10 recommendations per row. |
| Input/output boundary | Scorers use the reference graph, candidate IDs, and visible histories. Held-out positives enter hit calculations after ranking. |
| Model selection | Use validation for method choices. Record the selected settings before final test scoring. The current command scores validation. The audit reports test cohort counts. |

MovieLens ratings record when ratings were submitted, which can differ from
when a movie was watched. Missing ratings have unknown relevance. These metrics
measure recovery of recorded later likes. The target cohorts simulate limited
histories among users who already have at least ten earlier ratings.

For reproducibility, run from the repository root after installing
`requirements.txt` and obtaining the local `ml-1m/` data directory:

```bash
python3 run_milestone2.py audit --data-dir ml-1m
python3 run_milestone2.py evaluate --data-dir ml-1m
python3 run_milestone2.py report --data-dir ml-1m
```

Expected validation counts are 142 users, 3,063 eligible later likes, and 3,588
candidate movies. Personalized fallback counts by history are 63, 4, 0, and 0.
The test audit has 321 eligible users and 13,590 eligible later likes.
The baseline export has 1,704 rows, covering 142 users, four histories, and three
methods. Per-user fields include history length, visible rating count, top-10
movie IDs, hits, relevant-item count, fallback reason, precision, and recall.

## Review of the four validation analyses

The figures and CSVs below are generated by the report command. Their source
export fingerprints and bootstrap settings are recorded in
[analysis_config.json](outputs/milestone2/analysis/analysis_config.json).

### 1. History length

[Figure](outputs/milestone2/analysis/figure_1_history_length.png) and
[table](outputs/milestone2/analysis/validation_history_metrics.csv).

| Visible ratings | Popularity Precision@10 | Item cosine Precision@10 | Diffusion Precision@10 |
|---|---:|---:|---:|
| 1 | 4.86% | 5.14% | 5.07% |
| 5 | 5.07% | 5.00% | 4.15% |
| 10 | 5.21% | 4.86% | 4.15% |
| Full | 9.30% | 11.41% | 10.07% |

The one-rating gains are small. Popularity leads at five and ten ratings.
Item cosine has the highest full-history precision. Full-history recall is
5.59% for cosine and 5.61% for diffusion, a small difference with substantial
uncertainty.

Full history changes both preference evidence and seen-movie exclusions.
Early histories also omit more recent preference evidence. These curves
describe the combined effect of revealing earlier ratings and cannot isolate
history size alone.

### 2. Paired method differences

[Figure](outputs/milestone2/analysis/figure_2_paired_differences.png) and
[table](outputs/milestone2/analysis/validation_paired_differences.csv).

Each difference is the first method minus the second on the same users.
Positive values favor the first method. User wins, ties, and losses are exported
alongside the mean and bootstrap interval.

| Precision comparison | History | Difference in percentage points | 95% bootstrap interval |
|---|---|---:|---|
| Diffusion minus popularity | 1 | +0.21 | -0.63 to +0.99 |
| Diffusion minus popularity | 5 | -0.92 | -1.83 to -0.07 |
| Diffusion minus popularity | 10 | -1.06 | -1.90 to -0.21 |
| Item cosine minus popularity | Full | +2.11 | +0.63 to +3.59 |
| Diffusion minus popularity | Full | +0.77 | +0.14 to +1.41 |
| Diffusion minus item cosine | Full | -1.34 | -2.75 to +0.07 |

The current evidence does not establish an advantage for diffusion with limited
history. Full-history cosine and diffusion improve precision over popularity
in this snapshot. Their relative precision remains uncertain under the paired
interval. These are exploratory, unadjusted comparisons on one user cohort and
time window. They do not establish superiority across datasets or periods.

### 3. Seed availability and fallback

[Figure](outputs/milestone2/analysis/figure_3_fallback_subgroups.png) and
[table](outputs/milestone2/analysis/validation_fallback_subgroups.csv).

With one visible rating, 79 users have usable seeds and 63 users (44.4%) do not.
Among the 79 seed-available users, Precision@10 is 5.19% for popularity, 5.70%
for cosine, and 5.57% for diffusion. Within the 63 users without seeds, all three
methods have the same recommendations and 4.44% precision because the
personalized methods use the shared popularity fallback.

With five visible ratings, only four users lack usable seeds. That subgroup's
metrics are descriptive and based on very few users. At ten and full history,
the no-seed subgroup is empty. Blank subgroup metrics mean no observations.
Subgroup membership changes across history lengths, so comparisons within a
history are paired, while comparisons across subgroup curves also reflect
membership changes. Fallback counts are taken from the saved model export and
reported separately from independently computed seed availability.

### 4. Recommendation popularity and coverage

[Figure](outputs/milestone2/analysis/figure_4_recommendation_popularity.png) and
[table](outputs/milestone2/analysis/validation_recommendation_popularity.csv).

At ten visible ratings, cosine recommends 128 distinct movies (3.57% of the
catalog), diffusion recommends 50 (1.39%), and popularity recommends 17 (0.47%).
Mean positive reference degrees of their recommendations are approximately
1,117, 1,281, and 1,342, respectively. Across all four history conditions,
diffusion recommends more popular movies and reaches fewer distinct movies
than cosine. Catalog coverage describes aggregate exposure across users and
does not measure within-list genre diversity or prove that users prefer those
lists. These summaries include fallback recommendations.

At ten visible ratings, 822 of popularity's 1,420 recommendation slots (57.9%),
857 of cosine's (60.4%), and 888 of diffusion's (62.5%) are movies the users
rated earlier but whose ratings were hidden. They cannot match the later-rating
labels in this dataset with unique user-movie pairs. Full history removes
these movies from the lists. This diagnostic helps explain the full-history
increase, but cannot quantify how much of that increase comes from exclusions
versus additional preference evidence.

## Handoff after the validation review

The validation baseline is ready for Karthik to reproduce using the three
commands above. Next, compare binary diffusion with weighted seeds, then with
both weighted seeds and weighted reference transitions. The proposed rating
weights are 1 for four stars and 2 for five stars, with transitions normalized
by outgoing weight sums. Keep user cohorts, history definitions, candidates,
masks, fallback rules, and metrics fixed for those comparisons.

Record method definitions and any selected settings using validation before
running final test scoring. Weighted variants, test scoring, and node2vec remain future work.
The community analysis and graph export are available below.


## Movie communities and node2vec graph export

Build the validation reference movie graph and community outputs:

```bash
python3 run_milestone2.py communities --snapshot validation --data-dir ml-1m
```

The command uses reference users' earlier positive ratings to construct a
cosine movie graph with supported neighbours, then runs weighted Louvain
with three seeds. It exports all candidate movie IDs, weighted edges,
community assignments, genre composition, partition stability, and a figure
under `outputs/milestone2/communities/validation/`.

See the [node2vec handoff](docs/NODE2VEC_HANDOFF.md) for the graph rules,
validation findings, file schemas, loading example, isolate handling, and
instructions for rebuilding the final test graph. Graph thresholds are fixed
starting choices. Genre descriptions and modularity are exploratory.

[Community figure](outputs/milestone2/communities/validation/figure_movie_communities.png)
and [summary table](outputs/milestone2/communities/validation/community_summary.csv).
