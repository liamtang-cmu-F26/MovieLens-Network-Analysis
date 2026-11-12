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

