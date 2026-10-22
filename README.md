# MovieLens Network Analysis

Python analysis of the MovieLens 1M dataset as a weighted user–movie bipartite network.

## Structure

```text
main.py                    # command-line entry point
movielens_network/
├── data.py                # dataset loading and validation
├── analysis.py            # network summary statistics
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

