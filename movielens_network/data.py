"""MovieLens 1M data loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]
MOVIE_COLUMNS = ["movie_id", "title", "genres"]


@dataclass(frozen=True)
class MovieLensData:
    ratings: pd.DataFrame
    movies: pd.DataFrame


def load_movielens(data_dir: Path) -> MovieLensData:
    """Load and validate ratings and movie metadata from MovieLens 1M."""

    ratings_path = data_dir / "ratings.dat"
    movies_path = data_dir / "movies.dat"
    missing = [str(path) for path in (ratings_path, movies_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required MovieLens file(s): " + ", ".join(missing))

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=RATING_COLUMNS,
        dtype={
            "user_id": "int32",
            "movie_id": "int32",
            "rating": "int8",
            "timestamp": "int64",
        },
        encoding="latin-1",
    )
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=MOVIE_COLUMNS,
        dtype={"movie_id": "int32", "title": "string", "genres": "string"},
        encoding="latin-1",
    )

    _validate(ratings, movies)
    return MovieLensData(ratings=ratings, movies=movies)


def _validate(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    if ratings.empty or movies.empty:
        raise ValueError("The ratings and movies files must not be empty.")
    if ratings[["user_id", "movie_id"]].duplicated().any():
        raise ValueError("Duplicate user-movie rating pairs were found.")
    if not ratings["rating"].between(1, 5).all():
        raise ValueError("Ratings outside the expected 1–5 range were found.")
    if movies["movie_id"].duplicated().any():
        raise ValueError("Duplicate movie IDs were found in movies.dat.")

    unknown_movies = set(ratings["movie_id"].unique()) - set(movies["movie_id"].unique())
    if unknown_movies:
        raise ValueError(
            f"Ratings reference {len(unknown_movies)} movie IDs missing from movies.dat."
        )

