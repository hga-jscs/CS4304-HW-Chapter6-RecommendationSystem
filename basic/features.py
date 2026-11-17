"""Training-only category/frequency fitting with compact metadata lookup.

Neither fitting nor transformation reads target labels. Song/member tables are
looked up by unique identifiers, avoiding many-to-many joins and row expansion.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


INTERACTION_COLUMNS = [
    "msno", "song_id", "source_system_tab", "source_screen_name", "source_type"
]
CONTEXT_COLUMNS = ["source_type", "source_screen_name", "source_system_tab"]
STRING_DTYPE = "string[pyarrow]"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_interactions(
    path: Path, *, training: bool, chunksize: int = 200_000
) -> Iterator[pd.DataFrame]:
    columns = INTERACTION_COLUMNS + (["target"] if training else ["id"])
    dtype = {name: STRING_DTYPE for name in INTERACTION_COLUMNS}
    # Parse the label before narrowing its type so invalid values cannot wrap.
    dtype["target" if training else "id"] = "float64" if training else "int64"
    for frame in pd.read_csv(path, usecols=columns, dtype=dtype, chunksize=chunksize):
        if frame[["msno", "song_id"]].isna().any().any():
            raise ValueError(f"Missing user/song identifier in {path.name}")
        if training:
            if not frame["target"].isin([0, 1]).all():
                raise ValueError("Training target must contain only 0 and 1")
            frame["target"] = frame["target"].astype("int8")
        yield frame


def load_training(
    path: Path, *, max_rows: int = 0, seed: int = 42, chunksize: int = 200_000
) -> tuple[pd.DataFrame, dict]:
    """Take an exact uniform sample over the whole file, or use every row.

    Sampling is two-pass: count CSV records, choose row numbers, then read only
    selected records into the retained frame. A prefix is never used as a
    substitute for a random or chronological split.
    """
    if max_rows < 0 or chunksize <= 0:
        raise ValueError("max_rows must be nonnegative and chunksize positive")
    selected = None
    total_rows = None
    if max_rows:
        total_rows = sum(len(chunk) for chunk in pd.read_csv(
            path, usecols=["target"], chunksize=chunksize
        ))
        if max_rows < total_rows:
            selected = np.sort(np.random.default_rng(seed).choice(
                total_rows, size=max_rows, replace=False
            ))
    retained = []
    offset = 0
    for frame in read_interactions(path, training=True, chunksize=chunksize):
        stop = offset + len(frame)
        if selected is None:
            positions = np.arange(offset, stop, dtype=np.int64)
        else:
            lower, upper = np.searchsorted(selected, [offset, stop])
            positions = selected[lower:upper]
            frame = frame.iloc[positions - offset].copy()
        if len(frame):
            frame["row_id"] = positions
            retained.append(frame)
        offset = stop
    if not retained:
        raise ValueError("No training records were found")
    result = pd.concat(retained, ignore_index=True)
    return result, {
        "available_rows": offset,
        "selected_rows": len(result),
        "sampling": "uniform_without_replacement" if selected is not None else "all_rows",
        "sampling_seed": seed,
    }


def _metadata(path: Path, key: str, columns: list[str]) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=[key, *columns], dtype=STRING_DTYPE)
    if table[key].isna().any() or table[key].duplicated().any():
        raise ValueError(f"{path.name}: {key} must be present and unique")
    return table.set_index(key)


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float32").replace(
        [np.inf, -np.inf], np.nan
    )


class FeatureBuilder:
    """Serializable feature state fitted on the current training partition."""

    def __init__(self, data_dir: Path, mode: str = "baseline", *, disable_frequency_features: bool = False,
                 user_context_features: bool = False):
        if mode not in {"baseline", "enhanced"}:
            raise ValueError("feature mode must be baseline or enhanced")
        self.mode = mode
        self.disable_frequency_features = disable_frequency_features
        self.user_context_features = user_context_features
        self.members = _metadata(Path(data_dir) / "members.csv", "msno", [
            "city", "bd", "gender", "registered_via", "registration_init_time",
            "expiration_date",
        ])
        self.songs = _metadata(Path(data_dir) / "songs.csv", "song_id", [
            "song_length", "genre_ids", "artist_name", "composer", "lyricist", "language",
        ])
        self.extra = None
        if mode == "enhanced":
            self.extra = _metadata(Path(data_dir) / "song_extra_info.csv", "song_id", ["name", "isrc"])
        self.categories: dict[str, pd.Index] = {}
        self.frequencies: dict[str, pd.Series] = {}
        self.categorical_columns: list[str] = []
        self.feature_names: list[str] = []

    def raw_features(self, frame: pd.DataFrame):
        for name in INTERACTION_COLUMNS:
            yield name, frame[name], True
        for name in ["city", "gender", "registered_via"]:
            yield name, frame["msno"].map(self.members[name]), True
        age = _numeric(frame["msno"].map(self.members["bd"]))
        yield "age", age.where(age.between(1, 100)), False
        for name in ["registration_init_time", "expiration_date"]:
            yield name, frame["msno"].map(self.members[name]), False
        for name in ["genre_ids", "artist_name", "composer", "lyricist", "language"]:
            yield name, frame["song_id"].map(self.songs[name]), True
        length = _numeric(frame["song_id"].map(self.songs["song_length"]))
        yield "song_length", length.where(length > 0), False
        if self.mode == "enhanced":
            for name in ["genre_ids", "artist_name", "composer", "lyricist"]:
                values = frame["song_id"].map(self.songs[name])
                count = values.str.count(r"\|") + 1
                yield name + "_count", count.where(values.notna()), False
            registered = pd.to_datetime(
                frame["msno"].map(self.members["registration_init_time"]),
                format="%Y%m%d", errors="coerce",
            )
            expires = pd.to_datetime(
                frame["msno"].map(self.members["expiration_date"]),
                format="%Y%m%d", errors="coerce",
            )
            yield "registration_year", registered.dt.year, False
            yield "registration_month", registered.dt.month, False
            yield "membership_days", (expires - registered).dt.days, False
            assert self.extra is not None
            isrc = frame["song_id"].map(self.extra["isrc"]).str.upper()
            yield "isrc_country", isrc.str[:2], True
            yield "isrc_year_code", isrc.str[5:7], False
            yield "song_name_length", frame["song_id"].map(self.extra["name"]).str.len(), False

    def fit(self, frame: pd.DataFrame) -> "FeatureBuilder":
        self.categories = {}
        self.frequencies = {}
        self.categorical_columns = []
        self.feature_names = []
        counted = {"msno", "song_id", "artist_name", "genre_ids", "source_type"}
        for name, values, categorical in self.raw_features(frame):
            self.feature_names.append(name)
            if categorical:
                self.categories[name] = pd.Index(values.dropna().unique())
                self.categorical_columns.append(name)
            if self.mode == "enhanced" and not self.disable_frequency_features and name in counted:
                counts = values.value_counts(dropna=True, sort=False).astype("float32")
                self.frequencies[name] = counts
                self.feature_names.append(name + "_frequency")
        if getattr(self, "user_context_features", False):
            user_codes = self.categories["msno"].get_indexer(frame["msno"])
            self.context_user_counts = np.bincount(user_codes, minlength=len(self.categories["msno"]))
            self.context_categories = {}
            self.context_probabilities = {}
            self.context_pair_counts = {}
            for name in CONTEXT_COLUMNS:
                # Preserve NA as a category, distinct from a new context value.
                values = frame[name].astype(STRING_DTYPE)
                categories = pd.Index(values.unique())
                codes = categories.get_indexer(values)
                self.context_categories[name] = categories
                self.context_probabilities[name] = np.bincount(codes, minlength=len(categories)) / len(frame)
                pairs = user_codes.astype(np.int64) * (len(categories) + 1) + codes + 1
                self.context_pair_counts[name] = pd.Series(pairs).value_counts(sort=False)
                self.feature_names.append("msno_" + name + "_probability")
        return self

    def _user_context_columns(self, frame: pd.DataFrame, training_context: bool):
        user_codes = self.categories["msno"].get_indexer(frame["msno"])
        known_users = user_codes >= 0
        user_counts = np.zeros(len(frame), dtype=np.float64)
        user_counts[known_users] = self.context_user_counts[user_codes[known_users]]
        for name in CONTEXT_COLUMNS:
            categories = self.context_categories[name]
            codes = categories.get_indexer(frame[name].astype(STRING_DTYPE))
            known_context = codes >= 0
            prior = np.zeros(len(frame), dtype=np.float64)
            prior[known_context] = self.context_probabilities[name][codes[known_context]]
            pairs = user_codes.astype(np.int64) * (len(categories) + 1) + codes + 1
            pair_counts = self.context_pair_counts[name].reindex(pairs, fill_value=0).to_numpy(dtype=np.float64)
            if training_context:
                if not known_users.all() or (pair_counts < 1).any():
                    raise ValueError("Training context transformation requires rows included in feature fitting")
                numerator, denominator = pair_counts - 1, user_counts - 1
            else:
                numerator, denominator = pair_counts, user_counts
            yield "msno_" + name + "_probability", ((numerator + 10 * prior) / (denominator + 10)).astype(np.float32)

    def transform(self, frame: pd.DataFrame, *, training_context: bool = False) -> pd.DataFrame:
        if not self.feature_names:
            raise RuntimeError("Fit the feature builder before transforming")
        columns = {}
        for name, values, categorical in self.raw_features(frame):
            if categorical:
                columns[name] = self.categories[name].get_indexer(values).astype(np.int32)
            else:
                columns[name] = _numeric(values).to_numpy(dtype=np.float32, na_value=np.nan)
            if name in self.frequencies:
                columns[name + "_frequency"] = values.map(self.frequencies[name]).fillna(0).to_numpy(dtype=np.float32)
        # Older serialized states have no user_context_features attribute.
        if getattr(self, "user_context_features", False):
            columns.update(self._user_context_columns(frame, training_context))
        if list(columns) != self.feature_names:
            raise RuntimeError("Feature order differs from the fitted feature state")
        return pd.DataFrame(columns, index=frame.index, copy=False)
