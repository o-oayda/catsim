from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
import csv
import re
from typing import Iterable

import numpy as np


SECONDS_PER_DAY = 86400.0
MJD_UNIX_EPOCH_OFFSET_DAYS = 40587.0
ANTENNA_NAME_PATTERN = re.compile(r"(ak\d{2})")
ASKAP_LOCAL_UTC_OFFSET_HOURS = 8.0
DEFAULT_MAX_INTERPOLATION_GAP_MINUTES = 20.0


@dataclass(frozen=True)
class PafTemperatureMatch:
    antenna_names: tuple[str, ...]
    temperatures_c: np.ndarray
    matched_time_offsets_seconds: np.ndarray
    matched_unix_seconds: np.ndarray


def _mjd_scalar_to_unix_seconds(mjd_value: float) -> float:
    return (float(mjd_value) - MJD_UNIX_EPOCH_OFFSET_DAYS) * SECONDS_PER_DAY


def _normalise_data_dir(data_dir: Path | str) -> Path:
    return Path(data_dir).expanduser().resolve()


def _parse_antenna_name(path: Path) -> str:
    match = ANTENNA_NAME_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Could not parse ASKAP antenna name from {path}.")
    return match.group(1)


def _load_single_temperature_series(
    csv_path: Path,
    utc_offset_hours: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    valid_unix_seconds: list[float] = []
    valid_temperatures: list[float] = []

    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if len(row) < 2 or row[1] == "":
                continue
            timestamp = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            valid_unix_seconds.append(
                timestamp.timestamp() - utc_offset_hours * 3600.0
            )
            valid_temperatures.append(float(row[1]))

    return (
        np.asarray(valid_unix_seconds, dtype=float),
        np.asarray(valid_temperatures, dtype=float),
    )


def _load_single_paf_temperature_series(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    return _load_single_temperature_series(
        csv_path,
        utc_offset_hours=ASKAP_LOCAL_UTC_OFFSET_HOURS,
    )


@lru_cache(maxsize=None)
def _load_paf_temperature_series_cached(
    data_dir_str: str,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    data_dir = Path(data_dir_str)
    csv_paths = sorted(data_dir.glob("ak*csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No per-antenna PAF temperature files found in {data_dir}.")

    series: list[tuple[str, np.ndarray, np.ndarray]] = []
    for csv_path in csv_paths:
        antenna_name = _parse_antenna_name(csv_path)
        unix_seconds, temperatures_c = _load_single_paf_temperature_series(csv_path)
        series.append((antenna_name, unix_seconds, temperatures_c))

    antenna_names = [item[0] for item in series]
    if len(antenna_names) != 36:
        raise ValueError(
            f"Expected 36 ASKAP antenna temperature files, found {len(antenna_names)} in {data_dir}."
        )
    if antenna_names != [f"ak{antenna_index:02d}" for antenna_index in range(1, 37)]:
        raise ValueError("PAF temperature files do not cover the expected ak01-ak36 antennas.")

    return tuple(series)


def load_paf_temperature_series(
    data_dir: Path | str,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    normalised_data_dir = _normalise_data_dir(data_dir)
    series = _load_paf_temperature_series_cached(str(normalised_data_dir))
    print(
        f"Loaded {len(series)} PAF antenna temperature files from {normalised_data_dir}",
    )
    return series


def _interpolate_temperature_at_unix_seconds(
    unix_seconds: np.ndarray,
    temperatures_c: np.ndarray,
    target_unix_seconds: float,
    max_interpolation_gap_seconds: float,
) -> tuple[float, float, float]:
    if unix_seconds.size == 0:
        return np.nan, np.nan, np.nan

    insertion_index = int(np.searchsorted(unix_seconds, target_unix_seconds))
    if insertion_index < unix_seconds.size and unix_seconds[insertion_index] == target_unix_seconds:
        return (
            float(temperatures_c[insertion_index]),
            0.0,
            float(unix_seconds[insertion_index]),
        )

    if insertion_index == 0 or insertion_index == unix_seconds.size:
        return np.nan, np.nan, np.nan

    previous_index = insertion_index - 1
    next_index = insertion_index
    previous_unix_seconds = float(unix_seconds[previous_index])
    next_unix_seconds = float(unix_seconds[next_index])
    interpolation_gap_seconds = next_unix_seconds - previous_unix_seconds
    if interpolation_gap_seconds <= 0.0 or interpolation_gap_seconds > max_interpolation_gap_seconds:
        return np.nan, np.nan, np.nan

    previous_temperature_c = float(temperatures_c[previous_index])
    next_temperature_c = float(temperatures_c[next_index])
    interpolation_fraction = (
        (target_unix_seconds - previous_unix_seconds) / interpolation_gap_seconds
    )
    interpolated_temperature_c = previous_temperature_c + interpolation_fraction * (
        next_temperature_c - previous_temperature_c
    )

    previous_offset_seconds = abs(target_unix_seconds - previous_unix_seconds)
    next_offset_seconds = abs(next_unix_seconds - target_unix_seconds)
    if previous_offset_seconds <= next_offset_seconds:
        nearest_offset_seconds = previous_offset_seconds
        nearest_unix_seconds = previous_unix_seconds
    else:
        nearest_offset_seconds = next_offset_seconds
        nearest_unix_seconds = next_unix_seconds

    return (
        float(interpolated_temperature_c),
        float(nearest_offset_seconds),
        float(nearest_unix_seconds),
    )


def get_paf_antenna_temperatures_for_observation(
    observation_mjd: float,
    data_dir: Path | str,
    max_interpolation_gap_minutes: float = DEFAULT_MAX_INTERPOLATION_GAP_MINUTES,
) -> PafTemperatureMatch:
    target_unix_seconds = _mjd_scalar_to_unix_seconds(observation_mjd)
    max_interpolation_gap_seconds = float(max_interpolation_gap_minutes) * 60.0

    antenna_names: list[str] = []
    temperatures_c: list[float] = []
    matched_time_offsets_seconds: list[float] = []
    matched_unix_seconds: list[float] = []

    for antenna_name, unix_seconds, antenna_temperatures in load_paf_temperature_series(data_dir):
        antenna_names.append(antenna_name)
        (
            interpolated_temperature_c,
            nearest_offset_seconds,
            nearest_unix_seconds,
        ) = _interpolate_temperature_at_unix_seconds(
            unix_seconds,
            antenna_temperatures,
            target_unix_seconds,
            max_interpolation_gap_seconds,
        )
        temperatures_c.append(interpolated_temperature_c)
        matched_time_offsets_seconds.append(nearest_offset_seconds)
        matched_unix_seconds.append(nearest_unix_seconds)

    return PafTemperatureMatch(
        antenna_names=tuple(antenna_names),
        temperatures_c=np.asarray(temperatures_c, dtype=float),
        matched_time_offsets_seconds=np.asarray(matched_time_offsets_seconds, dtype=float),
        matched_unix_seconds=np.asarray(matched_unix_seconds, dtype=float),
    )


def get_mean_paf_temperature_for_observation(
    observation_mjd: float,
    data_dir: Path | str,
    max_interpolation_gap_minutes: float = DEFAULT_MAX_INTERPOLATION_GAP_MINUTES,
) -> float:
    match = get_paf_antenna_temperatures_for_observation(
        observation_mjd,
        data_dir=data_dir,
        max_interpolation_gap_minutes=max_interpolation_gap_minutes,
    )
    finite_temperatures = match.temperatures_c[np.isfinite(match.temperatures_c)]
    if finite_temperatures.size == 0:
        return float("nan")
    return float(np.mean(finite_temperatures, dtype=float))


def get_mean_paf_temperatures_for_observations(
    observation_mjd_values: Iterable[float],
    data_dir: Path | str,
    max_interpolation_gap_minutes: float = DEFAULT_MAX_INTERPOLATION_GAP_MINUTES,
) -> np.ndarray:
    mjd_array = np.asarray(observation_mjd_values, dtype=float)
    if mjd_array.size == 0:
        return np.asarray([], dtype=float)

    unique_mjd, inverse_indices = np.unique(mjd_array, return_inverse=True)
    mean_temperatures = np.asarray(
        [
            get_mean_paf_temperature_for_observation(
                observation_mjd=obs_mjd,
                data_dir=data_dir,
                max_interpolation_gap_minutes=max_interpolation_gap_minutes,
            )
            for obs_mjd in unique_mjd
        ],
        dtype=float,
    )
    return mean_temperatures[inverse_indices]
