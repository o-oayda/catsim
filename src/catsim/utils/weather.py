from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
import csv
import json
import logging
import re
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


LOGGER = logging.getLogger(__name__)
SECONDS_PER_DAY = 86400.0
MJD_UNIX_EPOCH_OFFSET_DAYS = 40587.0
ANTENNA_NAME_PATTERN = re.compile(r"(ak\d{2})")
ASKAP_LOCAL_UTC_OFFSET_HOURS = 8.0
ASKAP_LATITUDE_DEG = -26.696
ASKAP_LONGITUDE_DEG = 116.637
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_MAX_INTERPOLATION_GAP_MINUTES = 20.0


@dataclass(frozen=True)
class PafTemperatureMatch:
    antenna_names: tuple[str, ...]
    temperatures_c: np.ndarray
    matched_time_offsets_seconds: np.ndarray
    matched_unix_seconds: np.ndarray


def _mjd_scalar_to_unix_seconds(mjd_value: float) -> float:
    return (float(mjd_value) - MJD_UNIX_EPOCH_OFFSET_DAYS) * SECONDS_PER_DAY


def _mjd_to_unix_seconds(mjd_values: np.ndarray) -> np.ndarray:
    return (mjd_values - MJD_UNIX_EPOCH_OFFSET_DAYS) * SECONDS_PER_DAY


def _unix_seconds_to_mjd(unix_seconds: np.ndarray) -> np.ndarray:
    return unix_seconds / SECONDS_PER_DAY + MJD_UNIX_EPOCH_OFFSET_DAYS


def _normalise_data_dir(data_dir: Path | str) -> Path:
    return Path(data_dir).expanduser().resolve()


def _datetime_to_unix_seconds(time_value: datetime) -> float:
    return time_value.replace(tzinfo=UTC).timestamp()


def _normalise_coordinate_token(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def _open_meteo_cache_file_path(
    cache_dir: Path,
    date_str: str,
    latitude_deg: float,
    longitude_deg: float,
) -> Path:
    latitude_token = _normalise_coordinate_token(latitude_deg)
    longitude_token = _normalise_coordinate_token(longitude_deg)
    return cache_dir / f"{date_str}_lat_{latitude_token}_lon_{longitude_token}.json"


def _read_cached_open_meteo_hourly_temperature(
    cache_dir: Path | None,
    date_str: str,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if cache_dir is None:
        return None

    cache_file = _open_meteo_cache_file_path(
        cache_dir,
        date_str,
        latitude_deg,
        longitude_deg,
    )
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return (
            np.asarray(payload["time_unix"], dtype=float),
            np.asarray(payload["temperature_2m"], dtype=float),
        )
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _write_cached_open_meteo_hourly_temperature(
    cache_dir: Path | None,
    date_str: str,
    latitude_deg: float,
    longitude_deg: float,
    hourly_unix: np.ndarray,
    hourly_temperatures: np.ndarray,
) -> None:
    if cache_dir is None:
        return

    cache_file = _open_meteo_cache_file_path(
        cache_dir,
        date_str,
        latitude_deg,
        longitude_deg,
    )
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "time_unix": hourly_unix.astype(float).tolist(),
                    "temperature_2m": hourly_temperatures.astype(float).tolist(),
                }
            ),
            encoding="utf-8",
        )
    except OSError:
        return


def _parse_open_meteo_hourly_response(payload: dict) -> tuple[np.ndarray, np.ndarray]:
    hourly = payload.get("hourly")
    if hourly is None:
        raise RuntimeError("Open-Meteo response is missing the 'hourly' field.")

    time_strings = hourly.get("time")
    temperatures = hourly.get("temperature_2m")
    if time_strings is None or temperatures is None:
        raise RuntimeError(
            "Open-Meteo response is missing 'hourly.time' or "
            "'hourly.temperature_2m'."
        )

    hourly_unix = np.asarray(
        [
            _datetime_to_unix_seconds(datetime.fromisoformat(time_string))
            for time_string in time_strings
        ],
        dtype=float,
    )
    hourly_temperatures = np.asarray(temperatures, dtype=float)
    if hourly_unix.shape != hourly_temperatures.shape:
        raise RuntimeError("Open-Meteo hourly time and temperature arrays differ in length.")

    return hourly_unix, hourly_temperatures


def _fetch_open_meteo_hourly_temperature(
    start_date: str,
    end_date: str,
    latitude_deg: float,
    longitude_deg: float,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    params = urlencode(
        {
            "latitude": latitude_deg,
            "longitude": longitude_deg,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": "GMT",
        }
    )
    request_url = f"{OPEN_METEO_ARCHIVE_URL}?{params}"

    with urlopen(request_url, timeout=timeout) as response:
        payload = json.load(response)

    return _parse_open_meteo_hourly_response(payload)


def _date_range(start_date: str, end_date: str) -> list[str]:
    start = datetime.fromisoformat(start_date).date()
    end = datetime.fromisoformat(end_date).date()
    day_count = (end - start).days + 1
    return [(start + timedelta(days=day_index)).isoformat() for day_index in range(day_count)]


def _group_consecutive_dates(date_strings: list[str]) -> list[tuple[str, str]]:
    if not date_strings:
        return []

    sorted_dates = sorted(date_strings)
    ranges: list[tuple[str, str]] = []
    range_start = sorted_dates[0]
    previous = datetime.fromisoformat(sorted_dates[0]).date()

    for date_str in sorted_dates[1:]:
        current = datetime.fromisoformat(date_str).date()
        if current != previous + timedelta(days=1):
            ranges.append((range_start, previous.isoformat()))
            range_start = date_str
        previous = current

    ranges.append((range_start, previous.isoformat()))
    return ranges


def _split_hourly_data_by_date(
    hourly_unix: np.ndarray,
    hourly_temperatures: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped_data: dict[str, list[tuple[float, float]]] = {}
    for unix_time, temperature in zip(hourly_unix, hourly_temperatures, strict=True):
        date_str = datetime.fromtimestamp(float(unix_time), tz=UTC).date().isoformat()
        grouped_data.setdefault(date_str, []).append((float(unix_time), float(temperature)))

    return {
        date_str: (
            np.asarray([item[0] for item in date_values], dtype=float),
            np.asarray([item[1] for item in date_values], dtype=float),
        )
        for date_str, date_values in grouped_data.items()
    }


def _get_cached_or_fetch_open_meteo_hourly_temperature(
    start_date: str,
    end_date: str,
    latitude_deg: float,
    longitude_deg: float,
    timeout: float,
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    requested_dates = _date_range(start_date, end_date)
    cached_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing_dates: list[str] = []

    for date_str in requested_dates:
        cached = _read_cached_open_meteo_hourly_temperature(
            cache_dir,
            date_str,
            latitude_deg,
            longitude_deg,
        )
        if cached is None:
            missing_dates.append(date_str)
        else:
            cached_results[date_str] = cached

    for missing_start, missing_end in _group_consecutive_dates(missing_dates):
        fetched_unix, fetched_temperatures = _fetch_open_meteo_hourly_temperature(
            start_date=missing_start,
            end_date=missing_end,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            timeout=timeout,
        )
        split_results = _split_hourly_data_by_date(fetched_unix, fetched_temperatures)
        for date_str, date_values in split_results.items():
            _write_cached_open_meteo_hourly_temperature(
                cache_dir,
                date_str,
                latitude_deg,
                longitude_deg,
                *date_values,
            )
            cached_results[date_str] = date_values

    combined_unix = []
    combined_temperatures = []
    for date_str in requested_dates:
        if date_str not in cached_results:
            continue
        date_unix, date_temperatures = cached_results[date_str]
        combined_unix.append(date_unix)
        combined_temperatures.append(date_temperatures)

    if not combined_unix:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    return (
        np.concatenate(combined_unix).astype(float),
        np.concatenate(combined_temperatures).astype(float),
    )


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
    LOGGER.debug(
        "Loaded %d PAF antenna temperature files from %s",
        len(series),
        normalised_data_dir,
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


def get_open_meteo_temperatures_for_mjd(
    mjd_values: Iterable[float],
    latitude_deg: float = ASKAP_LATITUDE_DEG,
    longitude_deg: float = ASKAP_LONGITUDE_DEG,
    timeout: float = 60.0,
    cache_dir: Path | str | None = None,
) -> np.ndarray:
    """Fetch Open-Meteo 2 m temperatures and linearly interpolate to UTC MJDs."""
    mjd_array = np.asarray(mjd_values, dtype=float)
    if mjd_array.size == 0:
        return np.asarray([], dtype=float)

    unique_mjd, inverse_indices = np.unique(mjd_array, return_inverse=True)
    buffer_days = 1.0 / 24.0
    first_request_unix = _mjd_to_unix_seconds(
        np.asarray(unique_mjd.min() - buffer_days)
    )
    last_request_unix = _mjd_to_unix_seconds(
        np.asarray(unique_mjd.max() + buffer_days)
    )
    start_date = datetime.fromtimestamp(
        float(first_request_unix),
        tz=UTC,
    ).date().isoformat()
    end_date = datetime.fromtimestamp(
        float(last_request_unix),
        tz=UTC,
    ).date().isoformat()

    normalised_cache_dir = (
        Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    )
    hourly_unix, hourly_temperatures = _get_cached_or_fetch_open_meteo_hourly_temperature(
        start_date=start_date,
        end_date=end_date,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        timeout=timeout,
        cache_dir=normalised_cache_dir,
    )
    if hourly_unix.size == 0:
        return np.full(mjd_array.shape, np.nan, dtype=float)

    order = np.argsort(hourly_unix, kind="stable")
    hourly_mjd = _unix_seconds_to_mjd(hourly_unix[order])
    hourly_temperatures = hourly_temperatures[order]

    tolerance_days = 1.0 / SECONDS_PER_DAY
    in_range = (
        unique_mjd >= (hourly_mjd.min() - tolerance_days)
    ) & (
        unique_mjd <= (hourly_mjd.max() + tolerance_days)
    )
    clipped_unique_mjd = np.clip(unique_mjd, hourly_mjd.min(), hourly_mjd.max())

    interpolated_unique = np.full(unique_mjd.shape, np.nan, dtype=float)
    interpolated_unique[in_range] = np.interp(
        clipped_unique_mjd[in_range],
        hourly_mjd,
        hourly_temperatures,
    )
    return np.asarray(interpolated_unique[inverse_indices], dtype=float)
