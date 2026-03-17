from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np


ASKAP_LATITUDE_DEG = -26.696
ASKAP_LONGITUDE_DEG = 116.637
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_CACHE_DIR = Path.home() / ".catsim" / "weather_cache"
SECONDS_PER_DAY = 86400.0
MJD_UNIX_EPOCH_OFFSET_DAYS = 40587.0


def _datetime_to_unix_seconds(time_value: datetime) -> float:
    return time_value.replace(tzinfo=UTC).timestamp()


def _mjd_to_unix_seconds(mjd_values: np.ndarray) -> np.ndarray:
    return (mjd_values - MJD_UNIX_EPOCH_OFFSET_DAYS) * SECONDS_PER_DAY


def _unix_seconds_to_mjd(unix_seconds: np.ndarray) -> np.ndarray:
    return unix_seconds / SECONDS_PER_DAY + MJD_UNIX_EPOCH_OFFSET_DAYS


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


def _normalise_coordinate_token(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def _get_cache_file_path(date_str: str, latitude_deg: float, longitude_deg: float) -> Path:
    latitude_token = _normalise_coordinate_token(latitude_deg)
    longitude_token = _normalise_coordinate_token(longitude_deg)
    file_name = f"{date_str}_lat_{latitude_token}_lon_{longitude_token}.json"
    return WEATHER_CACHE_DIR / file_name


def _write_cached_hourly_temperature_for_date(
    date_str: str,
    latitude_deg: float,
    longitude_deg: float,
    hourly_unix: np.ndarray,
    hourly_temperatures: np.ndarray,
) -> None:
    cache_file = _get_cache_file_path(date_str, latitude_deg, longitude_deg)
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time_unix": hourly_unix.astype(float).tolist(),
            "temperature_2m": hourly_temperatures.astype(float).tolist(),
        }
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        return


def _read_cached_hourly_temperature_for_date(
    date_str: str,
    latitude_deg: float,
    longitude_deg: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    cache_file = _get_cache_file_path(date_str, latitude_deg, longitude_deg)
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except OSError:
        return None

    return (
        np.asarray(payload["time_unix"], dtype=float),
        np.asarray(payload["temperature_2m"], dtype=float),
    )


def _fetch_open_meteo_hourly_temperature(
    start_date: str,
    end_date: str,
    latitude_deg: float,
    longitude_deg: float,
    timeout: float = 60.0,
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


def _get_cached_or_fetch_hourly_temperature(
    start_date: str,
    end_date: str,
    latitude_deg: float,
    longitude_deg: float,
    timeout: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    requested_dates = _date_range(start_date, end_date)
    cached_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing_dates: list[str] = []

    for date_str in requested_dates:
        cached = _read_cached_hourly_temperature_for_date(
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
            _write_cached_hourly_temperature_for_date(
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


def get_temperatures_for_mjd(
    mjd_values: Iterable[float],
    latitude_deg: float = ASKAP_LATITUDE_DEG,
    longitude_deg: float = ASKAP_LONGITUDE_DEG,
    timeout: float = 60.0,
) -> np.ndarray:
    """Fetch interpolated hourly 2m temperatures for a sequence of UTC MJDs."""
    mjd_array = np.asarray(mjd_values, dtype=float)
    if mjd_array.size == 0:
        return np.asarray([], dtype=float)

    unique_source_mjd, inverse_indices = np.unique(mjd_array, return_inverse=True)

    buffer_days = 1.0 / 24.0
    first_request_unix = _mjd_to_unix_seconds(np.asarray(unique_source_mjd.min() - buffer_days))
    last_request_unix = _mjd_to_unix_seconds(np.asarray(unique_source_mjd.max() + buffer_days))
    start_date = datetime.fromtimestamp(float(first_request_unix), tz=UTC).date().isoformat()
    end_date = datetime.fromtimestamp(float(last_request_unix), tz=UTC).date().isoformat()

    hourly_unix, hourly_temperatures = _get_cached_or_fetch_hourly_temperature(
        start_date=start_date,
        end_date=end_date,
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        timeout=timeout,
    )

    if hourly_unix.size == 0:
        return np.full(mjd_array.shape, np.nan, dtype=float)

    hourly_mjd = _unix_seconds_to_mjd(hourly_unix)
    tolerance_days = 1.0 / SECONDS_PER_DAY
    in_range = (
        unique_source_mjd >= (hourly_mjd.min() - tolerance_days)
    ) & (
        unique_source_mjd <= (hourly_mjd.max() + tolerance_days)
    )
    clipped_source_mjd = np.clip(
        unique_source_mjd,
        hourly_mjd.min(),
        hourly_mjd.max(),
    )

    interpolated_unique = np.full(unique_source_mjd.shape, np.nan, dtype=float)
    interpolated_unique[in_range] = np.interp(
        clipped_source_mjd[in_range],
        hourly_mjd,
        hourly_temperatures,
    )
    return np.asarray(interpolated_unique[inverse_indices], dtype=float)
