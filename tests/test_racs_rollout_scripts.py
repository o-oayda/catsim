from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "compare_racs_noise_ensembles.py"
)
_SPEC = importlib.util.spec_from_file_location("compare_racs_noise_ensembles", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
downgrade_nested_count_maps = _MODULE.downgrade_nested_count_maps
ensemble_comparison = _MODULE.ensemble_comparison
load_artifact = _MODULE.load_artifact
main = _MODULE.main

_PRECOMPUTE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "precompute_racs_noise_lookup.py"
)
_PRECOMPUTE_SPEC = importlib.util.spec_from_file_location(
    "precompute_racs_noise_lookup", _PRECOMPUTE_PATH
)
assert _PRECOMPUTE_SPEC is not None and _PRECOMPUTE_SPEC.loader is not None
_PRECOMPUTE_MODULE = importlib.util.module_from_spec(_PRECOMPUTE_SPEC)
_PRECOMPUTE_SPEC.loader.exec_module(_PRECOMPUTE_MODULE)


def _science_config() -> dict[str, object]:
    return {
        "product": "mid1",
        "nside": 64,
        "chunk_size": 1000,
        "batch_size": 2,
        "flux_min": 15.0,
        "log10_n_initial_samples": 4.0,
        "cluster_count_model": "geometric",
        "max_cluster_children_per_parent": 16,
        "p_clus": 0.0,
        "clus_stop_prob": 1.0,
        "lambda_clus": 0.0,
        "observer_speed": 1.0,
        "temp_beta": 0.0,
        "elevation_amp": 0.0,
        "elevation_trough": 0.0,
        "fractional_error_eta": 0.0,
        "alpha_mean": 0.8,
        "alpha_sigma": 0.2,
        "cluster_r0_arcsec": 100.0,
        "cluster_r_cut_arcsec": 20.0,
        "temperature_model": "hot_linear",
        "paf_reference_temp_c": 25.0,
    }


def _artifact(path: Path, maps: np.ndarray, rejected: np.ndarray) -> None:
    np.savez_compressed(
        path,
        maps=maps,
        rejected_invalid_noise_maps=rejected,
        metadata_json=np.asarray(
            json.dumps(
                {
                    "commit": "0123456789abcdef",
                    "config": _science_config(),
                    "seeds": [10, 11],
                    "cache_identities": {"error": "lookup-sha"},
                    "environment": {"python": "test"},
                }
            )
        ),
    )


def test_nested_count_downgrade_preserves_counts_and_wholly_invalid_pixels():
    maps = np.arange(hp.nside2npix(8), dtype=np.float64)[None, :]
    maps[:, :4] = np.nan
    coarse = downgrade_nested_count_maps(maps, 4)
    assert coarse.shape == (1, hp.nside2npix(4))
    assert np.isnan(coarse[0, 0])
    assert coarse[0, 1] == np.sum(np.arange(4, 8))
    assert np.nansum(coarse) == pytest.approx(np.nansum(maps))


def test_ensemble_comparison_uses_uncertainty_of_the_two_means():
    old = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    new = np.asarray([[3.0, 4.0], [5.0, 6.0]])
    result = ensemble_comparison(old, new)
    np.testing.assert_allclose(result["difference"], [2.0, 2.0])
    np.testing.assert_allclose(result["difference_uncertainty"], [np.sqrt(2), np.sqrt(2)])
    np.testing.assert_allclose(result["standardized_difference"], np.sqrt(2))
    np.testing.assert_allclose(result["old_total_counts"], [3.0, 7.0])


def test_comparison_cli_writes_nside64_and_nside4_products(tmp_path: Path):
    npix = hp.nside2npix(64)
    old_maps = np.zeros((2, npix), dtype=np.float32)
    new_maps = np.zeros((2, npix), dtype=np.float32)
    old_maps[:, 0] = [1, 3]
    new_maps[:, 0] = [3, 5]
    old_rejected = np.zeros_like(old_maps)
    new_rejected = np.zeros_like(new_maps)
    new_rejected[:, 1] = [2, 4]
    old_path = tmp_path / "old.npz"
    new_path = tmp_path / "new.npz"
    _artifact(old_path, old_maps, old_rejected)
    _artifact(new_path, new_maps, new_rejected)

    output = tmp_path / "comparison"
    main(
        [
            "--old",
            str(old_path),
            "--new",
            str(new_path),
            "--output-dir",
            str(output),
            "--no-plots",
        ]
    )

    for nside in (64, 4):
        assert (output / f"comparison_nside{nside}.npz").is_file()
    metadata = json.loads((output / "comparison_metadata.json").read_text())
    assert metadata["old_artifact"]["metadata"]["commit"] == "0123456789abcdef"
    assert metadata["levels"]["64"]["new_rejected_total_mean"] == pytest.approx(3.0)
    assert metadata["verified_realization_paired_uncertainty"] is False
    with np.load(output / "comparison_nside4.npz") as data:
        assert data["difference"][0] == pytest.approx(2.0)
        assert data["rejected_new_mean"][0] == pytest.approx(3.0)


def test_comparison_artifact_requires_pinned_metadata(tmp_path: Path):
    npix = hp.nside2npix(64)
    path = tmp_path / "incomplete.npz"
    np.savez_compressed(
        path,
        maps=np.zeros((1, npix)),
        rejected_invalid_noise_maps=np.zeros((1, npix)),
        metadata_json=np.asarray(json.dumps({"commit": "abc"})),
    )
    with pytest.raises(ValueError, match="must pin commit"):
        load_artifact(path)


def test_comparison_requires_explicit_verified_pairing(tmp_path: Path):
    npix = hp.nside2npix(64)
    metadata = {
        "commit": "abc",
        "config": _science_config(),
        "seeds": [1, 2],
        "cache_identities": {"error": "lookup"},
        "environment": {"python": "test"},
        "pairing_verified": True,
        "paired_realization_ids": ["source-1", "source-2"],
    }
    paths = []
    for name in ("old", "new"):
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(
            path,
            maps=np.zeros((2, npix)),
            rejected_invalid_noise_maps=np.zeros((2, npix)),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
        paths.append(path)
    output = tmp_path / "paired"
    main(
        [
            "--old", str(paths[0]),
            "--new", str(paths[1]),
            "--output-dir", str(output),
            "--no-plots",
        ]
    )
    summary = json.loads((output / "comparison_metadata.json").read_text())
    assert summary["verified_realization_paired_uncertainty"] is True


def test_comparison_rejects_mismatched_science_configuration(tmp_path: Path):
    npix = hp.nside2npix(64)
    paths = []
    for name, flux_min in (("old", 15.0), ("new", 20.0)):
        config = _science_config()
        config["flux_min"] = flux_min
        path = tmp_path / f"{name}.npz"
        np.savez_compressed(
            path,
            maps=np.zeros((2, npix)),
            rejected_invalid_noise_maps=np.zeros((2, npix)),
            metadata_json=np.asarray(
                json.dumps(
                    {
                        "commit": name,
                        "config": config,
                        "seeds": [1, 2],
                        "cache_identities": {"error": name},
                        "environment": {"python": "test"},
                    }
                )
            ),
        )
        paths.append(path)
    with pytest.raises(ValueError, match="incompatible science configuration"):
        main(
            [
                "--old", str(paths[0]),
                "--new", str(paths[1]),
                "--output-dir", str(tmp_path / "comparison"),
                "--no-plots",
            ]
        )


def test_precompute_cli_exposes_default_production_grid():
    args = _PRECOMPUTE_MODULE.parse_args(["--product", "mid1"])
    assert args.noise_map_nside == 256
    assert args.noise_bins is None
    assert args.flux_bins is None
    assert args.noise_bounds is None
    assert args.flux_bounds is None
    assert args.min_cell_count == 10
    assert args.rebuild == "none"


def test_comparison_plot_helpers_create_outputs(tmp_path: Path):
    old = np.vstack([np.arange(12), np.arange(12) + 1.0])
    new = old + np.asarray([[0.0], [1.0]])
    comparison = ensemble_comparison(old, new)
    sky_path = tmp_path / "sky.png"
    optional_path = tmp_path / "power.png"
    _MODULE._plot_healpix_panels(comparison, sky_path, 1)
    _MODULE._plot_optional_ensemble(
        np.vstack([np.arange(4), np.arange(4) + 1]),
        np.vstack([np.arange(4) + 1, np.arange(4) + 2]),
        optional_path,
        "Power comparison",
    )
    assert sky_path.is_file()
    assert optional_path.is_file()
