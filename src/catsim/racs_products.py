from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RacsCatalogueColumns:
    """Column names needed to derive RACS empirical lookup products."""

    ra: str
    dec: str
    tile_id: str
    total_flux: str
    total_flux_error: str
    scan_start_mjd: str
    scan_length: str
    field_id: str


@dataclass(frozen=True)
class RacsProductSpec:
    """Dataset-specific metadata for one RACS epoch/passband catalogue."""

    key: str
    label: str
    data_loader_catalogue: str
    data_loader_variant: str
    data_dir_name: str
    columns: RacsCatalogueColumns
    default_mask_filename: str | None = None
    supports_paf_temperature: bool = True

    @property
    def data_loader_args(self) -> tuple[str, str]:
        return self.data_loader_catalogue, self.data_loader_variant


RACS_LOW3 = RacsProductSpec(
    key="low3",
    label="RACS LOW3",
    data_loader_catalogue="racs",
    data_loader_variant="low3",
    data_dir_name="racs_low3",
    default_mask_filename="racs-low3_mask_nside64_ring.npy",
    columns=RacsCatalogueColumns(
        ra="RA",
        dec="Dec",
        tile_id="SBID",
        total_flux="Total_flux",
        total_flux_error="E_Total_flux",
        scan_start_mjd="Scan_start_MJD",
        scan_length="Scan_length",
        field_id="Field_ID",
    ),
)

RACS_MID1 = RacsProductSpec(
    key="mid1",
    label="RACS MID1",
    data_loader_catalogue="racs",
    data_loader_variant="mid1",
    data_dir_name="racs_mid1",
    columns=RacsCatalogueColumns(
        ra="RA",
        dec="DEC",
        tile_id="SBID",
        total_flux="Total_flux",
        total_flux_error="E_Total_flux",
        scan_start_mjd="Scan_start_MJD",
        scan_length="Scan_length",
        field_id="Tile_ID",
    ),
)


RACS_PRODUCTS: Mapping[str, RacsProductSpec] = {
    RACS_LOW3.key: RACS_LOW3,
    RACS_MID1.key: RACS_MID1,
}


def resolve_racs_product(product: str | RacsProductSpec) -> RacsProductSpec:
    """Resolve a product key or spec to a registered RACS product spec."""
    if isinstance(product, RacsProductSpec):
        return product

    key = product.lower().replace("_", "-")
    aliases = {
        "racs-low3": "low3",
        "racs_low3": "low3",
        "racs-mid1": "mid1",
        "racs_mid1": "mid1",
    }
    key = aliases.get(key, key)
    if key not in RACS_PRODUCTS:
        available = ", ".join(sorted(RACS_PRODUCTS))
        raise ValueError(f"Unknown RACS product {product!r}. Available products: {available}.")
    return RACS_PRODUCTS[key]
