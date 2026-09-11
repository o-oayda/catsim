# RACS LOW1 tile indexing

## Problem

RACS LOW1 does not have a one-to-one relationship between scheduling block ID
(SBID) and sky tile. Most LOW1 observations used ASKAP's multi-field mode, so
one SBID can contain several sequential 15-minute field observations. Hale et
al. (2021) built the catalogue from individual mosaiced tiles and assigned an
overlap source to the tile whose centre was closest; `Tile_ID`, not SBID, is
therefore the catalogue's spatial ownership key.

In the galactic-cut catalogue used by CatSIM there are 784 unique `Tile_ID`
values but only 330 SBIDs. Thirty-seven SBIDs contain multiple tiles, accounting
for 491 of the 784 tiles. Every `Tile_ID` maps to exactly one catalogue
`Obs_Start_Time` and one SBID, and no two IDs describe the same nominal field
centre. Neighbouring tile footprints still overlap; without the original linear
mosaic weights, CatSIM uses the catalogue's owning tile as the available proxy.

References: [Hale et al. 2021](https://arxiv.org/abs/2109.00956) and
[Thomson et al. 2023](https://doi.org/10.1017/pasa.2023.38).

## Resolution

For LOW1 only, CatSIM sorts the unique `Tile_ID` strings and assigns the
collision-free, contiguous `int32` indices `0` through `N-1`. The original
strings remain in `tile_field_id` for provenance and diagnostics. Existing
downstream names such as `tile_sbids` are retained for compatibility, but for
LOW1 they hold encoded Tile_ID values rather than physical SBIDs.

Consumers that hold catalogue rows outside the simulator can call
`model.runtime_tile_ids(tile_ids)` to apply this same mapping. Unknown LOW1
labels map to `-1`; products that use physical integer SBIDs return those IDs
unchanged. Both `Racs` and `RacsJax` expose this method after initialisation.

Construction and cache loading verify that the integer/string mapping is
bijective, indices are unique and contiguous, each tile has one time and SBID,
and nominal field centres are unique. Versioned cache metadata prevents legacy
SBID-based LOW1 maps from being reused.

This change fixes spatial identity only. At the default 20-minute PAF
interpolation limit, 218 LOW1 tile observations still lack finite temperatures;
their interpolation or fallback policy is a separate issue.
