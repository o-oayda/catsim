from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class CatwiseConfig:
    cat_w1_max: float
    cat_w12_min: float
    magnitude_error_dist: Literal['gaussian', 'students-t']
    base_mask_version: Literal['S21', 'S22'] = 'S21'
    use_float32: bool = False
    chunk_size: int = 25_000
    store_final_samples: bool = False
    use_common_extra_error: Optional[bool] = False
    model_identifier: Optional[str] = None
    downscale_nside: Optional[int] = None
    generate_correlated_points: bool = False
    s21_catalogue_path: Optional[str] = None
    use_noecl_mask: bool = False
    add_confusion_noise: bool = False
    max_cluster_children_per_parent: int = 16
    cluster_r0_arcsec: float = 100.0
    cluster_r_cut_arcsec: float = 20.0

    def __post_init__(self) -> None:
        if self.magnitude_error_dist not in ('gaussian', 'students-t'):
            raise ValueError(
                "Magnitude_error_dist must be 'gaussian' or 'students-t'."
            )
        if self.max_cluster_children_per_parent < 0:
            raise ValueError("max_cluster_children_per_parent must be non-negative.")
        if self.cluster_r0_arcsec <= 0:
            raise ValueError("cluster_r0_arcsec must be positive.")
        if self.cluster_r_cut_arcsec < 0:
            raise ValueError("cluster_r_cut_arcsec must be non-negative.")

        if self.downscale_nside is not None:
            if self.downscale_nside <= 0:
                raise ValueError('downscale_nside must be a positive integer.')
