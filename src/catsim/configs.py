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

    def __post_init__(self) -> None:
        if self.magnitude_error_dist not in ('gaussian', 'students-t'):
            raise ValueError(
                "Magnitude_error_dist must be 'gaussian' or 'students-t'."
            )

        if self.downscale_nside is not None:
            if self.downscale_nside <= 0:
                raise ValueError('downscale_nside must be a positive integer.')
