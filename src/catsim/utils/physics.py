from typing import Optional
import numpy as np
from numpy.typing import NDArray
from astropy.modeling.rotations import RotateCelestial2Native
import astropy.units as u
import healpy as hp


def sample_spherical_points(
        n_points: int,
        rng: Optional[np.random.Generator] = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if rng is None:
        rng = np.random.default_rng()

    longitudes_deg = 360 * rng.random(n_points)
    latitudes_deg = np.rad2deg(
        np.arcsin(2 * rng.random(n_points) - 1)
    )
    return longitudes_deg, latitudes_deg


def compute_boosted_angles(
        source_frame_angles: NDArray,
        observer_speed: float
    ) -> NDArray[np.float64]:
    '''
    Given an angle between the direction of motion and the source
    in the source frame, find the boosted angle, corresponding to the
    angle perceived in the observer's frame.
    
    :param source_frame_angles: the angle in degrees between the
        direction of motion (i.e. the dipole vector) and the source.
    :param observer_speed: the speed (in units of c) of the observer.
    '''
    source_frame_angles = np.deg2rad(source_frame_angles)
    return np.rad2deg(
        np.arccos(
            (observer_speed + np.cos(source_frame_angles))
            / (observer_speed * np.cos(source_frame_angles) + 1)
        )
    )


def aberrate_points(
        rest_longitudes: NDArray,
        rest_latitudes: NDArray,
        observer_direction: tuple[float, float],
        observer_speed: float,
        rotation_matrices: tuple[NDArray[np.float64], NDArray[np.float64]] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''
    Aberrate points by transforming into a frame with the dipole vector as the
    pole, boosting the latitude angle, then transforming back into the native frame.

    :param rest_longitudes: Source frame longitudes in degrees.
    :param rest_latitudes: Source frame latitudes in degrees.
    :param observer_direction: Dipole direction in degrees, format (long, lat).
    :param observer_speed: Observer speed in units of c.
    '''
    dipole_longitude, dipole_latitude = observer_direction
    
    if rotation_matrices is None:
        forward_matrix, inverse_matrix = rotation_matrices_for_dipole(
            dipole_longitude=dipole_longitude,
            dipole_latitude=dipole_latitude
        )
    else:
        forward_matrix, inverse_matrix = rotation_matrices

    lon_rad = np.deg2rad(rest_longitudes)
    lat_rad = np.deg2rad(rest_latitudes)

    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)

    dipole_x = (
        forward_matrix[0, 0] * x
      + forward_matrix[0, 1] * y
      + forward_matrix[0, 2] * z
    )
    dipole_y = (
        forward_matrix[1, 0] * x
      + forward_matrix[1, 1] * y
      + forward_matrix[1, 2] * z
    )
    dipole_z = (
        forward_matrix[2, 0] * x
      + forward_matrix[2, 1] * y
      + forward_matrix[2, 2] * z
    )

    dipole_frame_longitudes = (np.degrees(np.arctan2(dipole_y, dipole_x)) + 360.0) % 360.0

    # i.e. the polar angle theta
    source_to_dipole_angle = np.degrees(np.arccos(np.clip(dipole_z, -1.0, 1.0)))
    
    boosted_source_to_dipole_angle = compute_boosted_angles(
        source_frame_angles=source_to_dipole_angle,
        observer_speed=observer_speed
    )
    boosted_dipole_frame_latitudes = 90. - boosted_source_to_dipole_angle
    del boosted_source_to_dipole_angle

    boosted_lat_rad = np.deg2rad(boosted_dipole_frame_latitudes)
    cos_boosted_lat = np.cos(boosted_lat_rad)

    boosted_x = cos_boosted_lat * np.cos(np.deg2rad(dipole_frame_longitudes))
    boosted_y = cos_boosted_lat * np.sin(np.deg2rad(dipole_frame_longitudes))
    boosted_z = np.sin(boosted_lat_rad)

    native_x = (
        inverse_matrix[0, 0] * boosted_x
      + inverse_matrix[0, 1] * boosted_y
      + inverse_matrix[0, 2] * boosted_z
    )
    native_y = (
        inverse_matrix[1, 0] * boosted_x
      + inverse_matrix[1, 1] * boosted_y
      + inverse_matrix[1, 2] * boosted_z
    )
    native_z = (
        inverse_matrix[2, 0] * boosted_x
      + inverse_matrix[2, 1] * boosted_y
      + inverse_matrix[2, 2] * boosted_z
    )

    boosted_longitudes = (np.degrees(np.arctan2(native_y, native_x)) + 360.0) % 360.0
    boosted_latitudes = np.degrees(np.arcsin(np.clip(native_z, -1.0, 1.0)))

    return boosted_longitudes, boosted_latitudes, source_to_dipole_angle


def lorentz_factor(observer_speed: float) -> float: 
    return 1 /  np.sqrt(1 - observer_speed ** 2)


def doppler_shift_factor(
        observer_speed: float,
        angle_to_source: NDArray,
    ) -> NDArray[np.float64]:
    angle_to_source = np.deg2rad(angle_to_source) # type: ignore
    gamma = lorentz_factor(observer_speed)
    return gamma * ( 1 + observer_speed * np.cos(angle_to_source) )


def boost_magnitudes(
        magnitudes: NDArray,
        angle_to_source: NDArray,
        observer_speed: float,
        spectral_index: float | NDArray,
    ) -> NDArray[np.float64]:
    '''
    Since m_nu = -2.5 log_10 (S_nu) + ZP and S'_nu = S_nu delta ** (1 + alpha),
    we can write the boosted magnitude as a function of function of rest frame
    magnitude:
    
    m'_nu = m_nu - 2.5 (1 + alpha) log_10 (delta). 
    '''
    delta = doppler_shift_factor(observer_speed, angle_to_source)
    return magnitudes - 2.5 * (1 + spectral_index) * np.log10(delta)


def spherical_to_cart_deg(
        longitudes_deg: NDArray[np.float64],
        latitudes_deg: NDArray[np.float64]
    ) -> NDArray[np.float64]:
    """Convert lon/lat (deg) to Cartesian unit vectors.

    Notes
    -----
    This uses the astronomical convention: longitude ``λ`` and latitude ``β``
    measured from the equatorial plane (β = 0° on the equator, +90° at the
    north pole). It therefore differs from the convention where
    ``θ`` is the colatitude. The trigonometric identities reduce to::

        x = cos β · cos λ
        y = cos β · sin λ
        z = sin β
    """
    lon_rad = np.deg2rad(longitudes_deg)
    lat_rad = np.deg2rad(latitudes_deg)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack((x, y, z), axis=-1).astype(np.float64)


def rotation_matrices_for_dipole(
        dipole_longitude: float,
        dipole_latitude: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rot_forward = RotateCelestial2Native(
        lon=dipole_longitude * u.degree, # pyright: ignore[reportAttributeAccessIssue]
        lat=dipole_latitude * u.degree,  # pyright: ignore[reportAttributeAccessIssue]
        lon_pole=0.0 * u.degree          # pyright: ignore[reportAttributeAccessIssue]
    )

    basis_lon = np.array([0.0, 90.0, 0.0])
    basis_lat = np.array([0.0, 0.0, 90.0])

    rot_lon, rot_lat = rot_forward(basis_lon, basis_lat)
    rotated_vectors = spherical_to_cart_deg(rot_lon, rot_lat)

    forward = rotated_vectors.T
    inverse = forward.T
    return forward.astype(np.float64), inverse.astype(np.float64)


def omega_to_theta(omega: float) -> np.float64:
    '''
    Convert solid angle in steradins to theta in radians for
    a cone section of a sphere.

    :param omega: solid angle in steradians.
    '''
    return np.arccos( 1 - omega / (2 * np.pi) )


def make_tangent_basis(unit_point_vector: NDArray) -> tuple[NDArray, NDArray]:
    '''
    For a given unit vector on the sphere, determine two orthonormal unit vectors
    lying in the tangent plane to the unit vector.
    '''
    zhat = np.asarray([0., 0., 1.])
    yhat = np.asarray([0., 1., 0.])

    cos_to_z = np.dot(unit_point_vector, zhat) 
    use_z = np.abs(cos_to_z) < 0.9

    ref = np.empty_like(unit_point_vector)
    ref[use_z] = zhat
    ref[~use_z] = yhat

    e1 = np.cross(ref, unit_point_vector)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)

    e2 = np.cross(unit_point_vector, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True)

    assert np.isclose(np.linalg.norm(e1, axis=1), 1.).all()
    assert np.isclose(np.linalg.norm(e2, axis=1), 1.).all()

    return e1, e2


def generate_clusters(
        parent_unit_vectors: NDArray,
        cluster_rate_param: float,
        kappa: float,
        rng: Optional[np.random.Generator]
) -> tuple[NDArray, NDArray]:
    if rng is None:
        rng = np.random.default_rng()

    n_parents = parent_unit_vectors.shape[0]
    per_cluster_n_offspring = rng.poisson(cluster_rate_param * np.ones(n_parents))
    parent_idxs = np.repeat(np.arange(n_parents), per_cluster_n_offspring)
    total_n_offspring = np.sum(per_cluster_n_offspring)
    sigma = kappa ** (-1/2)

    e1, e2 = make_tangent_basis(parent_unit_vectors)

    uv = rng.normal(scale=sigma, size=(int(total_n_offspring), 2))
    u = uv[:, 0]; v = uv[:, 1]

    # transform perturbation in tangent plane (u,v) back to xyz on unit sphere
    cluster_vectors = (
        parent_unit_vectors[parent_idxs, :]
      + u[:, None] * e1[parent_idxs]
      + v[:, None] * e2[parent_idxs]
    )

    norms = np.linalg.norm(cluster_vectors, axis=1, keepdims=True)
    cluster_vectors /= norms
    long_cluster_deg, lat_cluster_deg = hp.vec2ang(cluster_vectors, lonlat=True)

    return long_cluster_deg, lat_cluster_deg
