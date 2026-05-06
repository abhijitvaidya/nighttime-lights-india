"""
nighttime_lights.py - Python port of xKDR's NighttimeLights.jl

Cloud bias correction and data cleaning for VIIRS nighttime lights data.
Ported from the Julia package by Hande, Patnaik, Shah, Thomas (xKDR Forum).

Original Julia package: https://github.com/xKDR/NighttimeLights.jl
Paper: "But clouds got in my way: bias and bias correction of VIIRS nighttime
lights data in the presence of clouds" (Patnaik, Shah, Thomas, 2021)

Datacube convention: 3D numpy arrays with shape (rows, cols, time).
Time is always the LAST axis.
"""

from __future__ import annotations

__version__ = "0.1.0"

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import UnivariateSpline  # kept for backwards compat


# ---------------------------------------------------------------------------
# 1. replace_negative
# ---------------------------------------------------------------------------

def replace_negative(
    datacube: np.ndarray,
    replacement: float = np.nan,
) -> np.ndarray:
    """Replace all negative values in *datacube* with *replacement*.

    Parameters
    ----------
    datacube : np.ndarray
        Array of any shape (typically a 3-D radiance datacube).
    replacement : float, default ``np.nan``
        Value to substitute for negatives.

    Returns
    -------
    np.ndarray
        Modified datacube (a copy).
    """
    out = datacube.astype(float, copy=True)
    out[out < 0] = replacement
    return out


# ---------------------------------------------------------------------------
# 2. na_recode
# ---------------------------------------------------------------------------

def na_recode(
    radiance_datacube: np.ndarray,
    ncfobs_datacube: np.ndarray,
    replacement: float = np.nan,
) -> np.ndarray:
    """Set radiance to NaN wherever cloud-free observations equal zero.

    Months with zero cloud-free observations have no valid measurement; the
    recorded radiance is an artifact and should be treated as missing.

    Parameters
    ----------
    radiance_datacube : np.ndarray
        3-D radiance datacube (rows, cols, time).
    ncfobs_datacube : np.ndarray
        3-D cloud-free observations datacube, same shape.
    replacement : float, default ``np.nan``
        Value written into radiance where ncfobs == 0.

    Returns
    -------
    np.ndarray
        Recoded radiance datacube (a copy).
    """
    out = radiance_datacube.astype(float, copy=True)
    out[ncfobs_datacube == 0] = replacement
    return out


# ---------------------------------------------------------------------------
# 3. weighted_mean
# ---------------------------------------------------------------------------

def weighted_mean(
    numbers: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Compute weighted mean, skipping NaN entries.

    Parameters
    ----------
    numbers : np.ndarray
        1-D array of values (may contain NaN).
    weights : np.ndarray
        1-D array of weights, same length as *numbers*.

    Returns
    -------
    float
        Weighted mean, or 0.0 if all weights are zero or all numbers are NaN.
    """
    numbers = np.asarray(numbers, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = ~np.isnan(numbers)
    numbers = numbers[valid]
    weights = weights[valid]

    if len(weights) == 0 or np.sum(weights) == 0:
        return 0.0

    return float(np.sum(numbers * weights) / np.sum(weights))


# ---------------------------------------------------------------------------
# 4. bgnoise_PSTT2021
# ---------------------------------------------------------------------------

def bgnoise_PSTT2021(
    radiance_datacube: np.ndarray,
    ncfobs_datacube: np.ndarray,
    threshold: float = 0.4,
) -> np.ndarray:
    """Create a binary background-noise mask (lit vs dark pixels).

    Uses the weighted mean of the *last 12 months* of radiance (weighted by
    cloud-free observations).  Pixels whose weighted mean is <= *threshold*
    are considered dark and set to NaN; lit pixels are set to 1.

    Parameters
    ----------
    radiance_datacube : np.ndarray
        3-D radiance datacube (rows, cols, time).
    ncfobs_datacube : np.ndarray
        3-D cloud-free observations datacube, same shape.
    threshold : float, default 0.4
        Radiance threshold separating dark from lit.

    Returns
    -------
    np.ndarray
        2-D mask (rows, cols) with 1.0 for lit and NaN for dark.
    """
    nrows, ncols, ntimes = radiance_datacube.shape
    last12_rad = radiance_datacube[:, :, max(0, ntimes - 12):]
    last12_cf = ncfobs_datacube[:, :, max(0, ntimes - 12):]

    mask = np.empty((nrows, ncols), dtype=float)
    for i in range(nrows):
        for j in range(ncols):
            wm = weighted_mean(last12_rad[i, j, :], last12_cf[i, j, :])
            mask[i, j] = np.nan if wm <= threshold else 1.0

    return mask


# ---------------------------------------------------------------------------
# 5. detrend_ts
# ---------------------------------------------------------------------------

def detrend_ts(ts: np.ndarray) -> np.ndarray:
    """Remove a linear time trend via OLS.

    The x-axis is ``[0, 1/12, 2/12, ...]`` (months as fractions of years).

    Parameters
    ----------
    ts : np.ndarray
        1-D time series (no NaN expected; caller should pre-filter).

    Returns
    -------
    np.ndarray
        Residuals after removing the fitted linear trend.
    """
    ts = np.asarray(ts, dtype=float)
    if len(ts) <= 1:
        return ts.copy()

    x = np.arange(len(ts)) / 12.0
    coeffs = np.polyfit(x, ts, 1)          # [slope, intercept]
    trend = np.polyval(coeffs, x)
    return ts - trend


# ---------------------------------------------------------------------------
# 6. outlier_variance
# ---------------------------------------------------------------------------

def outlier_variance(
    radiance_datacube: np.ndarray,
    mask: Optional[np.ndarray] = None,
    cutoff: float = 0.999,
) -> np.ndarray:
    """Mask out pixels whose detrended standard deviation is very high.

    For each pixel the time series is detrended, and the standard deviation of
    residuals is computed.  Pixels above the *cutoff* percentile (among lit
    pixels) are set to NaN; the rest to 1.

    Parameters
    ----------
    radiance_datacube : np.ndarray
        3-D datacube (rows, cols, time).
    mask : np.ndarray or None
        2-D mask (rows, cols).  If ``None``, all pixels are considered.
    cutoff : float, default 0.999
        Quantile threshold for standard-deviation filtering.

    Returns
    -------
    np.ndarray
        2-D mask (rows, cols) with 1.0 for stable and NaN for outlier pixels.
    """
    nrows, ncols, ntimes = radiance_datacube.shape

    if mask is None:
        mask = np.ones((nrows, ncols), dtype=float)

    stds = np.zeros((nrows, ncols), dtype=float)

    for i in range(nrows):
        for j in range(ncols):
            if np.isnan(mask[i, j]):
                continue
            pixel_ts = radiance_datacube[i, j, :]
            frac_nan = np.sum(np.isnan(pixel_ts)) / len(pixel_ts)
            if frac_nan > 0.90:
                continue
            valid = pixel_ts[~np.isnan(pixel_ts)]
            residuals = detrend_ts(valid)
            stds[i, j] = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    # Compute threshold only over lit (non-NaN mask) pixels
    lit_stds = stds[~np.isnan(mask)]
    if len(lit_stds) == 0:
        return np.ones((nrows, ncols), dtype=float)

    threshold_val = float(np.quantile(lit_stds, cutoff))

    outlier_mask = np.where(stds < threshold_val, 1.0, np.nan)
    # Combine with the input mask (propagate existing NaN)
    outlier_mask = outlier_mask * mask
    return outlier_mask


# ---------------------------------------------------------------------------
# 7. outlier_hampel
# ---------------------------------------------------------------------------

def outlier_hampel(
    ts: np.ndarray,
    window_size: int = 5,
    n_sigmas: int = 3,
) -> np.ndarray:
    """Hampel filter for temporal outlier detection.

    Outliers are replaced with NaN.  NaN positions in the input are removed
    before filtering, then reinserted at their original indices.

    Parameters
    ----------
    ts : np.ndarray
        1-D time series (may contain NaN).
    window_size : int, default 5
        Half-width of the sliding window (the full window is
        ``2 * window_size + 1`` centred on the current observation, but
        following the Julia implementation the window spans
        ``[i - window_size + 1 .. i + window_size]``).
    n_sigmas : int, default 3
        Number of MAD-scaled deviations to flag an outlier.

    Returns
    -------
    np.ndarray
        Filtered time series with outliers set to NaN.
    """
    ts = np.asarray(ts, dtype=float).copy()

    # Record NaN positions and remove them
    nan_positions = np.where(np.isnan(ts))[0].tolist()
    new_series = ts[~np.isnan(ts)].copy()

    n = len(new_series)
    k = 1.4826  # scale factor for Gaussian distribution
    outlier_indices: list[int] = []

    # Julia range: (window_size):(n - window_size), 1-indexed
    # Python equivalent: window_size-1 to n-window_size-1 inclusive
    for i in range(window_size - 1, n - window_size):
        # Window: [i - window_size + 1 .. i + window_size] (inclusive)
        lo = i - window_size + 1
        hi = i + window_size + 1  # exclusive for numpy slicing
        window = new_series[lo:hi]

        x0 = float(np.median(window))
        s0 = k * float(np.median(np.abs(window - x0)))

        if abs(new_series[i] - x0) > n_sigmas * s0:
            outlier_indices.append(i)

    new_series[outlier_indices] = np.nan

    # Reinsert NaN at original positions
    result = new_series.tolist()
    for pos in sorted(nan_positions):
        result.insert(pos, np.nan)

    return np.array(result, dtype=float)


# ---------------------------------------------------------------------------
# 8. rank_correlation_test
# ---------------------------------------------------------------------------

def rank_correlation_test(
    radiance_ts: np.ndarray,
    ncfobs_ts: np.ndarray,
) -> float:
    """Test for significant positive Spearman correlation between detrended
    radiance and cloud-free observations.

    Steps:
    - Detrends radiance only (not ncfobs).
    - Computes Spearman rank correlation.
    - Uses the textbook t-test formula: stat = r * sqrt((n-2)/(1-r^2)).
    - Returns a one-tailed (right) p-value testing H1: rho > 0.

    Note: The Julia NighttimeLights.jl has an operator-precedence bug
    where ``r * sqrt((n-2)/1 - r^2)`` evaluates as ``r * sqrt(n-2-r^2)``
    instead of the correct ``r * sqrt((n-2)/(1-r^2))``. This Python
    version uses the correct formula. The practical difference is small
    for typical sample sizes (n > 50).

    Parameters
    ----------
    radiance_ts : np.ndarray
        1-D radiance time series (NaN-free; caller should pre-filter).
    ncfobs_ts : np.ndarray
        1-D cloud-free observations, same length.

    Returns
    -------
    float
        One-tailed (right) p-value.
    """
    from scipy.stats import t as t_dist

    radiance_ts = np.asarray(radiance_ts, dtype=float)
    ncfobs_ts = np.asarray(ncfobs_ts, dtype=float)

    if len(radiance_ts) < 3:
        return 1.0  # not enough data

    y_detrended = detrend_ts(radiance_ts)
    corr, _ = stats.spearmanr(y_detrended, ncfobs_ts)

    if np.isnan(corr):
        return 1.0

    n = len(y_detrended)
    # Correct textbook formula for t-test of Spearman correlation
    if abs(corr) >= 1.0:
        return 0.0 if corr > 0 else 1.0
    stat = corr * np.sqrt((n - 2) / (1 - corr ** 2))
    # One-tailed, right tail (H1: rho > 0)
    p_value = 1.0 - t_dist.cdf(stat, n - 2)

    return float(p_value)


# ---------------------------------------------------------------------------
# 9. bias_PSTT2021_pixel
# ---------------------------------------------------------------------------

def bias_PSTT2021_pixel(
    radiance_ts: np.ndarray,
    ncfobs_ts: np.ndarray,
    smoothing_param: float = 0.09,
) -> np.ndarray:
    """Cloud-bias correction for a single pixel time series.

    Steps:
        1. Remove NaN positions from both arrays.
        2. Detrend radiance.
        3. Fit a smoothing spline: detrended radiance ~ ncfobs.
        4. topval = max(spline predictions).
        5. corrected = detrended - spline_predictions + topval.
        6. Add time trend back.
        7. Cap: if corrected > max(original), use original.
        8. Reinsert NaN at original positions.

    Parameters
    ----------
    radiance_ts : np.ndarray
        1-D radiance time series (may contain NaN).
    ncfobs_ts : np.ndarray
        1-D cloud-free observations, same length.
    smoothing_param : float, default 0.09
        Smoothing parameter for the penalized cubic spline (``csaps``).
        This is the ``smooth`` parameter in [0, 1] where 0 = linear,
        1 = interpolation.  The value 0.09 was calibrated to match
        Julia's ``SmoothingSplines.jl`` with ``lambda=10.0``.

    Returns
    -------
    np.ndarray
        Bias-corrected radiance time series, same length as input.
    """
    from csaps import csaps

    radiance_ts = np.asarray(radiance_ts, dtype=float).copy()
    ncfobs_ts = np.asarray(ncfobs_ts, dtype=float).copy()

    # 1. Record and remove NaN positions
    nan_positions = np.where(np.isnan(radiance_ts))[0].tolist()
    y = radiance_ts[~np.isnan(radiance_ts)].copy()
    x = ncfobs_ts.copy()
    # Remove the same positions from ncfobs
    mask_valid = ~np.isnan(radiance_ts)
    x = x[mask_valid]

    if len(y) < 4:
        return radiance_ts  # not enough points for spline

    # Check for enough unique x values
    if len(np.unique(x)) < 4:
        return radiance_ts

    # 2. Detrend
    y_detrended = detrend_ts(y)
    trend = y - y_detrended

    # 3. Fit penalized cubic smoothing spline (matches Julia SmoothingSplines.jl)
    # Julia's SmoothingSpline maps duplicate x values to the same prediction,
    # so we average y at duplicate x values and use counts as weights.
    unique_x = np.sort(np.unique(x))
    y_means = np.array([y_detrended[x == xi].mean() for xi in unique_x])
    weights = np.array([float(np.sum(x == xi)) for xi in unique_x])

    try:
        m_unique = csaps(unique_x, y_means, unique_x,
                         smooth=smoothing_param, weights=weights)
        # Map predictions back to original x positions
        x_to_m = dict(zip(unique_x, m_unique))
        m = np.array([x_to_m[xi] for xi in x])
    except Exception:
        return radiance_ts  # spline fitting failed; return uncorrected

    # 5. topval and correction
    topval = float(np.max(m))
    fixed1 = y_detrended - m + topval

    # 6. Add time trend back
    fixed2 = fixed1 + trend

    # 7. Cap: if corrected > max(y), use original y
    y_max = float(np.max(y))
    for i in range(len(y)):
        if fixed2[i] > y_max:
            fixed2[i] = y[i]

    # 8. Reinsert NaN at original positions
    result = fixed2.tolist()
    for pos in sorted(nan_positions):
        result.insert(pos, np.nan)

    return np.array(result, dtype=float)


# ---------------------------------------------------------------------------
# 10. bias_PSTT2021
# ---------------------------------------------------------------------------

def bias_PSTT2021(
    radiance_datacube: np.ndarray,
    ncfobs_datacube: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply cloud-bias correction to every eligible pixel in the datacube.

    A pixel is corrected only if:
    - The mask value is not NaN (pixel is lit and stable).
    - Fewer than 90 % of radiance values are NaN.
    - Detrended radiance has a significant Spearman correlation with ncfobs
      (p < 0.05).

    Parameters
    ----------
    radiance_datacube : np.ndarray
        3-D radiance datacube (rows, cols, time).
    ncfobs_datacube : np.ndarray
        3-D cloud-free observations datacube, same shape.
    mask : np.ndarray or None
        2-D mask (rows, cols).  If ``None``, all pixels are considered.

    Returns
    -------
    np.ndarray
        Bias-corrected radiance datacube (a copy).
    """
    nrows, ncols, ntimes = radiance_datacube.shape
    out = radiance_datacube.astype(float, copy=True)

    if mask is None:
        mask = np.ones((nrows, ncols), dtype=float)

    for i in range(nrows):
        if i % 50 == 0:
            print(f"  bias_PSTT2021: processing row {i + 1}/{nrows}")
        for j in range(ncols):
            if np.isnan(mask[i, j]):
                continue

            pixel_ts = out[i, j, :]
            frac_nan = np.sum(np.isnan(pixel_ts)) / len(pixel_ts)
            if frac_nan > 0.90:
                continue

            # Prepare NaN-free arrays for rank correlation test
            valid_mask = ~np.isnan(pixel_ts)
            radiance_arr = pixel_ts[valid_mask]
            ncfobs_arr = ncfobs_datacube[i, j, :].astype(float)[valid_mask]

            if len(radiance_arr) < 4:
                continue

            p_val = rank_correlation_test(radiance_arr, ncfobs_arr)
            if p_val < 0.05:
                out[i, j, :] = bias_PSTT2021_pixel(
                    out[i, j, :].copy(),
                    ncfobs_datacube[i, j, :].astype(float).copy(),
                )

    return out


# ---------------------------------------------------------------------------
# 11. na_interp_linear
# ---------------------------------------------------------------------------

def na_interp_linear(ts: np.ndarray) -> np.ndarray:
    """Fill NaN values with linear interpolation.

    If more than 90 % of the values are NaN, a zero-filled array of the same
    length is returned.  Edge NaNs are filled via forward/backward fill.

    Parameters
    ----------
    ts : np.ndarray
        1-D time series (may contain NaN).

    Returns
    -------
    np.ndarray
        Interpolated time series with no NaN values.
    """
    ts = np.asarray(ts, dtype=float).copy()

    if len(ts) == 0:
        return ts

    frac_nan = np.sum(np.isnan(ts)) / len(ts)
    if frac_nan > 0.90:
        return np.zeros(len(ts), dtype=float)

    if not np.any(np.isnan(ts)):
        return ts

    s = pd.Series(ts)
    s = s.interpolate(method="linear")
    s = s.ffill().bfill()
    return s.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 12. apply_mask
# ---------------------------------------------------------------------------

def apply_mask(
    datacube: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Multiply each time slice of *datacube* by *mask*.

    NaN values in the mask propagate into the datacube.

    Parameters
    ----------
    datacube : np.ndarray
        3-D datacube (rows, cols, time).
    mask : np.ndarray
        2-D mask (rows, cols) with 1.0 for valid and NaN for masked.

    Returns
    -------
    np.ndarray
        Masked datacube (a copy).
    """
    out = datacube.astype(float, copy=True)
    # Broadcasting: mask[:, :, np.newaxis] has shape (rows, cols, 1)
    return out * mask[:, :, np.newaxis]


# ---------------------------------------------------------------------------
# 13. clean_complete
# ---------------------------------------------------------------------------

def clean_complete(
    radiance_datacube: np.ndarray,
    ncfobs_datacube: np.ndarray,
    threshold: float = 0.4,
    noise: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full 9-step PSTT2021 data-cleaning pipeline.

    Steps:
        1. ``na_recode`` -- mark zero-observation months as NaN.
        2. ``replace_negative`` -- replace negative radiance with NaN.
        3. ``bgnoise_PSTT2021`` -- build background-noise mask.
        4. ``apply_mask`` -- apply noise mask.
        5. ``outlier_variance`` -- build stable-pixel mask.
        6. ``apply_mask`` -- apply stability mask.
        7. ``outlier_hampel`` -- Hampel-filter each pixel time series.
        8. ``bias_PSTT2021`` -- cloud bias correction.
        9. ``na_interp_linear`` -- interpolate remaining NaN values.

    Parameters
    ----------
    radiance_datacube : np.ndarray
        3-D radiance datacube (rows, cols, time).
    ncfobs_datacube : np.ndarray
        3-D cloud-free observations datacube, same shape.
    threshold : float, default 0.4
        Radiance threshold for background noise detection.
    noise : np.ndarray or None, optional
        Pre-computed noise mask (rows, cols).  If ``None``, one is computed
        from the data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(cleaned_datacube, combined_mask)`` where *combined_mask* is
        the product of the noise and stability masks.
    """
    nrows, ncols, ntimes = radiance_datacube.shape

    # Step 1: na_recode
    print("Step 1/9: na_recode")
    tmp = na_recode(radiance_datacube, ncfobs_datacube)

    # Step 2: replace_negative
    print("Step 2/9: replace_negative")
    tmp = replace_negative(tmp)

    # Step 3: bgnoise mask
    if noise is None:
        print("Step 3/9: bgnoise_PSTT2021")
        noise = bgnoise_PSTT2021(tmp, ncfobs_datacube, threshold)
    else:
        print("Step 3/9: using pre-computed noise mask")

    # Step 4: apply noise mask
    print("Step 4/9: apply noise mask")
    tmp = apply_mask(tmp, noise)

    # Step 5: outlier_variance mask
    mask = noise.copy()
    try:
        print("Step 5/9: outlier_variance")
        stable_pixels = outlier_variance(tmp, noise)
        # Step 6: apply stability mask
        print("Step 6/9: apply stability mask")
        tmp = apply_mask(tmp, stable_pixels)
        mask = noise * stable_pixels
    except Exception as e:
        warnings.warn(
            f"outlier_variance failed, using all pixels as stable: {e}"
        )
        print("Step 5-6/9: skipped (outlier_variance failed)")

    # Step 7: Hampel filter
    print("Step 7/9: outlier_hampel")
    for i in range(nrows):
        if i % 50 == 0:
            print(f"  outlier_hampel: processing row {i + 1}/{nrows}")
        for j in range(ncols):
            if np.isnan(mask[i, j]):
                continue
            tmp[i, j, :] = outlier_hampel(tmp[i, j, :])

    # Step 8: bias correction
    print("Step 8/9: bias_PSTT2021")
    tmp = bias_PSTT2021(tmp, ncfobs_datacube, mask)

    # Check if everything is NaN (or a single unique value)
    if len(np.unique(tmp[~np.isnan(tmp)])) <= 1 and np.all(np.isnan(tmp) | (tmp == 0)):
        print("  All values are missing; returning zeros.")
        return np.zeros_like(radiance_datacube, dtype=float), mask

    # Step 9: linear interpolation
    print("Step 9/9: na_interp_linear")
    for i in range(nrows):
        if i % 50 == 0:
            print(f"  na_interp_linear: processing row {i + 1}/{nrows}")
        for j in range(ncols):
            if np.isnan(mask[i, j]):
                continue
            tmp[i, j, :] = na_interp_linear(tmp[i, j, :])

    print("Cleaning complete.")
    return tmp, mask


# ---------------------------------------------------------------------------
# 14. centre_of_mass
# ---------------------------------------------------------------------------

def centre_of_mass(image: np.ndarray) -> Tuple[float, float]:
    """Compute the radiance-weighted centroid of a 2-D image.

    Parameters
    ----------
    image : np.ndarray
        2-D array of radiance values.  NaN and zero values are ignored.

    Returns
    -------
    tuple[float, float]
        ``(row_index, col_index)`` of the weighted centroid (may be
        fractional).
    """
    img = np.asarray(image, dtype=float).copy()
    img = np.nan_to_num(img, nan=0.0)

    total = np.sum(img)
    if total == 0:
        return (0.0, 0.0)

    rows, cols = img.shape
    row_idx, col_idx = np.meshgrid(
        np.arange(rows, dtype=float),
        np.arange(cols, dtype=float),
        indexing="ij",
    )

    x_com = float(np.sum(row_idx * img) / total)
    y_com = float(np.sum(col_idx * img) / total)

    return (x_com, y_com)


# ---------------------------------------------------------------------------
# 15. clean_datacube_from_files  (convenience)
# ---------------------------------------------------------------------------

def clean_datacube_from_files(
    radiance_files: List[str],
    cfobs_files: List[str],
    threshold: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read GeoTIFF files and run the full cleaning pipeline.

    Parameters
    ----------
    radiance_files : list[str]
        Sorted list of file paths to radiance GeoTIFFs (one per month).
    cfobs_files : list[str]
        Sorted list of file paths to cloud-free observation GeoTIFFs,
        same length and order as *radiance_files*.
    threshold : float, default 0.4
        Radiance threshold for background noise detection.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(cleaned_datacube, combined_mask)`` from :func:`clean_complete`.

    Raises
    ------
    ImportError
        If ``rasterio`` is not installed.
    ValueError
        If the file lists are empty or have different lengths.
    """
    try:
        import rasterio
    except ImportError as exc:
        raise ImportError(
            "rasterio is required to read GeoTIFF files. "
            "Install it with: pip install rasterio"
        ) from exc

    if len(radiance_files) != len(cfobs_files):
        raise ValueError(
            f"Mismatch: {len(radiance_files)} radiance files vs "
            f"{len(cfobs_files)} cloud-free observation files."
        )
    if len(radiance_files) == 0:
        raise ValueError("File lists must not be empty.")

    # Read first file to determine spatial dimensions
    with rasterio.open(radiance_files[0]) as src:
        nrows, ncols = src.height, src.width

    ntimes = len(radiance_files)
    radiance_cube = np.empty((nrows, ncols, ntimes), dtype=float)
    cfobs_cube = np.empty((nrows, ncols, ntimes), dtype=float)

    print(f"Reading {ntimes} radiance files ...")
    for t, fpath in enumerate(radiance_files):
        with rasterio.open(fpath) as src:
            radiance_cube[:, :, t] = src.read(1).astype(float)

    print(f"Reading {ntimes} cloud-free observation files ...")
    for t, fpath in enumerate(cfobs_files):
        with rasterio.open(fpath) as src:
            cfobs_cube[:, :, t] = src.read(1).astype(float)

    return clean_complete(radiance_cube, cfobs_cube, threshold=threshold)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"nighttime_lights v{__version__}")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Demonstrate with synthetic data
    # -------------------------------------------------------------------
    np.random.seed(42)
    nrows, ncols, ntimes = 10, 10, 36  # 3 years of monthly data

    # Synthetic radiance: base signal + noise + some negatives
    base = np.random.uniform(0.5, 20.0, size=(nrows, ncols, 1))
    noise_arr = np.random.normal(0, 1.0, size=(nrows, ncols, ntimes))
    radiance = base + noise_arr
    # Inject a few negatives
    radiance[0, 0, 0] = -5.0
    radiance[1, 1, 3] = -2.0

    # Synthetic cloud-free observations (higher = clearer)
    cfobs = np.random.randint(0, 28, size=(nrows, ncols, ntimes)).astype(float)
    # Inject zero-observation months
    cfobs[2, 2, 5] = 0
    cfobs[3, 3, 10] = 0

    print(f"Radiance datacube shape: {radiance.shape}")
    print(f"Cfobs datacube shape:    {cfobs.shape}")
    print()

    # --- Individual function demos ---
    print("1. replace_negative")
    r = replace_negative(radiance)
    print(f"   Negatives remaining: {np.sum(r < 0)}")
    print()

    print("2. na_recode")
    r = na_recode(radiance, cfobs)
    print(f"   NaN from zero-obs months: {np.sum(np.isnan(r))}")
    print()

    print("3. weighted_mean")
    wm = weighted_mean(np.array([1.0, 2.0, np.nan, 4.0]),
                        np.array([1.0, 2.0, 1.0, 3.0]))
    print(f"   weighted_mean([1, 2, NaN, 4], [1, 2, 1, 3]) = {wm:.4f}")
    print()

    print("4. bgnoise_PSTT2021")
    noise_mask = bgnoise_PSTT2021(r, cfobs, threshold=0.4)
    n_lit = int(np.nansum(noise_mask))
    print(f"   Lit pixels: {n_lit}/{nrows * ncols}")
    print()

    print("5. detrend_ts")
    ts_ex = np.arange(24, dtype=float) * 0.5 + np.random.normal(0, 0.3, 24)
    residuals = detrend_ts(ts_ex)
    print(f"   Mean of residuals: {np.mean(residuals):.6f} (should be ~0)")
    print()

    print("6. outlier_hampel")
    ts_h = np.array([1, 2, 1, 2, 100, 1, 2, 1, 2, 1, 2, 1], dtype=float)
    filtered = outlier_hampel(ts_h)
    print(f"   Before: {ts_h}")
    print(f"   After:  {filtered}")
    print()

    print("7. na_interp_linear")
    ts_i = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])
    interp = na_interp_linear(ts_i)
    print(f"   Before: {ts_i}")
    print(f"   After:  {interp}")
    print()

    print("8. centre_of_mass")
    img = np.zeros((5, 5))
    img[2, 3] = 10.0
    com = centre_of_mass(img)
    print(f"   Centre of mass of point source at (2,3): {com}")
    print()

    # --- Full pipeline ---
    print("=" * 60)
    print("Running full clean_complete pipeline on synthetic data ...")
    print("=" * 60)
    cleaned, mask = clean_complete(radiance, cfobs, threshold=0.4)
    print()
    print(f"Cleaned datacube shape: {cleaned.shape}")
    print(f"NaN remaining in cleaned: {np.sum(np.isnan(cleaned))}")
    print(f"Mask lit pixels: {int(np.nansum(mask))}")
    print()
    print("Done.")
