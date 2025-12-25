import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.stats import mannwhitneyu, iqr


def extract_pressure_features(start_ts, end_ts, pressure, window=12):
    """
        Extract pressure stability features from a transient interval for automated identification.

        This function computes three stability metrics that characterize pressure behavior during
        a transient period: monotonicity ratio, local median deviation, and trend efficiency.
        These features enable quantitative assessment of pressure time-series reliability for automated
        pressure transient analysis (PTA) applications.

        Parameters
        ----------
        start_ts : pd.Timestamp or datetime-like
            Start timestamp of the pressure transient interval.
        end_ts : pd.Timestamp or datetime-like
            End timestamp of the pressure transient interval.
        pressure : pd.Series
            Time-indexed pressure measurements. 5 minutes resolution is recommended.
        window : int, optional
            Rolling window size (in data points) for local median calculation.
            Default is 12, which is 1 hour window for 5 minutes resampled data.

        Returns
        -------
        dict
            Dictionary containing three stability features:
            - 'monotonicity_ratio' : float
                Measure of pressure trend consistency, ranging from 0 to 1. Higher values
                indicate more monotonic behavior with fewer directional changes.
            - 'local_median_deviation' : float
                Normalized deviation from rolling median, scaled using hyperbolic tangent
                transformation. Values near 1 indicate low deviation (high stability).
            - 'trend_efficiency' : float
                Ratio of net pressure change to total path traveled, ranging from 0 to 1.
                Higher values indicate more direct (efficient) pressure trends.
    """
    pressure_transient = pressure[(pressure.index >= start_ts) & (pressure.index < end_ts)]

    # Handle insufficient data
    if len(pressure_transient) < 2:
        return {
            'monotonicity_ratio': np.nan,
            'local_median_deviation': np.nan,
            'trend_efficiency': np.nan,
        }

    pressure_transient_arr = pressure_transient.values.ravel()

    # Monotonicity ratio
    diffs = np.diff(pressure_transient_arr)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    max_changes = max(len(diffs) - 1, 1)
    monotonicity_ratio = float(1 - (sign_changes / max_changes))

    # Local Median Deviation
    window_size = min(window, len(pressure_transient_arr))
    rolling_pressure_mean = pressure_transient.rolling(
        window=window_size,
        center=False,
        min_periods=1
    ).mean()

    residuals = np.abs(pressure_transient - rolling_pressure_mean)
    mad_residuals = float(np.median(np.abs(residuals - np.median(residuals))))
    mad_residuals_scaled = max(mad_residuals * 1.4826, 1e-6)

    z = residuals / mad_residuals_scaled
    R = np.mean(z)

    sig_score = 1.0 / (1.0 + np.exp(0.2 * (R - 50)))
    local_deviation = float(sig_score)

    # Trend Efficiency
    net_change = float(np.abs(pressure_transient_arr[-1] - pressure_transient_arr[0]))
    total_path_travelled = float(np.sum(np.abs(diffs)))
    trend_efficiency = 1.0 if total_path_travelled == 0 else float(net_change / total_path_travelled)

    return {
        'monotonicity_ratio': monotonicity_ratio,
        'local_deviation': local_deviation,
        'trend_efficiency': trend_efficiency,
    }


def extract_rate_features(start_ts, end_ts, duration_hr, rate):
    """
        Extract rate variability features for transient stability assessment.

        This function computes statistical measures of rate variability during and immediately
        preceding a pressure transient interval. These features characterize the stability of
        flow conditions, which is essential for reliable pressure transient analysis.

        Parameters
        ----------
        start_ts : pd.Timestamp or datetime-like
            Start timestamp of the transient interval.
        end_ts : pd.Timestamp or datetime-like
            End timestamp of the transient interval.
        duration_hr : float
            Duration of the transient in hours, used to define the preceding time window.
        rate : pd.Series
            Time-indexed flow rate measurements (injection or production rate).
            1 hour resolution is recommended.

        Returns
        -------
        dict
            Dictionary containing three rate variability metrics:
            - 'rate_std' : float
                Standard deviation of rate during the transient interval.
            - 'rate_td_std' : float
                Standard deviation of rate during the preceding time window of equal duration.
            - 'rate_iqr' : float
                Interquartile range (90th - 10th percentile) of rate during the transient,
                providing robust measure of rate spread.
    """
    td_time = start_ts - timedelta(hours=duration_hr)

    # Extract relevant rate time windows
    rate_transient = rate[(rate.index >= start_ts) & (rate.index < end_ts)]
    rate_td_time = rate[(rate.index >= td_time) & (rate.index < start_ts)]

    # Variability statistics
    rate_std = float(rate_transient.std())
    rate_td_std = float(rate_td_time.std())
    rate_iqr = float(rate_transient.quantile(0.9)) - float(rate_transient.quantile(0.1))

    return {
        'rate_std': rate_std,
        'rate_td_std': rate_td_std,
        'rate_iqr': rate_iqr,
    }


def extract_stat_test_results(start_ts, end_ts, duration_hr, rate, window_size_hr=30):
    """
        Perform statistical hypothesis testing to quantify rate change significance at breakpoints.

        This function applies the Mann-Whitney U test and computes signal-to-noise ratio (SNR)
        to assess the statistical significance of rate changes at transient boundaries. These
        metrics support automated identification of true breakpoints in pressure data.

        Parameters
        ----------
        start_ts : pd.Timestamp or datetime-like
            Start timestamp of the transient interval (potential breakpoint).
        end_ts : pd.Timestamp or datetime-like
            End timestamp of the transient interval.
        duration_hr : float
            Duration of the transient in hours.
        rate : pd.Series
            Time-indexed flow rate measurements.
        window_size_hr : float, optional
            Size of time windows (hours) before and after start_ts for point-wise comparison.
            Default is 30 hours.

        Returns
        -------
        dict
            Dictionary containing four statistical metrics:
            - 'rate_snr_30pt' : float
                Signal-to-noise ratio computed using window_size_hr windows around start_ts.
            - 'p_mannwhitneyu' : float
                P-value from two-sided Mann-Whitney U test for window-based comparison.
            - 'rate_snr_full' : float
                Signal-to-noise ratio computed using full transient durations.
            - 'p_mannwhitneyu_full' : float
                P-value from two-sided Mann-Whitney U test for full transient comparison.

            Returns np.nan for metrics when insufficient data points are available.
    """
    window_start_ts = start_ts - timedelta(hours=window_size_hr)
    window_end_ts = start_ts + timedelta(hours=window_size_hr)
    rate_before = rate[
        (rate.index >= window_start_ts) & (rate.index < start_ts)]
    rate_after = rate[(rate.index >= start_ts) & (rate.index < window_end_ts)]
    if len(rate_before) > 0 and len(rate_after) > 0:
        # SNR
        mu_before, mu_after = float(rate_before.mean()), float(rate_after.mean())
        std_before, std_after = float(rate_before.std()), float(rate_after.std())
        snr = abs(mu_after - mu_before) / max(std_before, std_after, 1e-6)
        # Stat test
        _, p_u = mannwhitneyu(rate_before, rate_after, alternative='two-sided')
    else:
        snr = np.nan
        p_u = np.nan

    # Previous transient start/end
    prev_start_ts = start_ts - timedelta(hours=duration_hr)
    prev_end_ts = start_ts

    # Full transient duration rate
    rate_before_full = rate[(rate.index >= prev_start_ts) &
                            (rate.index < prev_end_ts)]
    rate_after_full = rate[(rate.index >= start_ts) &
                           (rate.index < end_ts)]

    # Get stats if data is available
    if len(rate_before_full) > 2 and len(rate_after_full) > 2:
        # SNR
        mu_before_full = float(rate_before_full.mean())
        mu_after_full = float(rate_after_full.mean())
        std_before_full = float(rate_before_full.std())
        std_after_full = float(rate_after_full.std())
        snr_full = abs(mu_after_full - mu_before_full) / max(std_before_full, std_after_full, 1e-6)
        # Stat test
        _, p_u_full = mannwhitneyu(rate_before_full, rate_after_full, alternative='two-sided')
    else:
        snr_full = np.nan
        p_u_full = np.nan

    return {
        'rate_snr_30pt': float(snr),
        'p_mannwhitneyu_30pt': float(p_u),
        'rate_snr_full': float(snr_full),
        'p_mannwhitneyu_full': float(p_u_full)
    }


def transform_p_value(p):
    """
        Transform p-value to a score in [0, 1] range where higher values indicate greater significance.

        Parameters
        ----------
        p : float
            P-value from statistical hypothesis test.

        Returns
        -------
        float
            Transformed score: (1 - p) if p is valid, 0 if p is NaN (penalizing missing values).
    """
    if np.isnan(p):
        return 0  # penalize missing
    return 1 - p


def transform_snr(snr, k=2):
    """
        Transform signal-to-noise ratio using sigmoid function to bounded [0, 1] score.

        Parameters
        ----------
        snr : float
            Signal-to-noise ratio value.
        k : float, optional
            Steepness parameter controlling sigmoid transition. Default is 2.

        Returns
        -------
        float
            Transformed score in range [0, 1], where higher values indicate stronger signal.
    """
    if np.isnan(snr):
        return 0.5
    return 1 / (1 + np.exp(-k * (snr - 1)))


def convert_rate_change_metrics_to_score(stat_test_results, weights=None):
    """
        Combine multiple statistical metrics into a single rate change significance score.

        This function aggregates p-values and SNR metrics from statistical tests into a unified
        score that quantifies the overall significance of rate changes at transient breakpoints,
        supporting automated transient identification workflows.

        Parameters
        ----------
        stat_test_results : dict
            Dictionary containing statistical metrics with keys including 'p_' for p-values
            and 'snr' for signal-to-noise ratios.
        weights : array-like, optional
            Weights for averaging transformed scores. If None, equal weights are applied.

        Returns
        -------
        float
            Weighted average of transformed statistical metrics, ranging from 0 to 1.
            Higher values indicate more significant rate changes.
    """
    scores = []
    for key, value in stat_test_results.items():
        if 'p_' in key:
            scores.append(transform_p_value(value))
        elif 'snr' in key:
            scores.append(transform_snr(value))
    if weights is None:
        weights = np.ones(len(scores))
    return np.average(scores, weights=weights)


def normalize_feature_value(value, percentiles_dict, percentiles_dict_col):
    """
        Normalize a single stability feature value to [0, 1] using precomputed percentile thresholds.

        This function applies piecewise linear normalization where values indicating high stability
        (below low percentile) map to 1, unstable values (above high percentile) map to 0, and
        intermediate values are linearly interpolated.

        Parameters
        ----------
        value : float
            Raw feature value to normalize.
        percentiles_dict : dict
            Dictionary mapping column names to dictionaries with 'low' and 'high' percentile values.
        percentiles_dict_col : str
            Feature name/column identifier used to retrieve percentile thresholds.

        Returns
        -------
        float
            Normalized stability score in [0, 1], where 1 indicates high stability.
    """
    low = percentiles_dict[percentiles_dict_col]['low']
    high = percentiles_dict[percentiles_dict_col]['high']

    if value < low:
        return 1.0
    elif value > high:
        return 0.0
    else:
        return 1 - (value - low) / (high - low)


def normalize_feature_series(series, percentiles_dict, percentiles_dict_col):
    """
        Normalize a pandas Series of stability features using precomputed percentile thresholds.

        This vectorized function applies the same normalization scheme as
        normalize_feature_value() to all elements of a series efficiently.

        Parameters
        ----------
        series : pd.Series or np.ndarray
            Array of raw feature values to normalize.
        percentiles_dict : dict
            Dictionary mapping column names to dictionaries with 'low' and 'high' percentile values.
        percentiles_dict_col : str
            Feature name/column identifier used to retrieve percentile thresholds.

        Returns
        -------
        np.ndarray
            Array of normalized stability scores in [0, 1], where 1 indicates high stability.
    """
    low = percentiles_dict[percentiles_dict_col]['low']
    high = percentiles_dict[percentiles_dict_col]['high']

    # Clip values to the percentile range
    clipped = np.clip(series, low, high)

    # Linear scaling between low and high
    normalized = 1 - (clipped - low) / (high - low)

    # Values below low get 1, above high get 0
    normalized[series < low] = 1
    normalized[series > high] = 0

    return normalized


def compute_row_features(row, rate_resampled, pressure_resampled):
    """
        Compute comprehensive feature set for a single transient interval.

        This function serves as a wrapper that orchestrates extraction of all stability and
        significance features for a given pressure transient, integrating pressure-based stability
        metrics, rate variability features, and statistical significance measures [file:1][file:2].

        Parameters
        ----------
        row : pd.Series
            Row from transient catalog containing temporal boundaries with keys:
            - 'start/timestamp' : pd.Timestamp or datetime-like, transient start time
            - 'end/timestamp' : pd.Timestamp or datetime-like, transient end time
            - 'duration/hr' : float, transient duration in hours
        rate_resampled : pd.Series
            Time-indexed, resampled flow rate measurements. 1 hour resolution is recommended.
        pressure_resampled : pd.Series
            Time-indexed, resampled pressure measurements. 5 minutes resolution is recommended.

        Returns
        -------
        dict
            Dictionary containing all computed features:
            - Pressure stability features from extract_pressure_features().
            - Rate variability features from extract_rate_features().
            - Statistical significance features from extract_stat_test_results().
    """
    start_ts = row['start/timestamp']
    end_ts = row['end/timestamp']
    duration_hr = row['duration/hr']

    # Pressure stability features
    pressure_features = extract_pressure_features(
        start_ts=start_ts, end_ts=end_ts,
        pressure=pressure_resampled
    )

    # Rate variability features
    rate_features = extract_rate_features(
        start_ts=start_ts, end_ts=end_ts, duration_hr=duration_hr,
        rate=rate_resampled
    )

    # Rate change significance
    stat_test_results = extract_stat_test_results(
        start_ts=start_ts, end_ts=end_ts, duration_hr=duration_hr,
        rate=rate_resampled
    )

    # Combine all into a single dictionary
    return {
        **pressure_features,
        **rate_features,
        **stat_test_results
    }


def compute_features(transients, rate_resampled, pressure_resampled, merge=True):
    """
        Compute a comprehensive feature set for all transient intervals.

        Extracts pressure stability metrics, rate variability measures, and statistical
        significance tests for each transient in the catalog.

        Parameters
        ----------
        transients : pd.DataFrame
            Transient catalog with columns: 'start/timestamp', 'end/timestamp', 'duration/hr'.
        rate_resampled : pd.Series
            Time-indexed flow rate measurements. 1 hour resolution recommended.
        pressure_resampled : pd.Series
            Time-indexed pressure measurements. 5 minute resolution recommended.
        merge : bool, optional
            If True, returns copy of transients with features appended as new columns.
            If False, returns only the features as a new DataFrame. Default is True.

        Returns
        -------
        features_df : pd.DataFrame
            If merge=True: Copy of input transients with additional feature columns.
            If merge=False: DataFrame with feature columns only.
            Features for each transient include:
            - Pressure: monotonicity_ratio, local_median_deviation, trend_efficiency
            - Rate: rate_std, rate_td_std, rate_iqr
            - Statistical: rate_snr_30pt, p_mannwhitneyu_30pt, rate_snr_full, p_mannwhitneyu_full
    """
    features = transients.apply(
        lambda row: compute_row_features(row, rate_resampled, pressure_resampled),
        axis=1
    )
    features_df = pd.DataFrame(features.tolist(), index=transients.index)
    if merge:
        features_df = pd.concat([transients, features_df], axis=1)

    return features_df


def compute_scores(transients, rate_resampled, presssure_resampled, merge=True):
    """
        Compute normalized quality scores for all transient intervals.

        Extracts raw features, applies percentile-based normalization to rate features,
        and aggregates into three composite scores quantifying transient quality for
        pressure transient analysis.

        Parameters
        ----------
        transients : pd.DataFrame
            Transient catalog with columns: 'start/timestamp', 'end/timestamp', 'duration/hr'.
        rate_resampled : pd.Series
            Time-indexed flow rate measurements. 1 hour resolution recommended.
        pressure_resampled : pd.Series
            Time-indexed pressure measurements. 5 minute resolution recommended.
        merge : bool, optional
            If True, returns copy of transients with scores appended as new columns.
            If False, returns only the scores as a new DataFrame. Default is True.

        Returns
        -------
        scores : pd.DataFrame
            If merge=True: Copy of input transients with three additional score columns.
            If merge=False: DataFrame with only three score columns.
            Score columns (0-1 range):
            - 'pressure_stability_score': Mean of pressure stability features
            - 'rate_stability_score': Mean of normalized rate variability features
            - 'rate_change_significance_score': Combined p-values and SNR metrics
        rate_feature_percentiles : dict
            Percentile thresholds (5th and 95th) for each rate feature, used for normalization.
    """
    features_df = compute_features(transients, rate_resampled, presssure_resampled)

    if len(transients) > 0:
        sample_row = transients.iloc[0]
        sample_pressure_features = extract_pressure_features(
            start_ts=sample_row['start/timestamp'],
            end_ts=sample_row['end/timestamp'],
            pressure=presssure_resampled
        )
        sample_rate_features = extract_rate_features(
            start_ts=sample_row['start/timestamp'],
            end_ts=sample_row['end/timestamp'],
            duration_hr=sample_row['duration/hr'],
            rate=rate_resampled
        )
        sample_rate_change_features = extract_stat_test_results(
            start_ts=sample_row['start/timestamp'], end_ts=sample_row['end/timestamp'],
            duration_hr=sample_row['duration/hr'],
            rate=rate_resampled
        )
        pressure_feature_cols = list(sample_pressure_features.keys())
        rate_feature_cols = list(sample_rate_features.keys())
        rate_change_feature_cols = list(sample_rate_change_features.keys())
    else:
        pressure_feature_cols = []
        rate_feature_cols = []
        rate_change_feature_cols = []

    rate_feature_percentiles = {}
    for col in rate_feature_cols:
        low = features_df[col].quantile(0.05)
        high = features_df[col].quantile(0.95)
        rate_feature_percentiles[col] = {'low': low, 'high': high}
        features_df[col] = normalize_feature_series(series=features_df[col], percentiles_dict=rate_feature_percentiles,
                                                    percentiles_dict_col=col)

    if merge:
        scores = transients.copy()
        scores['pressure_stability_score'] = features_df[pressure_feature_cols].mean(axis=1)
        scores['rate_stability_score'] = features_df[rate_feature_cols].mean(axis=1)
        scores['rate_change_significance_score'] = features_df[rate_change_feature_cols].apply(
            lambda row: convert_rate_change_metrics_to_score(row.to_dict()), axis=1)
    else:
        scores = pd.DataFrame()
        scores['pressure_stability_score'] = features_df[pressure_feature_cols].mean(axis=1)
        scores['rate_stability_score'] = features_df[rate_feature_cols].mean(axis=1)
        scores['rate_change_significance_score'] = features_df[rate_change_feature_cols].apply(
            lambda row: convert_rate_change_metrics_to_score(row.to_dict()), axis=1)

    return scores, rate_feature_percentiles