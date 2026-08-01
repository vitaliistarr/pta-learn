import pytest
import pandas as pd
from pta_learn import PTAClassifier

EXPECTED_OUTPUT_COLUMNS = [
    "Time", "Pressure", "Pressure_derivative",
    "Log_time", "Log_pressure", "Log_pressure_derivative",
    "Regime",
]
EXPECTED_DISTANCE_COLUMNS = [
    "Zero slope", "Linear-up 1", "Linear-up 2",
    "Linear-down 1", "Linear-down 2",
]
VALID_REGIME_VALUES = {0, 1, 2, 3, 4}  # Radial, Linear-up, Linear-down, Boundary, WBS


TRANSIENT_CASES = [
    ("data/Stable_Pattern_loglog_transient1.csv", pd.DataFrame, 3),
    ("data/Stable_Pattern_loglog_transient2.csv", pd.DataFrame, 4),
    ("data/Stable_Pattern_loglog_transient3.csv", pd.DataFrame, 4),
    ("data/Changing_Pattern_loglog_transient1.csv", pd.DataFrame, 2),
    ("data/Changing_Pattern_loglog_transient2.csv", pd.DataFrame, 2),
    ("data/Changing_Pattern_loglog_transient3.csv", pd.DataFrame, 4),
]


@pytest.mark.parametrize("transient, expected_type, expected_features", TRANSIENT_CASES)
def test_PTAClassifier_predict_optimize(transient, expected_type, expected_features):
    """Original smoke test: confirms predict_optimize runs end-to-end and
    detects the expected number of distinct flow regimes, including Radial."""
    data = pd.read_csv(transient)
    clf = PTAClassifier()
    clf.fit(data.values)
    result, dist, _ = clf.predict_optimize(
        max_window_length=1, min_filter_value=-0.5, max_filter_value=0.5
    )
    assert isinstance(result, expected_type), f"Result should be {expected_type}."
    assert isinstance(dist, expected_type), f"Dist output should be {expected_type}."
    assert result.Regime.nunique() == expected_features, (
        f"Expected {expected_features} distinct regime features for transient."
    )
    assert 0 in result.Regime.unique(), "Radial flow feature should be found in transient."


@pytest.mark.parametrize("transient, expected_type, expected_features", TRANSIENT_CASES)
def test_output_data_schema(transient, expected_type, expected_features):
    """Validates that result (self.output_data) always exposes exactly the
    seven documented columns, in the exact order produced by compile_output(),
    since it's built via np.concatenate rather than a labeled dict."""
    data = pd.read_csv(transient)
    clf = PTAClassifier()
    clf.fit(data.values)
    result, _, _ = clf.predict_optimize(
        max_window_length=1, min_filter_value=-0.5, max_filter_value=0.5
    )

    assert list(result.columns) == EXPECTED_OUTPUT_COLUMNS, (
        f"output_data columns mismatch for {transient}: "
        f"got {list(result.columns)}, expected {EXPECTED_OUTPUT_COLUMNS}"
    )
    assert not result.empty, f"output_data should not be empty for {transient}"

    # All columns except Regime are continuous measurements; Regime is a
    # discrete label. Both must be free of NaN for downstream plotting/export
    # to work correctly.
    assert not result.isnull().any().any(), (
        f"output_data contains NaN values for {transient}:\n{result.isnull().sum()}"
    )

    for col in ["Time", "Pressure", "Pressure_derivative",
                "Log_time", "Log_pressure", "Log_pressure_derivative"]:
        assert pd.api.types.is_numeric_dtype(result[col]), (
            f"Column {col} should be numeric, got dtype {result[col].dtype}"
        )