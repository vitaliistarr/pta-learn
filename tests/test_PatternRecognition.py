import pytest
import pandas as pd

from pta_learn import PatternRecognition

EXPECTED_INTERVAL_INDEX = ["start", "end", "confidence"]
EXPECTED_INTERVAL_COLUMNS = ["Radial", "Linear-up", "Linear-down"]

PATTERN_CASES = [
    ("Stable_Pattern", [1, 2, 3]),
    ("Changing_Pattern", [1, 2, 3]),
]


def _load_case(case, ids):
    data = []
    for id in ids:
        df = pd.read_csv(f"data/{case}_loglog_transient{id}.csv")
        data.append(df)
    return data


class TestPatternRecognitionStablePattern:
    """Runs PatternRecognition.detect_features() exactly once per case
    (StablePattern, ChangingPattern) via a class-scoped fixture, since it
    performs Bayesian optimization per transient and is expensive to repeat.
    Individual assertions are split into separate test methods for clearer
    failure reporting, without re-running the pipeline for each one.
    """

    @pytest.fixture(
        scope="class",
        params=PATTERN_CASES,
        ids=[case for case, _ in PATTERN_CASES],
    )
    def fitted_pr(self, request):
        case, ids = request.param
        data = _load_case(case, ids)
        PR = PatternRecognition()
        PR.fit(data)
        PR.detect_features()
        PR.get_stable_pattern()
        return case, PR

    def test_pattern_recognized(self, fitted_pr):
        case, PR = fitted_pr
        assert PR.pattern_recognized, f"Pattern was not found for {case} case."

    def test_radial_pattern_has_values(self, fitted_pr):
        case, PR = fitted_pr
        assert not PR.stable_pattern_intervals.Radial.isnull().any(), (
            f"No Radial pattern feature found for {case} case. "
            f"No values for start/end or confidence."
        )

    def test_stable_pattern_intervals_structural_schema(self, fitted_pr):
        """Must hold regardless of detection outcome: index/columns/shape."""
        case, PR = fitted_pr
        intervals = PR.stable_pattern_intervals

        assert list(intervals.index) == EXPECTED_INTERVAL_INDEX, (
            f"stable_pattern_intervals index mismatch for {case}: "
            f"got {list(intervals.index)}, expected {EXPECTED_INTERVAL_INDEX}"
        )
        assert list(intervals.columns) == EXPECTED_INTERVAL_COLUMNS, (
            f"stable_pattern_intervals columns mismatch for {case}: "
            f"got {list(intervals.columns)}, expected {EXPECTED_INTERVAL_COLUMNS}"
        )
        assert intervals.shape == (3, 3), (
            f"stable_pattern_intervals should be a 3x3 table, got shape "
            f"{intervals.shape} for {case}"
        )

    def test_at_least_one_regime_populated(self, fitted_pr):
        """If pattern_recognized is True, at least one regime column must
        actually contain values -- otherwise the flag is inconsistent with
        the data."""
        case, PR = fitted_pr
        if not PR.pattern_recognized:
            pytest.skip(f"No pattern recognized for {case}; nothing to check")

        intervals = PR.stable_pattern_intervals
        populated_columns = [
            col for col in EXPECTED_INTERVAL_COLUMNS
            if not intervals[col].isnull().any()
        ]
        assert populated_columns, (
            f"pattern_recognized is True for {case} but no regime column "
            f"in stable_pattern_intervals was populated."
        )

    def test_populated_intervals_have_valid_ranges(self, fitted_pr):
        """start < end and confidence in [0, 1], checked only for columns
        that were actually detected -- not every regime is guaranteed to be
        found in every case."""
        case, PR = fitted_pr
        if not PR.pattern_recognized:
            pytest.skip(f"No pattern recognized for {case}; nothing to check")

        intervals = PR.stable_pattern_intervals
        populated_columns = [
            col for col in EXPECTED_INTERVAL_COLUMNS
            if not intervals[col].isnull().any()
        ]

        for col in populated_columns:
            start = intervals.loc["start", col]
            end = intervals.loc["end", col]
            confidence = intervals.loc["confidence", col]

            assert start < end, (
                f"stable_pattern_intervals['{col}']: start ({start}) must be "
                f"< end ({end}) for {case}"
            )
            assert 0.0 <= confidence <= 1.0, (
                f"stable_pattern_intervals['{col}']: confidence ({confidence}) "
                f"must be in [0, 1] for {case}"
            )

    def test_confidence_dict_matches_populated_columns(self, fitted_pr):
        """PR.confidence is populated alongside stable_pattern_intervals
        inside get_interval() -- they should never drift apart."""
        case, PR = fitted_pr
        if not PR.pattern_recognized:
            pytest.skip(f"No pattern recognized for {case}; nothing to check")

        intervals = PR.stable_pattern_intervals
        populated_columns = [
            col for col in EXPECTED_INTERVAL_COLUMNS
            if not intervals[col].isnull().any()
        ]
        for col in populated_columns:
            assert col in PR.confidence, (
                f"Column {col} was populated in stable_pattern_intervals but "
                f"missing from PR.confidence for {case}"
            )