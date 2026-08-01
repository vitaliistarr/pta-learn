import pandas as pd
import pytest

from pta_learn.ti_workflow import (
    ti_workflow,
    validate_shutin_rate,
    validate_flowing_rate,
)

TRANSIENT_COLUMNS = [
    "start/hr", "end/hr", "duration/hr",
    "start/timestamp", "end/timestamp", "status",
]
BREAKPOINT_COLUMNS = ["Time", "Timestamp", "label"]
WRATE_COLUMNS = [
    "Start Timestamp", "End Timestamp",
    "Start Time", "End Time", "Weighted Averaged Rate",
]
TI_INTERVAL_THRESHOLD = 5



@pytest.fixture(scope="class", params=["ti", "c5"])
def dataset_pair(request, df_bhp, df_rate, df_bhp_c5, df_rate_c5):
    return {
        "ti": (df_bhp, df_rate),
        "c5": (df_bhp_c5, df_rate_c5),
    }[request.param]


class TestTiWorkflowRealDataIntegrity:
    """B1: Full pipeline run on ti_p.csv / ti_r.csv, verifying schema and
    cross-dataframe invariants that must hold for any valid decomposition
    of the pressure/rate history into shut-in, flowing, and breakpoint segments."""

    @pytest.fixture(scope="class")
    def workflow_result(self, dataset_pair):
        df_bhp, df_rate = dataset_pair
        return ti_workflow(
            df_bhp, df_rate, p=None,
            interval_shutin=TI_INTERVAL_THRESHOLD, interval_injection=TI_INTERVAL_THRESHOLD,
        )

    @pytest.fixture(scope="class")
    def unpacked(self, workflow_result):
        (
            shutin_filtered,
            flowing_filtered,
            shutin_bp_interval,
            TI_ft_filtered,
            all_breakpoints_filtered,
            w_rate,
            params,
        ) = workflow_result
        return {
            "shutin_filtered": shutin_filtered,
            "flowing_filtered": flowing_filtered,
            "shutin_bp_interval": shutin_bp_interval,
            "TI_ft_filtered": TI_ft_filtered,
            "all_breakpoints_filtered": all_breakpoints_filtered,
            "w_rate": w_rate,
            "params": params,
        }

    def test_returns_seven_element_tuple(self, workflow_result):
        assert len(workflow_result) == 7

    def test_transient_frames_have_expected_columns(self, unpacked):
        for key in ("shutin_filtered", "flowing_filtered"):
            assert list(unpacked[key].columns) == TRANSIENT_COLUMNS, (
                f"{key} columns mismatch: {list(unpacked[key].columns)}"
            )

    def test_all_breakpoints_have_expected_columns(self, unpacked):
        assert list(unpacked["all_breakpoints_filtered"].columns) == BREAKPOINT_COLUMNS

    def test_w_rate_has_expected_columns(self, unpacked):
        assert list(unpacked["w_rate"].columns) == WRATE_COLUMNS

    def test_breakpoint_length_invariant(self, unpacked):
        """Each consecutive pair of breakpoints defines exactly one weighted-rate
        interval, so len(w_rate) must be len(all_breakpoints) - 1."""
        w_rate = unpacked["w_rate"]
        all_bp = unpacked["all_breakpoints_filtered"]
        assert len(w_rate) == len(all_bp) - 1

    def test_all_transient_timestamps_appear_in_breakpoints(self, unpacked):
        """Every shut-in/flowing transient boundary must be represented as a
        breakpoint in all_breakpoints_filtered, otherwise the segmentation is inconsistent."""
        shutin = unpacked["shutin_filtered"]
        flowing = unpacked["flowing_filtered"]
        all_bp_timestamps = set(
            pd.to_datetime(unpacked["all_breakpoints_filtered"]["Timestamp"]).tolist()
        )

        transient_timestamps = []
        for frame in (shutin, flowing):
            if frame.empty:
                continue
            transient_timestamps += pd.to_datetime(frame["start/timestamp"]).tolist()
            transient_timestamps += pd.to_datetime(frame["end/timestamp"]).tolist()

        missing = [t for t in transient_timestamps if t not in all_bp_timestamps]
        assert not missing, f"Transient timestamps missing from breakpoints: {missing}"

    def test_transient_alignment_with_w_rate(self, unpacked):
        """For every transient, the weighted-rate row that begins at the transient's
        start/timestamp must end exactly at the transient's end/timestamp."""
        shutin = unpacked["shutin_filtered"]
        flowing = unpacked["flowing_filtered"]
        w_rate = unpacked["w_rate"]

        for frame in (shutin, flowing):
            if frame.empty:
                continue
            for _, row in frame.iterrows():
                start_ts = pd.Timestamp(row["start/timestamp"])
                end_ts = pd.Timestamp(row["end/timestamp"])
                matches = w_rate[
                    pd.to_datetime(w_rate["Start Timestamp"]) == start_ts
                ]
                assert not matches.empty, (
                    f"No w_rate row starts at transient start {start_ts}"
                )
                matched_end = pd.Timestamp(matches.iloc[0]["End Timestamp"])
                assert matched_end == end_ts, (
                    f"w_rate End Timestamp {matched_end} != transient end {end_ts}"
                )

    def test_breakpoints_time_strictly_increasing(self, unpacked):
        time_series = unpacked["all_breakpoints_filtered"]["Time"]
        assert time_series.is_monotonic_increasing
        assert time_series.is_unique

    def test_breakpoints_timestamp_strictly_increasing(self, unpacked):
        ts_series = pd.to_datetime(unpacked["all_breakpoints_filtered"]["Timestamp"])
        assert ts_series.is_monotonic_increasing
        assert ts_series.is_unique


class TestRateValidationReclassification:
    """B2: validate_shutin_rate() and validate_flowing_rate() must reclassify
    transients whose measured rate contradicts their structural label."""

    def _make_transient(self, start_ts, end_ts, status):
        return pd.DataFrame({
            "start/hr": [0.0],
            "end/hr": [10.0],
            "duration/hr": [10.0],
            "start/timestamp": [start_ts],
            "end/timestamp": [end_ts],
            "status": [status],
        })

    def test_shutin_reclassified_to_flowing_when_rate_nonzero(self):
        """If the weighted average rate during a nominally 'shut-in' transient is
        non-zero (e.g., a mis-detected low-rate flow period), validate_shutin_rate
        must move it into flowing_filtered with status 'flowing'."""
        t1 = pd.Timestamp("2026-01-01 00:00:00")
        t1_end = pd.Timestamp("2026-01-01 10:00:00")

        shutin = self._make_transient(t1, t1_end, "shutin")
        flowing = pd.DataFrame(columns=TRANSIENT_COLUMNS)
        w_rate = pd.DataFrame({
            "Start Timestamp": [t1],
            "End Timestamp": [t1_end],
            "Start Time": [0.0],
            "End Time": [10.0],
            "Weighted Averaged Rate": [250.0],
        })

        shutin_filtered, flowing_filtered = validate_shutin_rate(shutin, w_rate, flowing)

        assert shutin_filtered.empty
        assert len(flowing_filtered) == 1
        assert flowing_filtered.iloc[0]["status"] == "flowing"
        assert flowing_filtered.iloc[0]["start/timestamp"] == t1

    def test_flowing_reclassified_to_shutin_when_rate_zero(self):
        """If the weighted average rate during a nominally 'flowing' transient is
        zero, validate_flowing_rate must move it into shutin_filtered with status 'shutin'."""
        t2 = pd.Timestamp("2026-01-02 00:00:00")
        t2_end = pd.Timestamp("2026-01-02 08:00:00")

        flowing = self._make_transient(t2, t2_end, "flowing")
        shutin = pd.DataFrame(columns=TRANSIENT_COLUMNS)
        w_rate = pd.DataFrame({
            "Start Timestamp": [t2],
            "End Timestamp": [t2_end],
            "Start Time": [0.0],
            "End Time": [8.0],
            "Weighted Averaged Rate": [0.0],
        })

        shutin_filtered, flowing_filtered = validate_flowing_rate(shutin, w_rate, flowing)

        assert flowing_filtered.empty
        assert len(shutin_filtered) == 1
        assert shutin_filtered.iloc[0]["status"] == "shutin"
        assert shutin_filtered.iloc[0]["start/timestamp"] == t2

    def test_validate_shutin_rate_raises_on_none_inputs(self):
        with pytest.raises(ValueError):
            validate_shutin_rate(None, pd.DataFrame(), pd.DataFrame())

    def test_validate_flowing_rate_raises_on_none_inputs(self):
        with pytest.raises(ValueError):
            validate_flowing_rate(pd.DataFrame(), pd.DataFrame(), None)

    def test_validate_shutin_rate_missing_column_raises_keyerror(self):
        """Both shutin and w_rate must contain their documented required columns."""
        bad_shutin = pd.DataFrame({"not_start_timestamp": [1]})
        w_rate = pd.DataFrame({
            "Start Timestamp": [pd.Timestamp("2026-01-01")],
            "Weighted Averaged Rate": [100.0],
        })
        with pytest.raises(KeyError):
            validate_shutin_rate(bad_shutin, w_rate, pd.DataFrame())