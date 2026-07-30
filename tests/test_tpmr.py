import numpy as np
import pandas as pd
import pytest

from pta_learn.tpmr import TPMR

EXPECTED_TRANSIENT_COLUMNS = [
    "start/hr", "end/hr", "duration/hr",
    "start/timestamp", "end/timestamp", "status",
]
TPMR_INTERVAL_THRESHOLD = 5


@pytest.fixture(scope="class", params=["ti", "c5"])
def dataset_bhp(request, df_bhp, df_bhp_c5):
    return {"ti": df_bhp, "c5": df_bhp_c5}[request.param]

class TestTPMRSchemaOnRealData:
    """A1: Validate output schema and internal consistency"""

    @pytest.fixture(scope="class")
    def tpmr_result(self, dataset_bhp):
        return TPMR(dataset_bhp, p=None, interval_shutin=TPMR_INTERVAL_THRESHOLD)

    def test_returns_four_dataframes(self, tpmr_result):
        assert len(tpmr_result) == 4
        assert all(isinstance(f, pd.DataFrame) for f in tpmr_result)

    def test_transient_frames_have_expected_columns(self, tpmr_result):
        """shutin_transient_interval (idx 3)
        must expose exactly the six documented columns."""
        _, _, _, shutin_transient_interval = tpmr_result
        assert list(shutin_transient_interval.columns) == EXPECTED_TRANSIENT_COLUMNS

    def test_duration_equals_end_minus_start(self, tpmr_result):
        """Invariant: duration/hr == end/hr - start/hr for every detected transient."""
        _, _, _, frame = tpmr_result
        if frame.empty:
            pytest.skip("No transients detected in this configuration; invariant vacuously holds")
        expected_duration = frame["end/hr"] - frame["start/hr"]
        assert np.allclose(frame["duration/hr"].values, expected_duration.values, atol=1e-9)

    def test_status_column_is_always_shutin(self, tpmr_result):
        """TPMR only ever labels transients as 'shutin'."""
        _, _, _, shutin_transient_interval = tpmr_result
        if shutin_transient_interval.empty:
            pytest.skip("No transients detected; status invariant vacuously holds")
        assert (shutin_transient_interval["status"] == "shutin").all()

    def test_interval_frame_is_subset_of_all_frame_by_duration(self, tpmr_result):
        """Every row in shutin_transient_interval must satisfy the interval_shutin threshold
        used when calling TPMR (5 hours)"""
        _, _, _, shutin_transient_interval = tpmr_result
        if shutin_transient_interval.empty:
            pytest.skip("No transients met the interval threshold")
        assert (shutin_transient_interval["duration/hr"] >= TPMR_INTERVAL_THRESHOLD).all()

    def test_start_before_end_for_all_transients(self, tpmr_result):
        """Sanity check: start/timestamp must always precede end/timestamp."""
        _, _, _, shutin_transient_interval = tpmr_result
        if shutin_transient_interval.empty:
            pytest.skip("No transients detected")
        assert (
            pd.to_datetime(shutin_transient_interval["start/timestamp"])
            < pd.to_datetime(shutin_transient_interval["end/timestamp"])
        ).all()