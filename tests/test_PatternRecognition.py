import pytest
import pandas as pd
from pta_learn import PatternRecognition

@pytest.mark.parametrize('case, ids', [
    ('Stable_Pattern', [1,2,3]),
    ('Changing_Pattern', [1,2,3])
])
def test_PatternRecognition_get_stable_pattern(case, ids):

    data = []
    for id in ids:
        df = pd.read_csv(f'data/{case}_loglog_transient{id}.csv')
        data.append(df)

    PR = PatternRecognition()
    PR.fit(data)
    PR.detect_features()
    _ = PR.get_stable_pattern()

    assert PR.pattern_recognized, f'Pattern was not found for {case} case.'
    assert ~PR.stable_pattern_intervals.Radial.isnull().any(), (f'No Radial pattern feature pattern found for {case} case.'
                                                                f' No values for start/end or confidence.')