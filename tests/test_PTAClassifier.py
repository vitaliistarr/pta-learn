import pytest
import pandas as pd
from pta_learn import PTAClassifier


@pytest.mark.parametrize('transient, expected_type, expected_features', [
    ('data/Stable_Pattern_loglog_transient1.csv', pd.DataFrame, 3),
    ('data/Stable_Pattern_loglog_transient2.csv', pd.DataFrame, 4),
    ('data/Stable_Pattern_loglog_transient3.csv', pd.DataFrame, 4),
    ('data/Changing_Pattern_loglog_transient1.csv', pd.DataFrame, 2),
    ('data/Changing_Pattern_loglog_transient2.csv', pd.DataFrame, 2),
    ('data/Changing_Pattern_loglog_transient3.csv', pd.DataFrame, 4),
])
def test_PTAClassifier_predict_optimize(transient, expected_type, expected_features):

    data = pd.read_csv(transient)

    clf = PTAClassifier()
    clf.fit(data.values)
    result, dist, _ = clf.predict_optimize(max_window_length=1, min_filter_value=-0.5, max_filter_value=0.5)
    print(result.Regime.nunique())
    assert isinstance(result, expected_type), f'Result should be {expected_type}.'
    assert isinstance(dist, expected_type), f'Dist output should be {expected_type}.'
    assert result.Regime.nunique() == expected_features, f'Expected {expected_features} distinct regime features for {transient}.'
    assert 0 in result.Regime.unique(), f'Radial flow feature should be found in {transient}.'