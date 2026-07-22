import os
import pandas as pd
import pytest
import requests

from pta_learn.ti_misc import to_dataframe, to_dataframe_rate

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE_DIR = os.path.join(DATA_DIR, ".cache_external")

PTA_DATASETS_REPO = "alex11818/pta-datasets"
PTA_DATASETS_REF = "19ff153426c61452b82b912b925d00ee7cc0ef01"
C5_BHP_FILE = "C5R/BHP_C5R.csv"
C5_RATE_FILE = "C5R/q_C5R.csv"


#### Test pta-learn dataset ####

@pytest.fixture(scope="session")
def bhp_series():
    path = os.path.join(DATA_DIR, "ti_p.csv")
    df = pd.read_csv(path, index_col=0)
    # First column assumed timestamp, second assumed pressure/BHP
    ts_col, val_col = df.columns[0], df.columns[1]
    series = pd.Series(
        df[val_col].values,
        index=pd.to_datetime(df[ts_col]),
    )
    return series


@pytest.fixture(scope="session")
def rate_series():
    path = os.path.join(DATA_DIR, "ti_r.csv")
    df = pd.read_csv(path, index_col=0)
    ts_col, val_col = df.columns[0], df.columns[1]
    series = pd.Series(
        df[val_col].values,
        index=pd.to_datetime(df[ts_col]),
    )
    return series


@pytest.fixture(scope="session")
def df_bhp(bhp_series):
    """Preprocessed BHP dataframe with Timestamp, Pressure, Time columns."""
    return to_dataframe(bhp_series)


@pytest.fixture(scope="session")
def df_rate(rate_series):
    """Preprocessed rate dataframe with Timestamp, Rate, Time columns."""
    return to_dataframe_rate(rate_series)


####  C5 dataset ####

def _fetch_and_cache_csv(file_path: str, ref: str = PTA_DATASETS_REF) -> pd.DataFrame:
    """Downloads a CSV from the public pta-datasets repo once, caches it locally,
    and reuses the cached copy on subsequent test runs / CI re-runs."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_name = file_path.replace("/", "__")
    cache_path = os.path.join(CACHE_DIR, cache_name)

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0)

    url = f"https://raw.githubusercontent.com/{PTA_DATASETS_REPO}/{ref}/{file_path}"
    response = requests.get(url, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch {url}: HTTP {response.status_code}. "
            "Check that the file path/ref is correct in the pta-datasets repo."
        )

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(response.text)

    return pd.read_csv(cache_path, index_col=0)


@pytest.fixture(scope="session")
def c5_bhp_series():
    """C5 pressure/BHP data pulled from alex11818/pta-datasets."""
    try:
        df = _fetch_and_cache_csv(C5_BHP_FILE)
    except RuntimeError as e:
        pytest.skip(f"Could not fetch C5 pressure dataset: {e}")

    series = pd.Series(
        pd.to_numeric(df.iloc[:, 0], errors="coerce").values,
        index=pd.to_datetime(df.index, errors="coerce"),
    )
    series = series[series.index.notna()]
    return series.dropna().sort_index()


@pytest.fixture(scope="session")
def c5_rate_series():
    """C5 rate data pulled from alex11818/pta-datasets."""
    try:
        df = _fetch_and_cache_csv(C5_RATE_FILE)
    except RuntimeError as e:
        pytest.skip(f"Could not fetch C5 rate dataset: {e}")

    series = pd.Series(
        pd.to_numeric(df.iloc[:, 0], errors="coerce").values,
        index=pd.to_datetime(df.index, errors="coerce"),
    )
    series = series[series.index.notna()]
    return series.dropna().sort_index()


@pytest.fixture(scope="session")
def df_bhp_c5(c5_bhp_series):
    return to_dataframe(c5_bhp_series)


@pytest.fixture(scope="session")
def df_rate_c5(c5_rate_series):
    return to_dataframe_rate(c5_rate_series)