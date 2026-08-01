import pandas as pd


def calculate_rare(rare_df, t1=None, t2=None, hist_periods=2.0, max_rare=1.0):
    """Calculate RARE (Rate Absolute Relative Error) metric.

    Calculates RARE between t1 - hist_periods*(t2 - t1) and t2, i.e., for
    transient between t1 and t2, and its history before t1.

    Parameters
    ----------
    rare_df : pd.DataFrame
        Timeindexed DataFrame with columns for RARE calculation
        (see calculate_rare_inputs):
        1. q : original rate series
        2. q_rebuilt : rebuilt rate series (segmented by breakpoints),
          interpolated (ffill) at q.index
        3. Q : cumulative volume injected/produced: integral(q*dt)
        4. E : cumulative abs. misallocated volume: integral(|q_rebuilt - q|*dt)
    t1 : datetime, optional
        Start of period. Default is the first index of rare_df.
    t2 : datetime, optional
        End of period. Default is the last index of rare_df.
    hist_periods : float, default=2.0
        Number of historical periods (t2-t1) before t1 to be used
        for calculation.
    max_rare : float or None, default=1.0
        Cap for rare values (for visualization purposes). If None,
        no capping is applied.

    Returns
    -------
    float or None
        RARE as a fraction of misallocated volumes between t1 and t2
        and the historical periods before t1. Returns None if absolute
        volume is zero.

    Notes
    -----
    This function can be used to calculate RARE for the entire history
    of rare_df (by leaving t1 and t2 as None), or for specific periods
    (by specifying t1 and t2).
    """
    if t1 is None: t1 = rare_df.index[0]
    if t2 is None: t2 = rare_df.index[-1]

    dtdt = t2 - t1  # timedelta
    zxc = rare_df.loc[t1 - hist_periods * dtdt:t2, ['E', 'Q']]
    abs_vol = abs(zxc.Q.iloc[-1] - zxc.Q.iloc[0])  # volume inj./produced
    if abs_vol > 0:
        rare = (zxc.E.iloc[-1] - zxc.E.iloc[0]) / abs_vol
        if max_rare is not None:
            rare = min(rare, max_rare)  # capping
    else:
        rare = None

    return rare


def calculate_rare_inputs(rate, rate_rebuilt):
    '''calculates inputs for RARE (Rate Absolute Relative Error)
    Parameters
    ----------
    rate : pd.Series or df_q from ti_workflow (pd.DataFrame)
        if rate series should be timeindexed

    rate_rebuilt : pd.Series or weighted_rate from ti_workflow (pd.DataFrame)
        rate rebuilts

    Note
    ----
    RARE, between t1 and t2, is calculated as follows (see "calculate_rare"):

    Returns:
    --------
    Timeindexed DF for RARE calculation (see calculate_rare) with columns:
    1. q : original rate series
    2. q_rebuilt : rebuilt rate series (segmented by breakpoints),
        interpolated (ffill) at q.index
    3. Q : cumulative volume injected/produced: integral(q*dt)
    4. E : cumulative abs. misallocated volume: integral(|q_rebuilt - q|*dt)
    '''

    # prepocessing rate
    qs = rate.copy()
    if isinstance(rate, pd.Series):
        pass
    elif isinstance(rate, pd.DataFrame):
        # if rate comes directly from ti_workflow ...
        qs = qs.set_index('Timestamp')['Rate']
        if not pd.api.types.is_datetime64_any_dtype(qs.index):
            qs.index = pd.to_datetime(qs.index)
    else:
        raise ValueError("rate must be a pd.Series or pd.DataFrame")

    qs = qs.rename('q')
    qs = qs[~qs.index.duplicated()]

    # preprocessing rate_rebuilt
    qrs = rate_rebuilt.copy()
    if isinstance(rate_rebuilt, pd.Series):
        pass
    elif isinstance(rate_rebuilt, pd.DataFrame):
        # if rate_rebuilt comes directly from ti_workflow
        qrs = qrs.set_index(['Start Timestamp'])['Weighted Averaged Rate']
        if not pd.api.types.is_datetime64_any_dtype(qrs.index):
            qrs.index = pd.to_datetime(qrs.index)
        new_row = pd.Series([qrs.values[-1]], index=[qs.index[-1]])
        new_row.index.name = qrs.index.name
        qrs = pd.concat([qrs, new_row])
    else:
        raise ValueError("rate_rebuilt must be a pd.Series or pd.DataFrame")

    qrs = qrs.rename('q_rebuilt')

    rare_df = pd.concat([qrs, qs], axis=1)
    rare_df['q_rebuilt'] = rare_df['q_rebuilt'].ffill()
    rare_df['q'] = rare_df['q'].ffill()
    rare_df = rare_df.dropna()
    rare_df['dt'] = rare_df.index.to_series().diff().dt.total_seconds() / 86400
    rare_df['Q'] = (rare_df.dt * rare_df.q).cumsum()
    rare_df['E'] = ((rare_df.q_rebuilt - rare_df.q).abs() * rare_df.dt).cumsum()
    first_idx = rare_df.index[0]
    rare_df.loc[first_idx, 'Q'] = 0
    rare_df.loc[first_idx, 'E'] = 0

    return rare_df