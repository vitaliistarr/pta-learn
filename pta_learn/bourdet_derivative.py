from bisect import bisect_right
from .superposition_calculation import *


# function to calculate bourdet derivative
def der(df_super, L):

    """
    Computes Bourdet Derivatives

    This function calculates the derivative of pressure changes (`Delta_Pressure`) with respect to time (`Delta_Time`) 
    using Bourdet method. The derivative is computed for each point in the input DataFrame 
    based on neighboring points, using binary search for efficient indexing.

    Parameters:
    - df_super (pd.DataFrame): Input DataFrame containing the following columns:
        - `Delta_Time`: Time differences between consecutive measurements.
        - `Superposition`: Superposition time values from superposition_calculation.py.
        - `Delta_Pressure`: Pressure differences corresponding to each time step.
    - L (float): A scaling factor used to compute exponential time thresholds for selecting neighboring points.

    Returns:
    - df_derivative (pd.DataFrame): A DataFrame containing the computed derivatives with the following columns:
        - `Superposition`: Superposition time values.
        - `Delta_Pressure`: Pressure differences.
        - `Delta_Time`: Time differences.
        - `Derivative`: Computed derivative values (logarithmically scaled).

    Mathematical Formulation:
    The derivative is calculated using a weighted average of pressure changes over time:
        derivative = ( (|p1 - p0| / (t1 - t0) * (t2 - t1) + (|p2 - p1| / (t2 - t1) * (t1 - t0) ) / ( (t1 - t0) + (t2 - t1) )
    where:
        - t0, t1, t2: Superposition times for the current, next, and subsequent points.
        - p0, p1, p2: Pressure differences at times t0, t1, and t2.

    The result is then scaled by `1 / log(10)` to convert it to a logarithmic derivative.
    """

    # set the smoothing factor, this is important when the data goes to the end, to ensure the real response, the L should be smaller.
    exp_L = np.exp(L)
    exp_L_half = np.exp(L / 2)
    derivatives = []
    
    # Create sorted lists for binary search
    delta_times = df_super['Delta_Time'].tolist()
    superposition = df_super['Superposition'].tolist()
    delta_pressure = df_super['Delta_Pressure'].tolist()
    
    for i in range(len(delta_times) - 2):
        t0, p0, dt0 = superposition[i], delta_pressure[i], delta_times[i]
        dt_exp_L = dt0 * exp_L
        
        # Binary search for the next index
        idx1 = bisect_right(delta_times, dt_exp_L, i + 1)
        if idx1 < len(delta_times):
            t1, p1, dt1 = superposition[idx1], delta_pressure[idx1], delta_times[idx1]
            dt_exp_L_next = dt1 * exp_L
            
            # Binary search for the subsequent index
            idx2 = bisect_right(delta_times, dt_exp_L_next, idx1 + 1)
            if idx2 < len(delta_times):
                t2, p2 = superposition[idx2], delta_pressure[idx2]
                # Calculate derivatives
                x1, x2 = t1 - t0, t2 - t1
                pp1, pp2 = abs(p1 - p0), abs(p2 - p1)
                der = (pp1 / x1 * x2 + pp2 / x2 * x1) / (x1 + x2)
                derivatives.append([t1, p1, dt1, der / np.log(10)])
            else:
                dt_exp_L_half_next = dt1 * exp_L_half
                idx2 = bisect_right(delta_times, dt_exp_L_half_next, idx1 + 1)
                if idx2 < len(delta_times):
                    t2, p2 = superposition[idx2], delta_pressure[idx2]
                    # Calculate derivatives
                    x1, x2 = t1 - t0, t2 - t1
                    pp1, pp2 = abs(p1 - p0), abs(p2 - p1)
                    der = (pp1 / x1 * x2 + pp2 / x2 * x1) / (x1 + x2)
                    derivatives.append([t1, p1, dt1, der / np.log(10)])

    df_derivative = pd.DataFrame(derivatives, columns=['Superposition', 'Delta_Pressure', 'Delta_Time', 'Derivative'])
    
    return df_derivative





def select_period(df_bhp, df_rate, start_t, end_t):
    """
    Selects a period from the BHP (Bottom Hole Pressure) and rate DataFrames based on the specified start and end times.

    Parameters:
    - df_bhp (DataFrame): The input BHP DataFrame.
    - df_rate (DataFrame): The input rate DataFrame.
    - start_t (float): The start time of the period in hours.
    - end_t (float): The end time of the period in hours.

    Returns:
    - bhp_shut (DataFrame): The selected BHP DataFrame for the specified period.
    - rate_shut (DataFrame): The selected rate DataFrame for the specified period.
    """
    bhp_shut = df_bhp[(df_bhp['Time'] >= start_t) & (df_bhp['Time'] <= end_t)]
    rate_shut = df_rate[(df_rate['Time'] >= start_t) & (df_rate['Time'] <= end_t)]
    return bhp_shut, rate_shut


# new version of calculate loglog, this will be updated when the aw_viewer is ready

def cal_loglog_shut(df_bhp, df_rate, selected_list,all_breakpoints,rebuilt_rate,idx, L=0.1):

    # Make sure the provided index is within the range of selected_list
    if idx >= len(selected_list):
        raise ValueError(
            f"Provided index {idx} is out of range for selected_list of length {len(selected_list)}")

    df_bhp_1 = df_bhp.copy()
    df_rate_1 = df_rate.copy()

    # select the value in start/hr column in the row of idx
    ss = selected_list['start/hr'].iloc[idx]
    # select the value in end/hr column
    se = selected_list['end/hr'].iloc[idx]
    # select the value in start/timestamp column in the row of idx
    ss_time = selected_list['start/timestamp'].iloc[idx]
    # select the value in end/timestamp column
    se_time = selected_list['end/timestamp'].iloc[idx]
    # duration of the transient
    duration = selected_list['duration/hr'].iloc[idx]
    # select the data during shut-in
    df_target_s1 = df_bhp[(df_bhp_1['Time'] >= ss) & (df_bhp_1['Time'] <= se)].reset_index(drop=True)

    # the pressure for superposition plot
    super_p = df_target_s1['Pressure'].values[1:]

    # select the data before shut-in, this is necessary for loglog plot
    df_shut_bhp_1_before, df_shut_rate_1_before = select_period(df_bhp_1, df_rate_1, 0, se)

    # get all the breakpoints before the start of the shutin
    a1_breakpoints = all_breakpoints[all_breakpoints['Time'] <= ss].reset_index(drop=True)

    # get the rows from 0 to len(ia1_breakpoints) from rebuilt_rate
    rate_rebuilt_1 = rebuilt_rate.iloc[:len(a1_breakpoints)].reset_index(drop=True)

    # Extract the 'weighted average rate' column as a NumPy array
    weighted_rate_1 = rate_rebuilt_1['Weighted Averaged Rate'].to_numpy()

    # # superposition time in the shutin period
    df_super_detect_s1 = super_time_shutin(a1_breakpoints, weighted_rate_1, df_target_s1)

    # # derivative in the shutin period
    df_derivative_detect_s1 = der(df_super_detect_s1,L)

    return df_derivative_detect_s1, weighted_rate_1, a1_breakpoints

def cal_loglog_inj(df_bhp, df_rate, selected_list,all_breakpoints,rebuilt_rate,idx, L=0.2):

    # Make sure the provided index is within the range of selected_list
    if idx >= len(selected_list):
        raise ValueError(
            f"Provided index {idx} is out of range for selected_list of length {len(selected_list)}")

    df_bhp_1 = df_bhp.copy()
    df_rate_1 = df_rate.copy()

    # select the value in start/hr column in the row of idx
    ins = selected_list['start/hr'].iloc[idx]
    # select the value in end/hr column
    ine = selected_list['end/hr'].iloc[idx]
    # duration of the transient
    duration = selected_list['duration/hr'].iloc[idx]
    # select the value in start/timestamp column in the row of idx
    ins_time = selected_list['start/timestamp'].iloc[idx]
    # select the value in end/timestamp column
    ine_time = selected_list['end/timestamp'].iloc[idx]

    # select the data during shut-in
    df_target_i1 = df_bhp_1[(df_bhp_1['Time'] >= ins) & (df_bhp_1['Time'] <= ine)].reset_index(drop=True)
    df_target_rate_i1 = df_rate_1[(df_rate_1['Time'] >= ins) & (df_rate_1['Time'] <= ine)].reset_index(drop=True)
    
    # the pressure for superposition plot
    super_p = df_target_i1['Pressure'].values[1:]

    # select the data before shut-in, this is necessary for loglog plot
    df_inj_bhp_1_before, df_inj_rate_1_before = select_period(df_bhp_1, df_rate_1, 0, ine)

    # get all the breakpoints before the start of the shutin
    ia1_breakpoints = all_breakpoints[all_breakpoints['Time'] <= ins].reset_index(drop=True)

    # get the rows from 0 to len(ia1_breakpoints) from rebuilt_rate
    rate_rebuilt_1 = rebuilt_rate.iloc[:len(ia1_breakpoints)].reset_index(drop=True)

    # Extract the 'weighted average rate' column as a NumPy array
    weighted_rate_i1 = rate_rebuilt_1['Weighted Averaged Rate'].to_numpy()

    # superposition time in the inj period
    df_super_detect_i1 = super_time_inj(ia1_breakpoints,weighted_rate_i1,df_target_i1)

    # calculate the derivative in the inj period
    df_derivative_detect_i1 = der(df_super_detect_i1,L)

    return df_derivative_detect_i1, weighted_rate_i1,ia1_breakpoints

def ue_centered_moving(
    dfs,
    resample_time=500,
    log_half_window=0.2,
    k_std=2.0,
    time_col="time",
    deriv_col="deriv",
):
    """
    Compute moving uncertainty envelope (UE) from one or more transients
    using a centered sliding window defined in log10(time).

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        Each DataFrame must contain [time_col, deriv_col].
    resample_time : int
        Number of points in the common log-time grid.
    log_half_window : float
        Half-width (in log10 decades) of the centered window around each resample_time.
    k_std : float
        Number of std-dev (in log-space) for the UE limits.
    time_col, deriv_col : str
        Column names for time and derivative.

    Returns
    -------
    time_resampled, UE_moving_mean, UE_lower_limit, UE_upper_limit
    """
    # 1) resample input data to a log space
    t_min_all = max(min(df[time_col].min() for df in dfs), 1e-12)
    t_max_all = max(df[time_col].max() for df in dfs)

    time_resampled = np.logspace(np.log10(t_min_all), np.log10(t_max_all), num=resample_time)

    # Interpolate each transient onto time_resampled
    P = np.vstack([
        np.interp(
            time_resampled,
            df[time_col].to_numpy(),
            df[deriv_col].to_numpy(),
        )
        for df in dfs
    ])

    # 2) Convert log_window (decades) to number of resampled data points
    # log_step is the difference in decades between consecutive time_resampled data points
    log_step = np.log10(time_resampled[1] / time_resampled[0])
    half_window_pts = max(1, round(log_half_window / log_step))

    # 3) Compute centered-simple moving mean/std in log-space using pooled values from all transients
    mean_log = np.full(resample_time, np.nan)
    std_log  = np.full(resample_time, np.nan)

    for j in range(resample_time):
        left  = max(0, j - half_window_pts)
        right = min(resample_time, j + half_window_pts + 1)

        window_vals = P[:, left:right].ravel()
        window_vals = window_vals[window_vals > 0]

        if window_vals.size == 0:
            if j > 0:
                mean_log[j] = mean_log[j-1]
                std_log[j]  = std_log[j-1]
            else:
                mean_log[j] = np.nan
                std_log[j]  = 0.0
        else:
            window_log   = np.log10(window_vals)
            mean_log[j]  = window_log.mean()
            std_log[j]   = window_log.std(ddof=0)

    # UE limits in log-space then transform to linear
    lower_log = mean_log - k_std * std_log
    upper_log = mean_log + k_std * std_log

    UE_moving_mean = 10**mean_log
    UE_lower_limit = 10**lower_log
    UE_upper_limit = 10**upper_log

    return time_resampled, UE_moving_mean, UE_lower_limit, UE_upper_limit