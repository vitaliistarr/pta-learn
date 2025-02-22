import numpy as np
import pandas as pd

from tpmr import *
from lmir import *

def create_injection_periods(shutin_breakpoints, df_bhp_1):
    """
    Creates injection periods from the results of TPMR methods (shut-in breakpoints).

    Parameters:
    - shutin_breakpoints (pd.DataFrame): DataFrame containing start and end times of shut-in periods.
    - df_bhp_1 (pd.DataFrame): DataFrame containing pressure data.

    Returns:
    - pd.DataFrame: DataFrame containing start and end times of injection periods, excluding periods with zero duration.
    """

    # Create a new dataframe with start from 'end/hr' of shutin_breakpoints and end from the 2nd row of 'start/hr' of shutin_breakpoints
    injection_periods = pd.DataFrame({
        'start/hr': shutin_breakpoints['end/hr'].iloc[0:-1].values, 
        'end/hr': shutin_breakpoints['start/hr'].iloc[1:].values,
        'start/timestamp': shutin_breakpoints['end/timestamp'].iloc[0:-1].values, 
        'end/timestamp': shutin_breakpoints['start/timestamp'].iloc[1:].values
    })

    # Add a new row at the beginning of injection_periods
    first_row = pd.DataFrame({
        'start/hr': [df_bhp_1['Time'].iloc[0]],
        'end/hr': [shutin_breakpoints['start/hr'].iloc[0]],
        'start/timestamp': [df_bhp_1['Timestamp'].iloc[0]],
        'end/timestamp': [shutin_breakpoints['start/timestamp'].iloc[0]]
    })
    injection_periods = pd.concat([first_row, injection_periods], ignore_index=True)

    # Add a new row at the end of injection_periods
    last_row = pd.DataFrame({
        'start/hr': [shutin_breakpoints['end/hr'].iloc[-1]],
        'end/hr': [df_bhp_1['Time'].iloc[-1]],
        'start/timestamp': [shutin_breakpoints['end/timestamp'].iloc[-1]],
        'end/timestamp': [df_bhp_1['Timestamp'].iloc[-1]]
    })
    injection_periods = pd.concat([injection_periods, last_row], ignore_index=True)

    # Calculate duration and add it as a new column
    injection_periods['duration/hr'] = injection_periods['end/hr'] - injection_periods['start/hr']

    # Filter out rows where the duration is zero
    injection_periods = injection_periods[injection_periods['duration/hr'] != 0]

    # Add new column for status
    injection_periods['status'] = 'flowing'

    # Reorder the columns to match shutin_breakpoints
    injection_periods = injection_periods[['start/hr', 'end/hr', 'duration/hr', 'start/timestamp', 'end/timestamp', 'status']]

    # Reset the index of injection_periods
    injection_periods.reset_index(drop=True, inplace=True)
    
    return injection_periods


def identify_all_flowing(
    injection_periods: pd.DataFrame, 
    df_bhp: pd.DataFrame, 
    order: int = None,  # Set default to None to indicate that it will be determined inside
    start_filter_hours: int = None,  # Same here
    end_filter_hours: int = None,  # And here
) -> pd.DataFrame:
    """
    Detect injection breakpoints in pressure data within specified injection periods by using LMIR.
    
    Parameters:
    - injection_periods (pd.DataFrame): DataFrame containing start and end times of injection periods.
    - df_bhp (pd.DataFrame): DataFrame containing pressure data.
    - order (int, optional): The number of adjacent points on each side of a data point to compare when identifying local minima.
                             Example: If order = 5, a point will be considered a local minimum only if it is smaller than the 5 points to its left and the 5 points to its right. 
                             The exceptions are the same in  "argrelextrema" function in "scipy.signal"
  
    - start_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the start of the transient.
                                          Example: If start_filter_hours = 5, any breakpoint detected within the first 5 hours will be removed.

    - end_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the end of the transient.
                                        Example: If end_filter_hours = 5, any breakpoint detected within the last 5 hours will be removed.
    Returns:
    - pd.DataFrame: DataFrame containing detected injection breakpoints.
    """

    # get the pressure data for each injection period
    injection_pressure_list = [
        df_bhp[
            (df_bhp['Time'] >= start_time) & (df_bhp['Time'] <= end_time)
        ].reset_index(drop=True)
        for start_time, end_time in zip(injection_periods['start/hr'], injection_periods['end/hr'])
    ]

    # list to store the detected injection breakpoints
    injection_multibp_list = []

    # df to store the detected flowing transient
    flowing = pd.DataFrame()
    TI_ft = pd.DataFrame()

    for test_df in injection_pressure_list:

        # run LMIR on each injection period
        flowing_period, multirate_bp_period,filtered_minima_df, params = LMIR(test_df,
                                                                      order=order, 
                                                                      start_filter_hours=start_filter_hours, 
                                                                      end_filter_hours=end_filter_hours)
        # add detected multi_bp to the list
        injection_multibp_list.append(filtered_minima_df)      

        # add flowing_injection to flowing
        flowing = pd.concat([flowing, flowing_period], ignore_index=True)

        # add detected multi_bp including the start and end of flowing to TI_ft
        TI_ft = pd.concat([TI_ft, multirate_bp_period], ignore_index=True)

    # concatenate all the detected multi_bp
    injection_breakpoints = pd.concat(injection_multibp_list, ignore_index=True)    

    return flowing, TI_ft,injection_breakpoints, params

def filter_TIft(df_bhp,TI_ft, flowing, interval_flowing):

    """
    Filters the `flowing` DataFrame to include only periods where the duration is greater than or equal to `interval_flowing`.

    Parameters:
    - df_bhp (pd.DataFrame): DataFrame containing BHP (Bottom Hole Pressure) data. 
    - TI_ft (pd.DataFrame): DataFrame containing a 'Time' column.
    - flowing (pd.DataFrame): DataFrame containing flowing period data with a 'duration/hr' column.
    - interval_flowing (float or int): Threshold duration (in hours) used to filter both flowing periods and TI data.

    Returns:
    - flowing_filtered (pd.DataFrame): Filtered DataFrame containing only flowing periods with durations >= `interval_flowing`.
    - TI_ft_filtered (pd.DataFrame): Filtered DataFrame containing TI data where the time difference between consecutive rows exceeds `interval_flowing`.
    """

    # Filter the flowing periods based on the duration
    flowing_filtered = flowing[flowing['duration/hr'] >= interval_flowing].reset_index(drop=True)

    # Create an empty DataFrame with the same columns as df
    TI_ft_filtered = pd.DataFrame(columns=TI_ft.columns)

    # Calculate absolute time differences between consecutive rows
    time_diff = TI_ft['Time'].diff(-1).abs()

    # Find indices where time difference exceeds the threshold
    filtered_indices = time_diff[time_diff > interval_flowing].index

    # Select rows based on filtered indices and use concat to add them to TI_ft_filtered
    rows_to_add = []
    for idx in filtered_indices:
        if idx < len(TI_ft) - 1:
            rows_to_add.append(TI_ft.iloc[idx])
            rows_to_add.append(TI_ft.iloc[idx + 1])

    # Concatenate rows to the empty DataFrame
    if rows_to_add:
        TI_ft_filtered = pd.concat([TI_ft_filtered, pd.DataFrame(rows_to_add)], ignore_index=True).drop_duplicates().reset_index(drop=True)

    return flowing_filtered, TI_ft_filtered


def find_all_breakpoints(shutin_breakpoints: pd.DataFrame, 
                         injection_breakpoints: pd.DataFrame, 
                         df_bhp_1: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate all breakpoints by combining shutin and injection breakpoints.
    
    Parameters:
    - shutin_breakpoints (pd.DataFrame): DataFrame containing shutin breakpoints with 'start/hr', 'end/hr', 'start/timestamp', and 'end/timestamp' columns.
    - injection_breakpoints (pd.DataFrame): DataFrame containing injection breakpoints with 'Time' and 'Timestamp' columns.
    - df_bhp_1 (pd.DataFrame): DataFrame containing pressure data with 'Time' and 'Timestamp' columns.
    
    Returns:
    - pd.DataFrame: DataFrame containing all breakpoints with 'Time', 'Timestamp', and 'label' columns.
    """
    all_breakpoints = pd.DataFrame(columns=['Time', 'Timestamp', 'label'])
    
    # Add rows from shutin_breakpoints
    shutin_data = {
        'Time': shutin_breakpoints[['start/hr', 'end/hr']].values.flatten(),
        'Timestamp': shutin_breakpoints[['start/timestamp', 'end/timestamp']].values.flatten(),
        'label': ['shutin'] * (2 * len(shutin_breakpoints))
    }
    all_breakpoints = pd.concat([all_breakpoints, pd.DataFrame(shutin_data)], ignore_index=True)
    
    # Add rows from injection_breakpoints
    injection_data = {
        'Time': injection_breakpoints['Time'].values,
        'Timestamp': injection_breakpoints['Timestamp'].values,
        'label': ['multibp'] * len(injection_breakpoints)
    }
    all_breakpoints = pd.concat([all_breakpoints, pd.DataFrame(injection_data)], ignore_index=True)
    
    # Add the first and last rows from df_bhp_1
    edge_data = {
        'Time': [df_bhp_1['Time'].iloc[0], df_bhp_1['Time'].iloc[-1]],
        'Timestamp': [df_bhp_1['Timestamp'].iloc[0], df_bhp_1['Timestamp'].iloc[-1]],
        'label': ['start', 'end']
    }
    all_breakpoints = pd.concat([pd.DataFrame(edge_data), all_breakpoints], ignore_index=True)
    
    # Sort the values by 'Time'
    all_breakpoints = all_breakpoints.sort_values(by='Time').reset_index(drop=True)
    
    #remove duplicates
    all_breakpoints = all_breakpoints.drop_duplicates(subset=['Timestamp'], keep='first').reset_index(drop=True)

    return all_breakpoints

def filter_breakpoints(filtered_transients, all_breakpoints,threshold = 2):

    """
    Filters breakpoints based on a specified time threshold relative to the start of transients.

    This function removes breakpoints that fall within a specified time range (`threshold`) before the start of each transient.
    Because in practical cases, the breakpoints that are too close to the start of a transient are quite noisy and may lead to wrong rate calculations.


    Parameters:
    - filtered_transients (pd.DataFrame): DataFrame containing transient data with a 'start/hr' column indicating the start time of each transient.
    - all_breakpoints (pd.DataFrame): DataFrame containing all breakpoints with a 'Time' column indicating the time of each breakpoint.
    - threshold (float or int, optional): Time range (in hours) before the start of each transient within which breakpoints will be removed. Default is 2 hours.

    Returns:
    - filtered_breakpoints (pd.DataFrame): DataFrame containing breakpoints that do not fall within the specified threshold before any transient.
    - removed_rows (pd.DataFrame): DataFrame containing breakpoints that were removed based on the threshold.
    """

    # Create a copy of the dataframe to avoid modifying the original
    filtered_breakpoints = all_breakpoints.copy()
    
    # Initialize a dataframe to store the removed rows
    removed_rows = pd.DataFrame(columns=all_breakpoints.columns)
    
    # Iterate over each row in the filtered_transients dataframe
    for _, row in filtered_transients.iterrows():
        start_hr = row['start/hr']
        
        # Only filter if start_hr > threshold
        if start_hr > threshold:
            # Define the range
            lower_bound = start_hr - threshold
            upper_bound = start_hr
            
            # Find the rows to be removed
            rows_to_remove = filtered_breakpoints[
                (filtered_breakpoints['Time'] >= lower_bound) & (filtered_breakpoints['Time'] < upper_bound)
            ]
            
            # Append the removed rows to the removed_rows dataframe
            removed_rows = pd.concat([removed_rows, rows_to_remove], ignore_index=True)
            
            # Filter out the rows from the original dataframe
            filtered_breakpoints = filtered_breakpoints.drop(rows_to_remove.index)

            # Reset the index
            filtered_breakpoints = filtered_breakpoints.reset_index(drop=True)
    
    return filtered_breakpoints, removed_rows

def validate_shutin_rate(shutin, w_rate, flowing):

    """
    Validates shut-in periods by checking if the corresponding rate is zero and updates flowing periods accordingly.

    Parameters:
    - shutin (pd.DataFrame): DataFrame containing shut-in periods with a 'start/timestamp' column.
    - w_rate (pd.DataFrame): DataFrame containing rate data with 'Start Timestamp' and 'Weighted Averaged Rate' columns.
    - flowing (pd.DataFrame): DataFrame containing flowing periods.

    Returns:
    - shutin_filtered (pd.DataFrame): DataFrame containing valid shut-in periods (where the rate is zero).
    - flowing_filtered (pd.DataFrame): DataFrame containing updated flowing periods, including reclassified periods.
    """

    # Convert 'start/timestamp' to numpy array
    start_shutin = shutin['start/timestamp'].to_numpy()

    # Find the rows in w_rate from 'Start Timestamp' that are in start_shutin
    mask1 = w_rate['Start Timestamp'].isin(start_shutin)

    # Filter the rows
    w_rate_filtered = w_rate[mask1].reset_index(drop=True)

    # Pick the index from w_rate_filtered where the rate is not zero
    mask2 = w_rate_filtered['Weighted Averaged Rate'] != 0

    # Find the rows from shutin that are in the mask (where rate is not zero)
    reduced_rate = shutin[mask2].reset_index(drop=True)

    # Initialize flowing_filtered with the provided flowing data
    flowing_filtered = flowing.copy()

    # If there are any rows in reduced_rate, add them to flowing_filtered
    if not reduced_rate.empty:
        # Change the column 'status' to 'flowing'
        reduced_rate['status'] = 'flowing'  
        
        # Concatenate reduced_rate to flowing_filtered, sort by 'start/timestamp'
        flowing_filtered = pd.concat([flowing_filtered, reduced_rate], ignore_index=True)
        flowing_filtered = flowing_filtered.sort_values(by='start/timestamp').reset_index(drop=True)

    # Remove the rows from shutin that are in the mask
    shutin_filtered = shutin[~mask2].reset_index(drop=True)

    return shutin_filtered, flowing_filtered


def calculate_weighted_averaged_rate(rate_data, breakpoints, shutin_threshold=None, zero_q_frac=0.1):
    """
    Calculate the weighted averaged rate between breakpoints.

    Parameters:
    - rate_data: DataFrame with rate and timestamps.
    - breakpoints: DataFrame with breakpoints timestamps.
    - zero_q_frac: fraction of a characteristic rate (90% percentile) below which 
    the rate is set to zero  

    Returns:
    - A DataFrame with weighted averaged rates between breakpoints.
    """
    # Initialize the list to store the weighted averaged rates
    weighted_averaged_rates = []

    # Calculate the shut-in threshold
    if not shutin_threshold:
        shutin_threshold = zero_q_frac * rate_data['Rate'].abs().quantile(0.9)

    # Iterate over the breakpoints to calculate the weighted averaged rate
    for i in range(len(breakpoints) - 1):

        # Get the start and end timestamps
        start_time = breakpoints.iloc[i]['Timestamp']
        end_time = breakpoints.iloc[i + 1]['Timestamp']

        # Get the start and end times
        start = breakpoints.iloc[i]['Time']
        end = breakpoints.iloc[i + 1]['Time']

        # Selectthe  rate data based on the timestamps
        mask = (rate_data['Timestamp'] >= start_time) & (rate_data['Timestamp'] < end_time)
        interval_data = rate_data[mask]

        # Check if the interval_data is not empty
        if not interval_data.empty:

            timestamps = interval_data['Timestamp'].to_numpy()
            rates = interval_data['Rate'].to_numpy()

            # Calculate the time differences between consecutive timestamps
            time_diffs = np.diff(timestamps.astype('datetime64[s]').astype(float))
            first_time_diff = (timestamps[0] - start_time).total_seconds()
            last_time_diff = (end_time - timestamps[-1]).total_seconds()

            #  implementing weighted-averaging with in-between fill (vhv)
            #     ^  t1
            #  q1 |--o--|
            #     |     | t2
            #  q2 |     ---o---|
            #     |            |  t3
            #  q3 |            |---o-----|
            #     |            :         |
            #     |-----:------:---------|----->
            #     : dt1 :  dt2 :   dt3   :   time 
            #     :<--->:<---->:<------->:            
            # start_time               end_time   
            #  dt1 = (t1 - start_time) + (t2 - t1)/2
            #  dt2 = (t2 - t1)/2 + (t3 - t2)/2
            #  dt3 = (end_time - t3) + (t3 - t2)/2   
            #  weighted_average = (dt1*q1 + dt2*q2 + dt3*q3)/(dt1 + dt2 + dt3)

            # functions based on the above formula
            half_time_diffs = 0.5*time_diffs
            dt = np.concatenate(([first_time_diff], half_time_diffs)) + \
                 np.concatenate((half_time_diffs, [last_time_diff]))
            weighted_avg_rate = np.average(rates, weights=dt)
        else:
            weighted_avg_rate = 0

        # Check if the weighted average rate is below the shut-in threshold
        if abs(weighted_avg_rate) < shutin_threshold:
            weighted_avg_rate = 0

        # Append the results to the list
        weighted_averaged_rates.append({
            'Start Timestamp': start_time,
            'End Timestamp': end_time,
            'Start Time': start,
            'End Time': end,
            'Weighted Averaged Rate': weighted_avg_rate
        })

    # Convert the list to a DataFrame
    weighted_averaged_rates_df = pd.DataFrame(weighted_averaged_rates)
    
    weighted_averaged_rates_df['Start Time'] = \
        weighted_averaged_rates_df['Start Time'] - \
            weighted_averaged_rates_df['Start Time'].iloc[0]
    weighted_averaged_rates_df['End Time'] = \
        weighted_averaged_rates_df['End Time'] - \
            weighted_averaged_rates_df['Start Time'].iloc[0]
    
    return weighted_averaged_rates_df


def SRT(test_df,df_rate, interval_injection, order, start_filter_hours, end_filter_hours, shutin_threshold=None):
    """
    Perform the SRT (Step Rate Test) step rate test process.

    Parameters:
        - test_df (pd.DataFrame): The input DataFrame containing the raw pressure data.
        - interval_injection (float): The time interval used for identifying injection periods.
        - order (int, optional): The number of adjacent points on each side of a data point to compare when identifying local minima.
                             Example: If order = 5, a point will be considered a local minimum only if it is smaller than the 5 points to its left and the 5 points to its right. 
                             The exceptions are the same in  "argrelextrema" function in "scipy.signal"
  
        - start_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the start of the transient.
                                          Example: If start_filter_hours = 5, any breakpoint detected within the first 5 hours will be removed.

        - end_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the end of the transient.
                                        Example: If end_filter_hours = 5, any breakpoint detected within the last 5 hours will be removed.
 
    Returns:
        flowing (pd.DataFrame): DataFrame with flowing periods identified.
        TI_ft (pd.DataFrame): DataFrame with TI (Transient Injection) periods identified.
        shutin (pd.DataFrame): DataFrame with the same columns as flowing, initialized empty.
        TI (pd.DataFrame): DataFrame with the same columns as TI_ft, initialized empty.
        all_breakpoints (pd.DataFrame): DataFrame containing all breakpoints with labels.
    """
    
    # run LMIR on the test data
    flowing_period, multirate_bp_period, filtered_minima_df, params = LMIR(test_df,
                                                                    order=order,
                                                                    start_filter_hours=start_filter_hours,
                                                                    end_filter_hours=end_filter_hours)
    # Filter the flowing periods based on the interval_injection
    flowing_filtered, TI_ft_filtered = filter_TIft(test_df,multirate_bp_period, flowing_period, interval_injection)

    # Initialize shutin and TI DataFrames to match with the return with the other functions
    shutin = pd.DataFrame(columns=flowing_filtered.columns)
    TI = pd.DataFrame(columns=TI_ft_filtered.columns)
    
    # Create all_breakpoints DataFrame
    all_breakpoints = pd.DataFrame(columns=['Time', 'Timestamp', 'label'])
    
    # Add rows from multirate_bp_period to all_breakpoints
    injection_data = {
        'Time': multirate_bp_period['Time'].values,
        'Timestamp': multirate_bp_period['Timestamp'].values,
        'label': ['multibp'] * len(multirate_bp_period)
    }
    
    # Concatenate the data to all_breakpoints
    all_breakpoints = pd.concat([all_breakpoints, pd.DataFrame(injection_data)], ignore_index=True)

    # Calculate the weighted average rate for each transient
    w_rate = calculate_weighted_averaged_rate(rate_data=df_rate, breakpoints=all_breakpoints, shutin_threshold=shutin_threshold)

    return shutin,flowing_filtered, TI, TI_ft_filtered, all_breakpoints, w_rate, params



# ti_workflow is the function to detect the transients longer than certain intervals by using TPMR for shutin transient and LMIR for flowing transients.
def ti_workflow(df_bhp, df_rate, p: float, interval_shutin: float, interval_injection: float,
         shutin_threshold = None,
         order: int = None,  # Add these optional parameters
         start_filter_hours: int = None,
         end_filter_hours: int = None):
    """
    ti_workflow is the function to detect the transients longer than certain intervals by 
    using TPMR for shutin transient
    using LMIR for flowing transients.

    Parameters:
    - df_bhp (pd.DataFrame): DataFrame containing pressure data. 
    - df_rate (pd.DataFrame): DataFrame containing rate data for calcaulte rebuit_rate.
    - p (float): Pressure threshold, the detail should refer to tpmr.py.
    - interval_shutin (float): Time interval for shut-in transient.
    - interval_injection (float): Time interval for injection transient.
    - order (int, optional): The number of adjacent points on each side of a data point to compare when identifying local minima.
                             Example: If order = 5, a point will be considered a local minimum only if it is smaller than the 5 points to its left and the 5 points to its right. 
                             The exceptions are the same in  "argrelextrema" function in "scipy.signal"
  
    - start_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the start of the transient.
                                          Example: If start_filter_hours = 5, any breakpoint detected within the first 5 hours will be removed.

    - end_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the end of the transient.
                                        Example: If end_filter_hours = 5, any breakpoint detected within the last 5 hours will be removed.
 
                                        
    Note: p is for TPMR function and only works for shutin transient detection.
          order is for LMIR function and only works for flowing transient detection.
          start_filter_hours is for LMIR function and only works for flowing transient detection.
          end_filter_hours is for LMIR function and only works for flowing transient detection.

    Returns:
    - pd.DataFrame: DataFrame containing detected shutin trnasients and flowing transients and all the breakpoints
    """
    # run TPMR to detect shut-in transients
    shutin_bp_all, shutin_bp_interval, shutin_transient_all, shutin = TPMR(df_bhp, p, interval_shutin)

    # detect if there is any shut-in transient, if non shut-in transient detected, run SRT function by only using LMIR to detect injection transient
    if shutin.empty:
        shutin,flowing_filtered, TI, TI_ft_filtered, all_breakpoints, w_rate, params = SRT(df_bhp, df_rate,
                                                                        interval_injection,
                                                                        order = order, 
                                                                        start_filter_hours = start_filter_hours,
                                                                        end_filter_hours = end_filter_hours,
                                                                        shutin_threshold = shutin_threshold)
        # leave this space for the future use
        all_breakpoints_filtered = all_breakpoints
        # leave this space for the future use
        shutin_filtered = shutin
    else:
        # Create injection periods based on all shut-in transients without interval limitation 
        injection_periods = create_injection_periods(shutin_transient_all, df_bhp)
        # Detect all the multi-rate breakpoints inside the injection periods
        flowing, TI_ft, injection_breakpoints, params = identify_all_flowing(injection_periods, df_bhp,
                                                                        order = order,
                                                                        start_filter_hours = start_filter_hours,
                                                                        end_filter_hours = end_filter_hours)
        
        # Filter the injection transients based on the interval_injection
        flowing_filtered, TI_ft_filtered = filter_TIft(df_bhp,TI_ft, flowing, interval_injection)
        # Get all the breakpoints
        all_breakpoints = find_all_breakpoints(shutin_transient_all, injection_breakpoints, df_bhp)

        # concatenate shutin and flowing_filtered dataframes, name filtered_transients
        transients = pd.concat([shutin, flowing_filtered], ignore_index=True).sort_values(by='start/timestamp').reset_index(drop=True)

        # Filter out the rows where Time is between start/hr and start/hr - 2 to get all breakpoints
        all_breakpoints_filtered, removed_rows = filter_breakpoints(transients, all_breakpoints)

        # Calculate the weighted average rate for each transient
        w_rate = calculate_weighted_averaged_rate(rate_data=df_rate, breakpoints=all_breakpoints_filtered, shutin_threshold=shutin_threshold)
        
        # assign the shutin that calculated rate is not 0 as reduced_rate
        shutin_filtered, flowing_filtered = validate_shutin_rate(shutin, w_rate,flowing_filtered)

        # Check if both shutin_filtered and flowing_filtered are not empty
        if not shutin_filtered.empty and not flowing_filtered.empty:
            # Compare the start/hr in the first row from flowing and shutin
            if shutin_filtered['start/hr'].iloc[0] < flowing_filtered['start/hr'].iloc[0]:
                # Drop the first row from shutin_filtered if start/hr is smaller
                shutin_filtered = shutin_filtered.drop(shutin_filtered.index[0]).reset_index(drop=True)
            else:
                # Drop the first row from flowing_filtered otherwise
                flowing_filtered = flowing_filtered.drop(flowing_filtered.index[0]).reset_index(drop=True)


    return shutin_filtered, flowing_filtered, shutin_bp_interval, TI_ft_filtered, all_breakpoints_filtered,w_rate, params

