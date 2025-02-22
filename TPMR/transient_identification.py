import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates

from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from kneebow.rotor import Rotor
from datetime import datetime,timedelta
from bisect import bisect_right


def to_dataframe(df_bhp):
    """
    Adjusts the BHP DataFrame from aw_io to be compatible with TI and loglog function:
    Note:
    1. Removes NaN values.
    2. Creates a new column 'Timestamp' with the index values and resets the index.
    3. Renames the column 'BHP' to 'Pressure'.
    4. Creates a new column 'Time' with the time in hours since the first timestamp.

    Parameters:
    - df_bhp (Pandas Series): The input pandas.core.series.Series containing BHP data.

    Returns:
    - df_bhp (DataFrame): The adjusted BHP DataFrame with 'Timestamp', 'Pressure', and 'Time' columns.
    """

    # Remove NaN values
    df_bhp = df_bhp.dropna()

    df_bhp = pd.DataFrame(df_bhp)
    # rename the column to 'Pressure' from unknown column name
    df_bhp = df_bhp.rename(columns={df_bhp.columns[0]: 'Pressure'})
    # Create a new column 'Timestamp' with the index values and reset the index
    df_bhp['Timestamp'] = df_bhp.index
    df_bhp.index = range(len(df_bhp))
    
    # # Rename column to 'Pressure'
    # df_bhp = df_bhp.rename(columns={'BHP': 'Pressure'})
    
    # Create a new column 'Time' with the time in hours since the first timestamp
    df_bhp['Time'] = (df_bhp['Timestamp'] - df_bhp.loc[0, 'Timestamp']).dt.total_seconds() / 3600
    
    # Reorder the columns
    df_bhp = df_bhp[['Timestamp', 'Pressure', 'Time']]
    
    return df_bhp


def to_dataframe_rate(df_rate):
    """
    Adjusts the Rate DataFrame from aw_io to be compatible with TI and loglog function:
    Note:
    1. Adds a 'Time' column calculated as the time difference in hours from the first timestamp in the DataFrame.
    2. Reverses the 'Rate' values by multiplying each value by -1.
    
    Parameters:
    - df_rate (Pandas Series): The input pandas.core.series.Series containing rate data.
    
    Returns:
    - df_rate (DataFrame): The adjusted Rate DataFrame with 'Timestamp', 'Rate', and 'Time' columns.
    """
    # remove nan values
    df_rate = df_rate.dropna()
    # change df_rate to dataframe
    df_rate = pd.DataFrame(df_rate)
    # rename the column to 'Rate' from unknown column name
    df_rate = df_rate.rename(columns={df_rate.columns[0]: 'Rate'})
    # add a column with index and rename it to 'Timestamp'
    df_rate['Timestamp'] = df_rate.index
    # change index to integer from 0
    df_rate.index = range(len(df_rate))
    # add a column with accumulated time in hr and rename it to 'Time'
    df_rate['Time'] = (df_rate['Timestamp'] - df_rate.loc[0,'Timestamp']).dt.total_seconds()/3600
    # Timestamp is the first column
    df_rate = df_rate[['Timestamp', 'Rate', 'Time']]

    # # set the absolute value of Rate smaller than 10 to 0
    # df_rate.loc[abs(df_rate['Rate']) < 10, 'Rate'] = 0
    
    return df_rate


def to_dataframe_bht(df_bht):
    '''
    Adjusts the BHT DataFrame from aw_io to be compatible with TI and loglog function:
    Note:
    1. Removes NaN values.
    2. Creates a new column 'Timestamp' with the index values and resets the index.
    3. Renames the column 'BHT' to 'Temperature'.
    4. Creates a new column 'Time' with the time in hours since the first timestamp.

    Parameters:
    - df_bht (Pandas Series): The input pandas.core.series.Series containing BHT data.

    Returns:
    - df_bht (DataFrame): The adjusted BHT DataFrame with 'Timestamp', 'Temperature', and 'Time' columns.
    '''

    # Remove NaN values
    df_bht = df_bht.dropna()
    df_bht = pd.DataFrame(df_bht)
    # rename the column to 'Temperature' from unknown column name
    df_bht = df_bht.rename(columns={df_bht.columns[0]: 'Temperature'})
    # Create a new column 'Timestamp' with the index values and reset the index
    df_bht['Timestamp'] = df_bht.index
    df_bht = df_bht.reset_index(drop=True)

    # Create a new column 'Time' with the time in hours since the first timestamp
    df_bht['Time'] = (df_bht['Timestamp'] - df_bht.loc[0, 'Timestamp']).dt.total_seconds() / 3600

    # Reorder the columns
    df_bht = df_bht[['Timestamp', 'Temperature', 'Time']]
    
    return df_bht

def to_dataframe_whp (df_whp):
    # Remove NaN values
    df_whp = df_whp.dropna()

    df_whp = pd.DataFrame(df_whp)
    # rename the column to 'Pressure' from unknown column name
    df_whp = df_whp.rename(columns={df_whp.columns[0]: 'Pressure'})
    # Create a new column 'Timestamp' with the index values and reset the index
    df_whp['Timestamp'] = df_whp.index
    df_whp = df_whp.reset_index(drop=True)

    # Create a new column 'Time' with the time in hours since the first timestamp
    df_whp['Time'] = (df_whp['Timestamp'] - df_whp.loc[0, 'Timestamp']).dt.total_seconds() / 3600

    # Reorder the columns
    df_whp = df_whp[['Timestamp', 'Pressure', 'Time']]
    
    return df_whp


# Functions inside TPMR
def detect_bottombp(df_bhp, prominence_value):
    """
    This function detects the bottom Breakpoints in a given time series data.
    
    Parameters:
    df_bhp (pd.DataFrame): The input data frame containing the necessary columns from to_dataframe.
    prominence_value (float): The prominence value to be used in the find_peaks function to identify valley points.
    
    Returns:
    pd.DataFrame: A data frame containing the detected bottom BHPs with labels 'flowing'.
    """

    # Find the valley points using the find_peaks function with the inverted pressure values
    valley_points = find_peaks(-df_bhp['Pressure'].values, prominence=prominence_value)[0]
    
    # Create a new data frame 'PTA_f' with the valley points
    PTA_f = pd.DataFrame(df_bhp.iloc[valley_points, :])
    
    # Add a new column 'label' with the value 'flowing'
    PTA_f['label'] = 'flowing'
    
    # Reset the index of the 'PTA_f' data frame
    PTA_f.index = range(len(PTA_f))
    
    return PTA_f

def split_dataframe(df_bhp, PTA_f):
    """
    This function splits the df_bhp DataFrame based on every timestamp from PTA_f.Timestamp to the next timestamp.
    
    Parameters:
    df_bhp (pd.DataFrame): The input data frame containing the necessary columns from to_dataframe.
    PTA_f (pd.DataFrame): The data frame containing the bottom BHPs with labels 'flowing'.
    
    Returns:
    list: A list of DataFrames, each containing a segment of df_bhp between two consecutive points in PTA_f.Timestamp.
    """
    
    # Get the list of timestamp values from PTA_f
    timestamp_values = PTA_f['Timestamp'].values

    # add the start time of df_bhp to the beginning of the timestamp_values list
    timestamp_values = np.insert(timestamp_values, 0, df_bhp['Timestamp'].values[0])
    
    # Create a list to store the split DataFrames
    split_dataframes = []
    
    # Iterate over the timestamp values to get the start and end points for each segment
    for i in range(len(timestamp_values)-1):
        # Get the start and end timestamp values for the current segment
        start_timestamp = timestamp_values[i]
        end_timestamp = timestamp_values[i+1]
        
        # Create a mask to select the rows between the start and end timestamp values
        mask = (df_bhp['Timestamp'] >= start_timestamp) & (df_bhp['Timestamp'] < end_timestamp)
        
        # Select the rows between the start and end timestamp values and append the resulting DataFrame to the list
        segment_df = df_bhp[mask].copy()
        segment_df.reset_index(drop=True, inplace=True)
        split_dataframes.append(segment_df)
    
    return split_dataframes


def detect_topbp(split_dataframes):
    """
    Detect the top breakpoint in each segment in split_dataframes.
    
    Parameters:
    - split_dataframes (list[pd.DataFrame]): List of DataFrames each containing a segment of data.
    
    Returns:
    - pd.DataFrame: DataFrame containing detected top breakpoints with an additional 'label' column set to 'shutin'.
    """

    # Create an empty dataframe to store the top breakpoints
    PTA_s = pd.DataFrame()

    # Create an instance of the Rotor class
    rotor = Rotor()

    # Iterate over the split dataframes
    for df in split_dataframes:
        if not df.empty:

            # sort the data by Time in descending order
            df = df.sort_values(by=['Time'], ascending=False, ignore_index=True)
            # Find the index of the maximum pressure point
            max_pressure_idx = df['Pressure'].idxmax()
            
            # Get the data from time with maximum pressure to the end of the segment
            # Due to the Rotor, the data should be in descending order in Time
            # This is also due to your shape of data
            data = df.loc[:max_pressure_idx, ['Time', 'Pressure']].values

            # Apply the rotation using the Rotor class
            rotor.fit_rotate(data)

            # Get the elbow index
            elbow_index = rotor.get_elbow_index()

            # Judge if elbow_index is 0, means this is a typical segment and the max pressure is the top breakpoint
            if elbow_index == 0:
                # get the row with max_pressure_idx
                row = df.loc[max_pressure_idx]
                # use concat to add the row to top_bp
                PTA_s = pd.concat([PTA_s, row.to_frame().T])
            else:
                # get the row with elbow_index
                row = df.loc[elbow_index]
                # make another judge that the pressure value of elbow_index is 
                # larger than the average of maximum pressure and the last pressure in df
                if row['Pressure'] > (df.loc[max_pressure_idx, 'Pressure'] + df.iloc[0, 1]) / 2:
                    # use concat to add the row to top_bp
                    PTA_s = pd.concat([PTA_s, row.to_frame().T])
                else:
                    # get the row with max_pressure_idx
                    row = df.loc[max_pressure_idx]
                    # use concat to add the row to top_bp
                    PTA_s = pd.concat([PTA_s, row.to_frame().T])
        else:
            continue
        # add a new column 'label' with the value 'shutin'
        PTA_s['label'] = 'shutin'
        # reset the index of the 'PTA_s' data frame
        PTA_s.index = range(len(PTA_s))

    return PTA_s     


def detect_PTA(df_bhp, PTA_f, PTA_s):
    """
    Combine the bottom and top breakpoints into a single DataFrame.

    Parameters:
    - df_bhp (pd.DataFrame): The input data frame containing the necessary columns from to_dataframe.
    - PTA_f (pd.DataFrame): The data frame containing the bottom BHPs with labels 'flowing'.
    - PTA_s (pd.DataFrame): The data frame containing the top BHPs with labels 'shutin'.

    Returns:
    - pd.DataFrame: DataFrame containing the detected points of start flowing and shut-in, sorted by time.
    """

    PTA = pd.concat([PTA_f, PTA_s])

    # Get the first row from df_bhp and add a new column 'label' with the value 'start'
    PTs = df_bhp.iloc[:1].copy()
    PTs['label'] = 'start'

    # Get the last row from df_bhp and add a new column 'label' with the value 'end'
    PTe = df_bhp.iloc[-1:].copy()
    PTe['label'] = 'end'

    # Concatenate PTs and PTe to PTA
    PTA = pd.concat([PTA, PTs, PTe])

    # Sort PTA by the "Time" column in ascending order
    PTA = PTA.sort_values("Time", ascending=True)

    # delete the rows with the same time
    PTA = PTA.drop_duplicates(subset='Timestamp', keep='first')
    # Reset the index of PTA
    PTA.reset_index(drop=True, inplace=True)

    return PTA


def detect_TI(PTA_f, PTA_s, interval):
    """
    Detect the shut-in periods based on the time interval.

    Parameters:
    - PTA_f (pd.DataFrame): The data frame containing the bottom BHPs with labels 'flowing'.
    - PTA_s (pd.DataFrame): The data frame containing the top BHPs with labels 'shutin'.
    - interval (float): The time interval in hours between start of shutin and end of shutin.

    Returns:
    - pd.DataFrame: DataFrame containing the detected shut-in periods with start and end times, sorted by time.
    """
    # Create an empty list to store the rows that satisfy the condition
    valid_rows = []

    # Calculate the duration between corresponding 'Time' values in PTA_f and PTA_s
    duration = PTA_f['Time'].values - PTA_s['Time'].values

    # Find the maximum duration
    max_duration = max(duration)

    # Check if the maximum duration is less than the interval
    if max_duration < interval:
        print("The interval is too big, the dataframe is empty")
        return pd.DataFrame()

    # Find the indices of the rows that satisfy the condition
    valid_indices = np.where(duration > interval)[0]

    # Add the valid rows from PTA_s and PTA_f to the valid_rows list
    for i in valid_indices:
        valid_rows.append(PTA_s.iloc[i])
        valid_rows.append(PTA_f.iloc[i])

    # Create a DataFrame from the valid_rows list
    TI = pd.DataFrame(valid_rows)

    # Sort the DataFrame by the 'Time' column in ascending order
    TI = TI.sort_values(by='Time', ascending=True)

    # Reset the index of the DataFrame
    TI.reset_index(drop=True, inplace=True)

    return TI

def detect_breakps(df_bhp, p = None, interval = None):
    """
    Detects both the start and end points of shut-in transients in a given BHP DataFrame.

    Parameters
    ----------
    df_bhp : DataFrame
        The input Bottom Hole Pressure (BHP) DataFrame.
    p : float
        The pressure threshold (in units) for detecting shut-in breakpoints.
    interval : float
        The time interval in hours between start of shutin and end of shutin.

    Returns
    -------
    PTA : DataFrame
        The DataFrame containing the detected points of start flowing and shut-in, sorted by time.
    PTA_f : DataFrame
        The DataFrame containing the detected points of start flowing, sorted by time.
    PTA_s : DataFrame
        The DataFrame containing the detected points of start shut-in, sorted by time.
    TI : DataFrame
        The DataFrame containing the detected shut-in periods with start and end times, sorted by time.

    Notes
    -----
    - Ensure `df_bhp` contains a 'Time' column with time data.
    - The pressure threshold `p` is used in the `detect_bottombp` function to identify shut-in points based on a certain pressure criterion.
    """
# Set the default value for p if it is not provided
    if p is None:
        p = (np.percentile(df_bhp['Pressure'], 90) - np.percentile(df_bhp['Pressure'], 5)) / 2

    if interval is None:
        interval = 50 if interval is None else interval
        
    # Detect the bottom breakpoints
    PTA_f = detect_bottombp(df_bhp, p)

    # if PTAs is empty, then make empty PTA, PTA_s, TI with the same column names as PTA_f
    if PTA_f.empty:
        PTA = pd.DataFrame(columns=PTA_f.columns)
        PTA_s = pd.DataFrame(columns=PTA_f.columns)
        TI = pd.DataFrame(columns=PTA_f.columns)

    else:
        # Split the df_bhp DataFrame into segments based on the bottom breakpoints
        split_dataframes = split_dataframe(df_bhp, PTA_f)
        # Detect the top breakpoints in each segment
        PTA_s = detect_topbp(split_dataframes)
        # Combine the bottom and top breakpoints into a single DataFrame
        PTA = detect_PTA(df_bhp, PTA_f, PTA_s)
        # Detect the shut-in periods based on the time interval
        TI = detect_TI(PTA_f, PTA_s, interval)

    return PTA,PTA_f, PTA_s, TI

def get_shutin_bps(PTA):
    """
    Returns: all detected shut-in transients without interval limitation
    """

    if PTA.empty:
        shutin_breakpoints = pd.DataFrame()
    
    else:
        # select row with shutin label
        PTA_shutin = PTA.loc[PTA['label'] == 'shutin']
        # select the column Time
        PTA_shutin_start = PTA_shutin[['Timestamp','Time']].reset_index(drop=True)

        # select row with flow label
        PTA_flow = PTA.loc[PTA['label'] == 'flowing']
        # select the column Time 
        PTA_flow_end = PTA_flow[['Timestamp','Time']].reset_index(drop=True)

        # make a new dataframe with the start and end times
        shutin_breakpoints = pd.DataFrame({'start/hr': PTA_shutin_start.Time, 'end/hr': PTA_flow_end.Time, 
                            'duration/hr': PTA_flow_end.Time - PTA_shutin_start.Time,
        'start/timestamp': PTA_shutin_start.Timestamp, 'end/timestamp': PTA_flow_end.Timestamp})

        # add each row a new column with the status of "shutin"
        shutin_breakpoints['status'] = 'shutin'

    return shutin_breakpoints

def get_shutin_intervals(TI):
    """
    Get the shut-in intervals from the shut-in periods longer than interval.
    """
    if TI.empty:
        shutin = pd.DataFrame()
    else:
        # Select rows with 'shutin' label and reset index
        TI_shutin = TI.loc[TI['label'] == 'shutin'].reset_index(drop=True)

        # Select rows with 'flowing' label and reset index
        TI_flow = TI.loc[TI['label'] == 'flowing'].reset_index(drop=True)

        # Create a new DataFrame with the start and end times of each shut-in period
        shutin = pd.DataFrame({
            'start/hr': TI_shutin.Time, 
            'end/hr': TI_flow.Time, 
            'duration/hr': TI_flow.Time - TI_shutin.Time,
            'start/timestamp': TI_shutin.Timestamp, 
            'end/timestamp': TI_flow.Timestamp
        })

        # Add a new column with the status 'shutin'
        shutin['status'] = 'shutin'

    return shutin

def TPMR(df_bhp, p: float, interval_shutin: float):
    """
    Detect shut-in breakpoints and transients.

    This function performs three main tasks:
    1. Detects all shut-in breakpoints within the provided data.
    2. Identifies shut-in transients without interval limitations.
    3. Identifies shut-in transients that meet a specified interval limitation.

    Parameters:
    -----------
    df_bhp : DataFrame
        A pandas DataFrame containing bottom hole pressure (BHP) data, which is required for analysis.
    p : float
        A pressure threshold or parameter used in detecting breakpoints. 
    interval_shutin : float
        A time or interval threshold used to filter shut-in breakpoints and transients.

    Returns:
    --------
    shutin_bp_all : 
        All detected shut-in breakpoints.
    shutin_bp_interval : 
        Shut-in breakpoints that meet the specified interval limitation.
    shutin_transient_all : 
        All detected shut-in transients, regardless of the interval limitation.
    shutin_transient_interval : 
        Shut-in transients that meet the specified interval limitation.

    Notes:
    ------
    - `detect_breakps(df_bhp, p, interval_shutin)` is assumed to return four outputs:
        1. `shutin_bp_all`: All detected shut-in breakpoints.
        2. `PTA_f`: The bottom breakpoint in a shutin transient.
        3. `PTA_s`: The top breakpoint in a shutin transient.
        4. `shutin_bp_interval`: Breakpoints filtered by the interval threshold.
    - `get_shutin_bps(shutin_bp_all)` returns all detected shut-in transients without interval limitation.
    - `get_shutin_intervals(shutin_bp_interval)` returns shut-in transients that meet the specified interval limitation.

    """
    
    # Detect the shut-in breakpoints
    shutin_bp_all, PTA_f, PTA_s, shutin_bp_interval = detect_breakps(df_bhp, p, interval_shutin)
    
    # Detect shut-in transients without interval limitation
    shutin_transient_all = get_shutin_bps(shutin_bp_all)

    # Get shut-in transients with interval limitation, which are often shown in plots
    shutin_transient_interval = get_shutin_intervals(shutin_bp_interval)
    
    return shutin_bp_all, shutin_bp_interval, shutin_transient_all, shutin_transient_interval



def create_injection_periods(shutin_breakpoints, df_bhp_1):
    """
    Creates injection periods from shut-in breakpoints and pressure data.

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


def rotate_new(df):
    """
    Rotate the pressure-time data in the input DataFrame to align the data trend with the x-axis or y-axis.

    The function calculates the angle of rotation based on the slope between the first and last points in the input data.
    It then constructs a rotation matrix using this angle and applies it to the data points to obtain rotated data.
    The rotated data is returned as a new DataFrame with the same structure as the input DataFrame.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the pressure-time data to be rotated. 
                         It must contain at least the following columns: ['Time', 'Pressure', 'Timestamp'].

    Returns:
    - pd.DataFrame: A DataFrame containing the rotated data, preserving the original 'Timestamp' column. 
                    The returned DataFrame has the same structure as the input DataFrame: ['Time', 'Pressure', 'Timestamp'].

    Note:
    - The rotation is performed in a way that the line formed by the first and last points in the input data 
      aligns with one of the axes (depending on the data trend) in the rotated data.
    - The 'Timestamp' column is preserved in the rotated data to maintain a reference to the original time points.
    """

    # Calculate the Angle
    m = (df.loc[len(df)-1, 'Pressure'] - df.loc[0, 'Pressure']) / (df.loc[len(df)-1, 'Time'] - df.loc[0, 'Time'])
    theta = np.arctan(m)
    
    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    
    # Apply the rotation matrix to the data points
    rotated_data = np.dot(df[['Time', 'Pressure']].values, rotation_matrix.T)
    
    # Create a new DataFrame with rotated data
    rotated_df = pd.DataFrame(rotated_data, columns=['Time', 'Pressure'])
    rotated_df['Timestamp'] = df['Timestamp']
    
    return rotated_df

def find_local_minima(df, rotated_df, order):
    """
    Find local minima in the rotated DataFrame and map back to the original DataFrame.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        rotated_df (pd.DataFrame): The rotated DataFrame.
        order (int): How many points on each side to use for the comparison to consider
                     a point as a relative extrema.
        
    Returns:
        minima_df (pd.DataFrame): Rows from the original DataFrame corresponding to local minima in the rotated DataFrame.
    """
    
    # Find local minima indices in the rotated DataFrame
    local_minima_indices = argrelextrema(rotated_df['Pressure'].values, np.less, order=order)[0]
    
    # Map indices back to original DataFrame
    minima_df = df.iloc[local_minima_indices]

    # add the row in df with max pressure to the minima_df with concat
    minima_df = pd.concat([minima_df, df.iloc[df['Pressure'].idxmax()].to_frame().T], axis=0)   
    # keep one if there are duplicated rows
    minima_df = minima_df.drop_duplicates(subset=['Time'], keep='first') 
    # sort the dataframe by time
    minima_df = minima_df.sort_values(by=['Time']).reset_index(drop=True)

    return minima_df


def filter_minima_by_time(minima_df, df, n, m):
    """
    Filter out local minima that occur within n hours from the start and m hours from the end.
    
    Parameters:
        minima_df (pd.DataFrame): DataFrame containing potential local minima.
        df (pd.DataFrame): The original DataFrame.
        n (int): Number of hours to exclude minima from the start.
        m (int): Number of hours to exclude minima from the end.
        
    Returns:
        filtered_minima_df (pd.DataFrame): DataFrame containing filtered local minima.
    """
    
    start_time = df['Time'].iloc[0]
    end_time = df['Time'].iloc[-1]
    # Define time boundaries for filtering
    min_time_boundary = start_time + n
    max_time_boundary = end_time - m
    
    # Filter minima outside of the boundaries
    filtered_minima_df = minima_df[(minima_df['Time'] > min_time_boundary) & 
                                   (minima_df['Time'] < max_time_boundary)]
    
    return filtered_minima_df


# splict the injection data separate flowing transients
def split_injection(df_injection):
    flowing = pd.DataFrame()
    TI_ft = pd.DataFrame()
    # Iterate over the rows by index
    for i in range(len(df_injection) - 1):
        # Select consecutive pairs of rows
        pair_df = df_injection.iloc[i:i+2]

        # Add start/hr, end/hr, start/timestamp, end/timestamp, and duration to flowing
        flowing.loc[i, 'start/hr'] = pair_df['Time'].iloc[0]
        flowing.loc[i, 'end/hr'] = pair_df['Time'].iloc[1]
        flowing.loc[i, 'duration/hr'] = pair_df['Time'].iloc[1] - pair_df['Time'].iloc[0]        
        flowing.loc[i, 'start/timestamp'] = pair_df['Timestamp'].iloc[0]
        flowing.loc[i, 'end/timestamp'] = pair_df['Timestamp'].iloc[1]
        flowing.loc[i, 'status'] = 'flowing'

        # put the row of pair_df into TI_ft
        TI_ft = pd.concat([TI_ft, pair_df], axis=0)
        # remove the repeated rows in TI_ft
        TI_ft = TI_ft.drop_duplicates(subset=['Timestamp'], keep='first')
        # add a column 'status' to TI_ft with 'flowing' but the last row is 'shutin'
        TI_ft['status'] = 'flowing'
        TI_ft.loc[TI_ft.index[-1], 'status'] = 'shutin'

    return flowing, TI_ft

def LMIR(test_df, order: int = None, start_filter_hours: float = None, end_filter_hours: float = None):
    """
    Analyze pressure data to identify local minima and flowing periods.

    This function performs several operations:
    1. Rotates the input pressure data.
    2. Finds local minima in the rotated data.
    3. Filters the identified minima by time, with parameters adjusted based on data duration.

    Parameters:
    -----------
    test_df : DataFrame
        A pandas DataFrame containing the pressure data to be analyzed.
    day : int, optional
        The threshold in days to decide order. more dense data needs bigger order, but sparse data needs smaller order.  Default is 10 days.
    order : int, optional
        The order parameter used in identifying local minima. Calculated based on data length if not provided.
    start_filter_hours : float, optional
        parameter for filtering local minima after start time. Default value depends on the test type.
    end_filter_hours : float, optional
        parameter for filtering local minima before end time. Default value depends on the test type.
    Returns:
    --------
    flowing_period : DataFrame
        A DataFrame representing the flowing period based on the identified breakpoints.
    multirate_bp_period : DataFrame
        A DataFrame representing the multi-rate breakpoint period.

    """
    # set the default value
    order = 500 if order is None else order
    start_filter_hours = 5 if start_filter_hours is None else start_filter_hours
    end_filter_hours = 5 if end_filter_hours is None else end_filter_hours

    # Rotate the pressure data
    test_rotated = rotate_new(test_df)

    # Find local minima
    minima_df = find_local_minima(test_df, test_rotated, order=order)

    # Filter minima by time at the beginning and end
    filtered_minima_df = filter_minima_by_time(minima_df, test_df, start_filter_hours, end_filter_hours)

    # Get a DataFrame with the first and last rows of test_df
    first_row = test_df.iloc[[0]]
    last_row = test_df.iloc[[-1]]

    # Concatenate first and last row with filtered minima DataFrame
    multibp_df = pd.concat([first_row, filtered_minima_df, last_row], ignore_index=True)

    # Identify flowing periods based on the number of multibps
    flowing_period, multirate_bp_period = split_injection(multibp_df)

    params = {
        'order': order,
        'start_filter_hours': start_filter_hours,
        'end_filter_hours': end_filter_hours
    }

    return flowing_period, multirate_bp_period, filtered_minima_df, params


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
    - order (int, optional): Order parameter for `find_local_minima`. for step rate test the order should be smaller(30)), while for long duration the order should be bigger(1000)).
    - start_filter_hours (int, optional): Start time filter for `filter_minima_by_time`. Its default is determined based on df_bhp.
    - end_filter_hours (int, optional): End time filter for `filter_minima_by_time`. Its default is determined based on df_bhp.
    
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

    injection_breakpoints = pd.concat(injection_multibp_list, ignore_index=True)    
    return flowing, TI_ft,injection_breakpoints, params

def filter_TIft(df_bhp,TI_ft, flowing, interval_flowing):

    # Filter the flowing periods based on the duration
    flowing_filtered = flowing[flowing['duration/hr'] >= interval_flowing].reset_index(drop=True)

    # # Use pressure to filter flowing periods??? It is a good way for practical data but not for synthetic data
    # flowing_filtered_p = filter_by_pressure_difference(flowing_filtered, df_bhp)

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
    Find all breakpoints by combining shutin and injection breakpoints.
    
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


def SRT(test_df,df_rate, interval_injection, order, start_filter_hours, end_filter_hours, shutin_threshold=None):
    """
    Perform the SRT step rate test process.

    Parameters:
        test_df (pd.DataFrame): The input DataFrame containing the raw pressure data.
        interval_injection (float): The time interval used for identifying injection periods.
        order (int): The order used for finding local minima.
        start_filter_hours (float): Hours to filter minima at the beginning of the data.
        end_filter_hours (float): Hours to filter minima at the end of the data.

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
    
    all_breakpoints = pd.concat([all_breakpoints, pd.DataFrame(injection_data)], ignore_index=True)

    # Calculate the weighted average rate for each transient
    w_rate = calculate_weighted_averaged_rate(rate_data=df_rate, breakpoints=all_breakpoints, shutin_threshold=shutin_threshold)

    return shutin,flowing_filtered, TI, TI_ft_filtered, all_breakpoints, w_rate, params


# ti_workflow is the function to detect the transients longer than certain intervals by using TPMR for shutin transient and LMIR for injection transients.
def ti_workflow(df_bhp, df_rate, p: float, interval_shutin: float, interval_injection: float,
         shutin_threshold = None,
         order: int = None,  # Add these optional parameters
         start_filter_hours: int = None,
         end_filter_hours: int = None):
    """
    ti_workflow is the function to detect the transients longer than certain intervals by 
    using TPMR for shutin transient
    using LMIR for injection transients.
    
    Parameters:
    - df_bhp (pd.DataFrame): DataFrame containing pressure data. 
    - df_rate (pd.DataFrame): DataFrame containing rate data for calcaulte rebuit_rate.
    - p (float): Pressure threshold for pressure increase.
    - interval_shutin (float): Time interval for shut-in transient.
    - interval_injection (float): Time interval for injection transient.
    - order (int, optional): Order parameter for `find_local_minima`. for step rate test the order should be smaller(30)), while for long duration the order should be bigger(1000)).
    - start_filter_hours (int, optional): Start time filter for `filter_minima_by_time`. Its default is determined based on df_bhp.
    - end_filter_hours (int, optional): End time filter for `filter_minima_by_time`. Its default is determined based on df_bhp.
    
    Returns:
    - pd.DataFrame: DataFrame containing detected shutin trnasients and injections transients and all the breakpoints
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

#shutin,flowing,TI,TI_ft,all_breakpoints,w_rate = ti_workflow(df_bhp, df_rate, p, interval_shutin, interval_injection)


def validate_local_minima(filtered_minima_df, df,n = 3, x = 1):
    """
    Validate local minima based on mean pressure values before and after the minima.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame.
        minima_df (pd.DataFrame): DataFrame containing potential local minima.
        n (int): Number of hours to calculate mean pressure before and after minima.
        x (float): Threshold for validating local minima (in bar).
        
    Returns:
        true_minima_df (pd.DataFrame): DataFrame containing validated local minima.
    """
    true_minima_indices = []
    
    for index, row in filtered_minima_df.iterrows():
        # Define time window for P_before and P_after
        time_before = row['Time'] - n
        time_after = row['Time'] + n
        
        # Calculate P_before and P_after
        P_before = df[(df['Time'] >= time_before) & (df['Time'] < row['Time'])]['Pressure'].mean()
        P_after = df[(df['Time'] <= time_after) & (df['Time'] > row['Time'])]['Pressure'].mean()
        
        # Validate local minima
        if abs(P_after - P_before) > x:
            true_minima_indices.append(index)
    
    # Create DataFrame containing true local minima
    true_minima_df = df.loc[true_minima_indices]
    
    return true_minima_df

def filter_shutin_by_temp(df_bht_1, shutin):
    """
    Filters shutin data based on temperature data.

    Parameters:
    df_bht_1 (pd.DataFrame): The data frame containing temperature data.
    shutin (pd.DataFrame): The data frame containing shutin data.

    Returns:
    pd.DataFrame: A data frame containing the filtered shutin data.
    """
    
    # df_bht_1 = to_dataframe_bht(df_bht_1)
    # Make an empty dataframe to store the filtered shutin
    shutin_filtered = pd.DataFrame()
    
    for i in range(len(shutin)):
        # Select the temperature data in df_bht_1
        mask = (df_bht_1['Time'] >= shutin['start/hr'][i]) & (df_bht_1['Time'] <= shutin['end/hr'][i]-1)
        data_df = df_bht_1[mask]
        
        # Reset the index
        data_df = data_df.reset_index(drop=True)
        
        # Get the minimum temperature in data_df
        min_temp = data_df['Temperature'].min()
        
        # Get the min_temp within the first 1% of the data
        min_temp_1 = data_df['Temperature'][0:int(len(data_df)*0.1)].min()

        # Make a judgement to filter the false shutin, 

        # at the same time, the maximum temperature from 80% t0 85% of the data is at least 1.5 as the maximum temperature from 0% to 1% of the data
        #if min_temp_1 == min_temp and data_df['Temperature'][int(len(data_df)*0.8):int(len(data_df)*0.85)].max() >= 1.5*data_df['Temperature'][0:int(len(data_df)*0.01)].max():

        # if the minimum temperature in the first 1% of the data is the same as the minimum temperature in the data
        if min_temp_1 == min_temp:
            # Use concatenation to add the filtered shutin to the shutin_filtered
            shutin_filtered = pd.concat([shutin_filtered, shutin.loc[i].to_frame().T], ignore_index=True)
    
    return shutin_filtered

def filter_shutin_by_rate(df_rate, shutin):
    """
    Filters shutin data based on rate data.

    Parameters:
    df_rate_1 (pd.DataFrame): The data frame containing rate data.
    shutin (pd.DataFrame): The data frame containing shutin data.

    Returns:
    pd.DataFrame: A data frame containing the filtered shutin data.
    """
    
    # df_rate_1 = to_dataframe_rate(df_rate)
    df_rate_1 = df_rate.copy()
    # Make an empty dataframe to store the filtered shutin
    shutin_filtered = pd.DataFrame()
    
    for i in range(len(shutin)):
        # Select the temperature data in df_bht_1
        mask = (df_rate_1['Time'] >= shutin['start/hr'][i]+3) & (df_rate_1['Time'] <= shutin['end/hr'][i]-3)
        data_df = df_rate_1[mask]
        
        # Reset the index
        data_df = data_df.reset_index(drop=True)
        
        # replace nan with 0
        data_df = data_df.fillna(0)
        
        # Get the median of the rate data
        mean_rate = data_df['Rate'].mean()

        # Make a judgement to filter the false shutin, 

        # if the abs value of median of the rate data is more than 50, then the shutin is false
        if abs(mean_rate) < 10:
            # Use concatenation to add the filtered shutin to the shutin_filtered
            shutin_filtered = pd.concat([shutin_filtered, shutin.loc[i].to_frame().T], ignore_index=True)
        
        # Delete rows in shutin_filtered that rows duration bigger than 1000
        # shutin_filtered = shutin_filtered[shutin_filtered['duration/hr'] < 1000].reset_index(drop=True)
    return shutin_filtered

def filter_flowing_rotation(df_bhp, flowing, **kwargs):
    """
    Filters out false transients from a list of flowing periods by rotation to check multi_bps.

    Parameters
    ----------
    df_bhp : DataFrame
        The input Bottom Hole Pressure (BHP) DataFrame with a 'Time' column.
    flowing : DataFrame
        The DataFrame containing the flowing periods with 'start/hr' and 'end/hr' columns.
    **kwargs : dict, optional
        Additional keyword arguments. Possible keys:
        - 'order' (int): Order parameter for find_local_minima function. Default is 100.
        - 'beginning' (int): Hours to filter at the beginning of the transient. Default is 5.
        - 'end' (int): Hours to filter at the end of the transient. Default is 5.
        - 'n' (int): Hours for validate_local_minima function. Default is 3.
        - 'x' (float): Pressure difference for validate_local_minima function. Default is 1.

    Returns
    -------
    flowing_filtered : DataFrame
        DataFrame passed the filtering criteria.
    """
    # Set default values for keyword arguments if not provided
    order = kwargs.get('order', 100)
    beginning = kwargs.get('beginning', 5)
    end = kwargs.get('end', 5)
    n = kwargs.get('n', 3)
    x = kwargs.get('x', 1)

    df_bhp = df_bhp.copy()
    flowing_filtered = pd.DataFrame()
    # Function to filter out the false injection transients
    for i in range(len(flowing)):
        transient_data = df_bhp[(df_bhp['Time'] >= flowing['start/hr'][i]) & 
                                (df_bhp['Time'] <= flowing['end/hr'][i])].reset_index(drop=True)
        # rotate the pressure data
        transient_rotated = rotate_new(transient_data)
        # find local minima in the rotated data
        minima_transient = find_local_minima(transient_data, transient_rotated, order=order)
        # filter the local minima by time in the beginning and end
        filtered_minima_transient = filter_minima_by_time(minima_transient, transient_data, beginning, end)
        # validate the local minima by pressure difference before and after n hr the minima & more than x bar
        multi_bps_transient = validate_local_minima(filtered_minima_transient, transient_data, n=n, x=x)

        # if multi_bps_transient is empty, concatenate the row in flowing to the flowing_filtered
        if multi_bps_transient.empty:
            row = flowing.iloc[i].to_frame().T
            flowing_filtered = pd.concat([flowing_filtered, row], axis=0)
        else:
            pass
    # reset the index of the flowing_filtered
    flowing_filtered = flowing_filtered.reset_index(drop=True)

    return flowing_filtered


# filter the flowing periods by rate
def filter_flowing_rate(df_rate, flowing, all_breakpoints, **kwargs):
    """
    Filters out false transients from a list of flowing periods by rate.

    Parameters
    ----------
    df_rate : DataFrame
        The input Bottom Hole Pressure (BHP) DataFrame with a 'Time' column.
    flowing : DataFrame
        The DataFrame containing the flowing periods with 'start/hr' and 'end/hr' columns.
    **kwargs : dict, optional
        Additional keyword arguments. Possible keys:

    Returns
    -------
    flowing_filtered : DataFrame
        DataFrame passed the filtering criteria.
    """
    # Set default values for keyword arguments if not provided
    b = kwargs.get('b', 2)
    rate_previous = kwargs.get('rate_previous', 100)

    df_rate = df_rate.copy()
    
    rate_base = abs(df_rate['Rate'].median())/b
    flowing_filtered = pd.DataFrame()
    # filter the flowing periods by rate less than the median of the overall rate
    for i in range(len(flowing)):
        transient_rate = df_rate[(df_rate['Time'] >= flowing['start/hr'][i]) & 
                                (df_rate['Time'] <= flowing['end/hr'][i])].reset_index(drop=True)
        transient_rate_before = df_rate[(df_rate['Time'] >= 0) &
                                        (df_rate['Time'] <= flowing['end/hr'][i])].reset_index(drop=True) 
        
        # ia1_breakpoints is the breakpoints that <= ins -2h 
        # (this 2h need to be determined, because of the rate before shut-in is crucial for the loglog plot)
        ia1_breakpoints = all_breakpoints[all_breakpoints['Time'] <= flowing['start/hr'][i]-2]

        # make a new row with the same columns as ia1_breakpoints
        new_row = pd.DataFrame({'Time': flowing['start/hr'][i], 'Timestamp': flowing['start/timestamp'][i],'label':'shutin' }, index=[0])

        # concatenate the new row to ia1_breakpoints
        ia1_breakpoints = pd.concat([ia1_breakpoints, new_row], ignore_index=True)

        # reset the index
        ia1_breakpoints = ia1_breakpoints.reset_index(drop=True)

        # rebuild the rate history before the injection period, also necessary for loglog plot
        rate_values_i1 = rebuild_rate_inj(ia1_breakpoints, transient_rate_before,transient_rate)

        # if the median of the transient rate is greater than baserate, concatenate the row in flowing to the flowing_filtered
        if abs(transient_rate['Rate'].median()) > rate_base and abs(rate_values_i1[-2]) < rate_previous:
            row = flowing.iloc[i].to_frame().T
            flowing_filtered = pd.concat([flowing_filtered, row], axis=0)
        else:
            pass

    # reset the index of the flowing_filtered
    flowing_filtered = flowing_filtered.reset_index(drop=True)

    return flowing_filtered


def rebuild_rate_shutin(all_breakpoints, df_rate_1):
    """
    Calculates the rate between each pair of breakpoints and returns the array of rate values.
    with adding rate = 0 to the end of rate_values, becasue it is a shut-in period
    
    Parameters:
    - all_breakpoints (pandas.DataFrame): DataFrame of breakpoints.
    - df_rate_1 (pandas.DataFrame): DataFrame containing the rate data before shut-in.

    Returns:
    - rate_values (numpy.ndarray): Array of average rate values between breakpoints.

    Notes:
    - len(rate_values) = len(all_breakpoints)
    """    

    # use 'Time' column in all_breakpoints to get the breakpoints
    # this can adjust to Timestamp if needed
    all_breakpoints = all_breakpoints['Time'].values

    # Calculate the median of rate between every 2 breakpoints period
    rate_values = []
    for i in range(len(all_breakpoints)-1):
        # get the start and end time of each period
        start_time = all_breakpoints[i]
        end_time = all_breakpoints[i+1]

        # get the rate data between start and end time
        rate_data = df_rate_1[(df_rate_1['Time'] >= start_time) & (df_rate_1['Time'] <= end_time)]['Rate']

        # check if rate_data is empty, if so set rate_value to a near 0 value, else calculate the median
        if rate_data.empty:
            rate_value = 0.1
        else:
        # the rebuild rate is the mean bewteen 2 breakpoints
            rate_value = np.mean(rate_data)

        # append the rate value to the list
        rate_values.append(rate_value)

    # add rate = 0 to the end of rate_values, becasue it is a shut-in period
    rate_values = np.append(rate_values,0)

    return rate_values

def rebuild_rate_inj(all_breakpoints, df_rate_1, df_target_inj_rate):
    """
    Calculates the rate between each pair of breakpoints and returns the array of rate values.
    with adding target_inj_rate to the end of rate_values, becasue it is a injection period

    Parameters:
    - all_breakpoints (pandas.DataFrame): DataFrame of breakpoints.
    - df_rate_1 (pandas.DataFrame): DataFrame containing the rate data before shut-in.

    Returns:
    - rate_values (numpy.ndarray): Array of average rate values between breakpoints.

    Notes:
    - len(rate_values) = len(all_breakpoints)
    """    

    # use 'Time' column in all_breakpoints to get the breakpoints
    # this can adjust to Timestamp if needed
    all_breakpoints = all_breakpoints['Time'].values

    # Calculate the median of rate between every 2 breakpoints period
    rate_values = []
    for i in range(len(all_breakpoints)-1):
        # get the start and end time of each period
        start_time = all_breakpoints[i]
        end_time = all_breakpoints[i+1]

        # get the rate data between start and end time
        rate_data = df_rate_1[(df_rate_1['Time'] >= start_time) & (df_rate_1['Time'] <= end_time)]['Rate']

        # check if rate_data is empty, if so set rate_value to 0, else calculate the median
        if rate_data.empty:
            rate_value = 0.1
        else:
        # the rebuild rate is the mean bewteen 2 breakpoints
            rate_value = np.mean(rate_data)

        # append the rate value to the list
        rate_values.append(rate_value)

    # get the median of target injection rate
    target_inj_rate = df_target_inj_rate['Rate'].median()

    # add target_inj_rate to the end of rate_values, becasue it is a injection period
    rate_values = np.append(rate_values,target_inj_rate)

    return rate_values


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

def find_no0_index_from_bottom(arr):
    """
    Finds the index of the first non-zero element from the bottom of the array.
    This is for the purpose of shutin, because reference rate is the last non-zero rate value before shutin.


    Args:
        arr (numpy.ndarray): The input array.

    Returns:
        int or None: The index of the first non-zero element from the bottom of the array.
            If there are no non-zero elements, None is returned.

    Examples:
        >>> import numpy as np
        >>> arr = np.array([0, 0, 0, 5, 2, 0, 0])
        >>> find_no0_index_from_bottom(arr)
        -3
    """
    reversed_arr = np.flip(arr)
    nonzero_indices = np.nonzero(reversed_arr)[0]
    if len(nonzero_indices) > 0:
        index_from_bottom = -1 * (nonzero_indices[0] + 1)
        return index_from_bottom
    else:
        return None

# calculate the superposition time for shutin
def super_time_shutin(bp,rate,df_shutin_target):

    """
    Calculates the superposition time for shut-in periods based on breakpoints, rates, and target pressure data.
    Note:
    1. qn is the last non-zero rate value in the rate list.
    2. t = df_shutin_target.Time.values[1:]-bp[i] is the superposition time in each period.t must be positive.

    Parameters:
    - bp (list): pandas.DataFrame containing the breakpoints.
    - rate (list): np.array of rates.
    - df_shutin_target (pandas.DataFrame): DataFrame containing the target pressure data during shut-in.

    Returns:
    - df_super (pandas.DataFrame): DataFrame containing the superposition time, delta pressure, and delta time.
    """
    
    bp = bp['Time'].values
    
    # use find_no0_index_from_bottom to find the index of the last non-zero value in the rate list
    qn = rate[find_no0_index_from_bottom(rate)]

    # make an array to store the superposition time
    t_s = np.zeros(len(df_shutin_target)-1)
    
    # to calculate each superposition time
    for i in range(len(bp)): # loop through the breakpoints

        # the superposition time in each period
        t = df_shutin_target.Time.values[1:]-bp[i]

        # when i= 0, the first rate is rate[i]-0
        if i == 0:
            t_s = -(rate[i]-0)/qn*np.log10(t)

        # when i >0, the first rate is rate[i]-rate[i-1]
        else:
            t_s = t_s + (-(rate[i]-rate[i-1])/qn*np.log10(t))

    # example for understanding
    #bp = [0,2000,2200,2600,2800,3200]

    #rate = [5000,0,3000,0,9000,0]

    # inj1 = -(5000-0)/qn*np.log10(df_inj.Time.values[1:]-0)

    # bui1= -(0-5000)/qn*np.log10(df_inj.Time.values[1:]-2000)

    # inj2= -(3000-0)/qn*np.log10(df_inj.Time.values[1:]-2200)

    # bui2 = -(0-3000)/qn*np.log10(df_inj.Time.values[1:]-2600)

    # inj3 = -(9000-0)/qn*np.log10(df_inj.Time.values[1:]-2800)

    # bui3 = -(0-9000)/qn*np.log10(df_inj.Time.values[1:]-3200)

    # the superposition time is the sum of the all parts 
    
    #   t_s = inj1+bui1+inj2+bui2+inj3


# the pressure value from the beginning of shutin to the end of shutin
    dp = abs(df_shutin_target.Pressure.values[1:]-df_shutin_target.Pressure.values[0])

# the time value from the beginning of shutin to the end of shutin  
    dt = df_shutin_target.Time.values[1:]-df_shutin_target.Time.values[0] 

# add t_s and dp and dt to a dataframe
    df_super = pd.DataFrame({'Superposition':t_s,'Delta_Pressure':dp,'Delta_Time':dt})
    
# return the dataframe
    return df_super

def find_non_zero_index(arr):
    """
    Finds the index of the first non-zero element in the array.
    This is used to calculate the derivative, because the first non-zero rate is used to normalization.

    Args:
        arr (list or numpy.ndarray): The input array.

    Returns:
        int: The index of the first non-zero element.
            If no non-zero element is found, -1 is returned.

    Examples:
        >>> arr = [0, 0, 0, 5, 2, 0, 3]
        >>> find_non_zero_index(arr)
        3
    """
    for i, num in enumerate(arr):
        if num != 0:
            return i
    return -1  # Return -1 if no non-zero element is found




# calculate the superposition time
def super_time_inj(bp,rate,df_inj):
    """
    Calculates the superposition time for injection periods based on breakpoints, rates, and target pressure data.
    Note:
    1. qn is the stable injection rate, which is the biggest difference from shut-in function.
    2. t = df_inj.Time.values[1:]-bp[i] is the superposition time in each period.t must be positive.

    Parameters:
    - bp (list): List of breakpoints.
    - rate (list): List of rates.
    - df_inj (pandas.DataFrame): DataFrame containing the target pressure data during injection.

    Returns:
    - df_super (pandas.DataFrame): DataFrame containing the superposition time, delta pressure, and delta time.
    """
    
    bp = bp['Time'].values
    # the rate of target injection period
    qn = rate[-1]

    #make an array to store the superposition time
    t_s = np.zeros(len(df_inj)-1)
    
    # to calculate each superposition time
    for i in range(len(bp)):

        # the superposition time in each period
        
        # when i= 0, the first rate is rate[i]-0
        if i == 0:
            t_s = (rate[i]-0)/qn*np.log10(df_inj.Time.values[1:]-bp[i])

        # when i >0, the first rate is rate[i]-rate[i-1]
        else:
            t_s = t_s + (rate[i]-rate[i-1])/qn*np.log10(df_inj.Time.values[1:]-bp[i])

    # example for understanding

    # bp = [0,2000,2200,2600,2800]
    # rate = [5000,0,3000,0,9000]
    # inj1 = (5000-0)/qn*np.log10(df_inj.Time.values[1:]-0)

    # bui1= (0-5000)/qn*np.log10(df_inj.Time.values[1:]-2000)

    # inj2= (3000-0)/qn*np.log10(df_inj.Time.values[1:]-2200)

    # bui2 = (0-3000)/qn*np.log10(df_inj.Time.values[1:]-2600)

    # inj3 = (9000-0)/qn*np.log10(df_inj.Time.values[1:]-2800)

    # #the superposition time is the sum of the all parts 
    
    # t_s = inj1+bui1+inj2+bui2+inj3


# the pressure value from the beginning of shutin to the end of shutin
    dp = abs(df_inj.Pressure.values[1:]-df_inj.Pressure.values[0])

# the time value from the beginning of shutin to the end of shutin  
    dt = df_inj.Time.values[1:]-df_inj.Time.values[0] 

# add t_s and dp and dt to a dataframe
    df_super = pd.DataFrame({'Superposition':t_s,'Delta_Pressure':dp,'Delta_Time':dt})
    
# return the dataframe
    return df_super


# function to calculate bourdet derivative
def der(df_super, L):
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



def normal_calc(df_drv, rate, ref,r_list):
    """
    Calculates the normalized derivative and normalized pressure for the derivative 
    and pressure data.

    Parameters:
    - df_drv (pandas.DataFrame): DataFrame containing the derivative and pressure data.
    - rate (list): List of rates preceeding to the selected transient
    - ref (int): Reference index.
    - r_list (list): List of rates.

    Returns:
    - df_derivative (pandas.DataFrame): DataFrame containing the normalized 
      derivative and normalized pressure.
    """   
    # Normalization calculations
    base_rate = r_list[ref]
    ind = find_no0_index_from_bottom(base_rate)
    qr = base_rate[ind]

    # to ensure getting a value for q[-2] if it is not available 
    qr_prev = base_rate[ind-1] if abs(ind-1) <= len(base_rate) else 0

    q = rate[find_no0_index_from_bottom(rate)]
    
    df_drv = df_drv.copy()
    
    if len(rate) > 1:
        if rate[-1] != rate[-2]:
            rate_difference = rate[-1] - rate[-2]
        else:
            if rate[-1] !=0:
                rate_difference = rate[-1]
            else:
                rate_difference = rate[find_no0_index_from_bottom(rate)]  # Add a default case if needed
    else:
        rate_difference = rate[-1]  # Handle when there's only one element

    # Calculate 'pn_saphir' and 'pn_autowell' using the determined rate difference
    df_drv['pn_saphir'] = df_drv['Delta_Pressure'] * abs(qr / rate_difference)
    df_drv['pn_autowell'] = df_drv['Delta_Pressure'] * abs((qr - qr_prev) / rate_difference)

    # Calculate 'derivative_saphir' and 'derivative_autowell' similarly
    df_drv['derivative_saphir'] = df_drv['Derivative'] * abs(qr / q)
    df_drv['derivative_autowell'] = df_drv['Derivative'] * abs((qr - qr_prev) / q)

    return df_drv


def normalize_to_ref_transient(loglog_dict, ref, saphir=True):
    """
    Normalizes derivative and pressure to the selected reference transient

    Parameters
    ----------
    loglog_dict : dict 
        The dict. is produced by loglog_calculate callback below. 
        each item has the following keys:
        'q': float
            characteristic rate. 
            q = the last non-zero prior rate for shut-in period  
            q = current flowing rate for flowing period

        'q_': float  
            q_ = 0  for shut-in period
            q_ = prior rate for flowing period
        
        'dt_dp_drv' : pd.DataFrame  or dict
            raw transient 'Delta_Time', 'Delta_Pressure' and 'Derivative' columns
            or its dict (i.e. like df.to_dict('list') or df.to_dict('records'))

    ref :  str, int, None
        key of loglog_dict pointing at the reference transient

    saphir : bool
        triggers Saphir-like normalization, otherwise the AW? method is employed
        (see the code)
    
    Returns
    -------
    dict each item of which represents pd.DataFrame with normalized dp and drv.
    """

    q_ref = loglog_dict[ref]['q']
    q_ref_ = loglog_dict[ref].get('q_',0)

    norm_loglog_dict = {}

    for k,v in loglog_dict.items():

        if isinstance(v['dt_dp_drv'],pd.DataFrame):
            df = v['dt_dp_drv'].copy()
        elif isinstance(v['dt_dp_drv'],dict):
            df = pd.DataFrame(data=v['dt_dp_drv'])
        else:
            raise ValueError(f'transient {k}: unknown dt_dp_drv type')
        
        q  = loglog_dict[k]['q']
        q_ = loglog_dict[k].get('q_',0)  
        norm_loglog_dict[k] = df

        if saphir:
            df['Delta_Pressure'] *= abs(q_ref/(q - q_))
            df['Derivative'] *= abs(q_ref/ q)
            # print(f'v={v}, ref={ref}')
            # print(df)
        else:
            df['Delta_Pressure'] *= abs((q_ref - q_ref_) / (q - q_))
            df['Derivative'] *= abs((q_ref - q_ref_) / q)

    return norm_loglog_dict



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

#--------------------------------------------

def create_vfm_dataframe(PTA_s, PTA_f, injection_breakpoints, PTA):
    """
    Create a dataframe for vfm with columns 'Time', 'top', 'bottom', and 'multirate'.
    
    Parameters:
    - PTA_s, PTA_f, injection_breakpoints, PTA: DataFrames with a 'Timestamp' column
    
    Returns:
    - df_vfm: DataFrame with vfm data
    """
    def initialize_vfm_df(source_df, active_column):
        """
        Initialize a vfm dataframe and set the specified active column to 1.
        
        Parameters:
        - source_df: DataFrame to copy the 'Timestamp' column from
        - active_column: String, the column to set to 1
        
        Returns:
        - vfm_df: DataFrame initialized with 'Time' and vfm columns
        """
        vfm_df = pd.DataFrame(columns=['Time', 'top', 'bottom', 'multirate'])
        vfm_df['Time'] = source_df['Timestamp']
        vfm_df[['top', 'bottom', 'multirate']] = 0
        vfm_df[active_column] = 1
        return vfm_df
    
    # Initialize the vfm dataframes
    df_vfm_1 = initialize_vfm_df(PTA_s, 'top')
    df_vfm_2 = initialize_vfm_df(PTA_f, 'bottom')
    df_vfm_3 = initialize_vfm_df(injection_breakpoints, 'multirate')
    
    # Concatenate the dataframes
    df_vfm = pd.concat([df_vfm_1, df_vfm_2, df_vfm_3], ignore_index=True)
    
    # Add additional rows
    df_vfm.loc[-1] = [PTA.iloc[0]['Timestamp'], 0, 1, 0]
    df_vfm.loc[-2] = [PTA.iloc[-1]['Timestamp'], 1, 0, 0]
    
    # Sort the dataframe by column 'Time'
    df_vfm = df_vfm.sort_values(by=['Time']).reset_index(drop=True)
    
    return df_vfm



def calculate_median_rate(rate_data, breakpoints):
    """
    Calculate the median rate between breakpoints.

    Parameters:
    - rate_data: DataFrame with rate and timestamps.
    - breakpoints: DataFrame with breakpoints timestamps.

    Returns:
    - A DataFrame with median rates between breakpoints.
    """
    median_rates = []

    for i in range(len(breakpoints) - 1):
        start_time = breakpoints.iloc[i]['Timestamp']
        end_time = breakpoints.iloc[i + 1]['Timestamp']
        start = breakpoints.iloc[i]['Time']
        end = breakpoints.iloc[i + 1]['Time']

        mask = (rate_data['Timestamp'] >= start_time) & (rate_data['Timestamp'] <= end_time)
        interval_data = rate_data[mask].reset_index(drop=True)

        median_rate = interval_data['Rate'].median()

        # Replace NaN values in median_rate with 0
        if pd.isna(median_rate):
            median_rate = 0

        median_rates.append({
            'Start Timestamp': start_time,
            'End Timestamp': end_time,
            'Start Time': start,
            'End Time': end,
            'Median Rate': median_rate,
        })

    median_rates_df = pd.DataFrame(median_rates)
    median_rates_df['Start Time'] = median_rates_df['Start Time'] - median_rates_df['Start Time'].iloc[0]
    
    return median_rates_df

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
    weighted_averaged_rates = []
    if not shutin_threshold:
        shutin_threshold = zero_q_frac * rate_data['Rate'].abs().quantile(0.9)

    for i in range(len(breakpoints) - 1):
        start_time = breakpoints.iloc[i]['Timestamp']
        end_time = breakpoints.iloc[i + 1]['Timestamp']
        start = breakpoints.iloc[i]['Time']
        end = breakpoints.iloc[i + 1]['Time']

        mask = (rate_data['Timestamp'] >= start_time) & (rate_data['Timestamp'] < end_time)
        interval_data = rate_data[mask]

        if not interval_data.empty:
            timestamps = interval_data['Timestamp'].to_numpy()
            rates = interval_data['Rate'].to_numpy()

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

            half_time_diffs = 0.5*time_diffs
            dt = np.concatenate(([first_time_diff], half_time_diffs)) + \
                 np.concatenate((half_time_diffs, [last_time_diff]))
            weighted_avg_rate = np.average(rates, weights=dt)
        else:
            weighted_avg_rate = 0
        if abs(weighted_avg_rate) < shutin_threshold:
            weighted_avg_rate = 0

        weighted_averaged_rates.append({
            'Start Timestamp': start_time,
            'End Timestamp': end_time,
            'Start Time': start,
            'End Time': end,
            'Weighted Averaged Rate': weighted_avg_rate
        })

    weighted_averaged_rates_df = pd.DataFrame(weighted_averaged_rates)
    weighted_averaged_rates_df['Start Time'] = \
        weighted_averaged_rates_df['Start Time'] - \
            weighted_averaged_rates_df['Start Time'].iloc[0]
    weighted_averaged_rates_df['End Time'] = \
        weighted_averaged_rates_df['End Time'] - \
            weighted_averaged_rates_df['Start Time'].iloc[0]
    return weighted_averaged_rates_df


def rate_qaqc(rate, rebuilt_rate):
    """
        Performs QAQC check on rebuilt rate.

        Parameters:
        - rate: DataFrame with rate and timestamps (output of ti.to_dateframe function).
        - rebuilt_rate: DataFrame with weighted averaged rates (output of ti.calculate_weighted_averaged_rate function).

        Returns:
        - rate_sum: Actual - sum(rate[i]*time_diffs[i]).
        - rebuilt_rate_sum: Averaged - sum(rebuilt_rate[i]*time_diffs[i])
        - difference: Averaged / Actual - 1
    """
    rate_time_diffs = rate['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
    rate_shift = rate['Rate'].shift().fillna(0)

    rebuilt_rate_time_diffs = rebuilt_rate['Start Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
    rebuilt_rate_shift = rebuilt_rate['Weighted Averaged Rate'].shift().fillna(0)
    last_rebuilt_rate = rebuilt_rate.iloc[-1]['Weighted Averaged Rate']
    last_time_diff = (rebuilt_rate.iloc[-1]['End Timestamp'] - rebuilt_rate.iloc[-1]['Start Timestamp']).total_seconds()

    rate_sum = (rate_shift * rate_time_diffs).sum()
    rebuilt_rate_sum = (rebuilt_rate_shift * rebuilt_rate_time_diffs).sum() + (last_rebuilt_rate * last_time_diff)

    difference = rebuilt_rate_sum / rate_sum - 1
    print(f'injection difference is {difference}')

    return rate_sum, rebuilt_rate_sum, difference


