import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from kneebow.rotor import Rotor


# Functions inside TPMR
def detect_bottombp(df_bhp, prominence_value):
    """
    This function detects the bottom Breakpoints in a pressure data.
    
    Parameters:
    df_bhp (pd.DataFrame): The input data containing the necessary columns from to_dataframe.
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
                # eventhough this is not used in good quality data,but it is useful for some extreme cases
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
# Set the default value for p if it is not provided, this is the rule of thumb from real data
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
        A pressure threshold or parameter used in detecting shutin breakpoints. 
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
    """
    
    # Detect the shut-in breakpoints
    shutin_bp_all, PTA_f, PTA_s, shutin_bp_interval = detect_breakps(df_bhp, p, interval_shutin)
    
    # Detect shut-in transients without interval limitation
    shutin_transient_all = get_shutin_bps(shutin_bp_all)

    # Get shut-in transients with interval limitation, which are often shown in plots
    shutin_transient_interval = get_shutin_intervals(shutin_bp_interval)
    
    return shutin_bp_all, shutin_bp_interval, shutin_transient_all, shutin_transient_interval
