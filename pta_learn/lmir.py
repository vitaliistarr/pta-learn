import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


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
    3. Filters the identified minima by time.

    Parameters:
    -----------
    test_df : DataFrame
        A pandas DataFrame containing the pressure data to be analyzed.
    order : int, optional
        The order parameter used in identifying local minima.
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

    # Default value for `day`. If data is less than 10 days, it's considered a step rate test; otherwise, a long-term test.
    order = 100 if order is None else order
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