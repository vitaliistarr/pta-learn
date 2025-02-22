import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.dates as mdates
import matplotlib.cm as cm
import os


#### plot functions ####

def plot_whole(df_bhp, df_rate):
    """
    Plots BHP (Bottom Hole Pressure) and Rate data against Time.

    Parameters:
    - df_bhp (pd.DataFrame): DataFrame containing 'Time' and 'Pressure' columns.
    - df_rate (pd.DataFrame): DataFrame containing 'Time' and 'Rate' columns.
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))

    # Plot BHP data
    ax1.scatter(df_bhp['Time'], df_bhp['Pressure'], s=5, c='r', label='Pressure')
    ax1.set_ylabel('Pressure [bar]')
    ax1.set_xlabel('Time [hr]')

    # Plot Rate data
    ax2.scatter(df_rate['Time'], df_rate['Rate'], s=5, c='b', label='Rate')
    ax2.set_ylabel('Rate [STM3/D]')
    ax2.set_xlabel('Time [hr]')

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.3)

    # Show plot
    plt.show()


def plot_target(df_bhp, df_rate, sel_shutin, sel_flowing):
    """
    Plots BHP (Bottom Hole Pressure) and Rate data with highlighted shut-in and flowing intervals.

    Parameters:
    - df_bhp: DataFrame containing 'Time' and 'Pressure' columns.
    - df_rate: DataFrame containing 'Time' and 'Rate' columns.
    - sel_shutin: DataFrame containing selected shut-in intervals with 'start/hr' and 'end/hr' columns.
    - sel_flowing: DataFrame containing selected flowing intervals with 'start/hr' and 'end/hr' columns.
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 12))

    # Plot BHP and Rate data
    ax1.scatter(df_bhp['Time'], df_bhp['Pressure'], s=1, c='r', label='Pressure')
    ax2.scatter(df_rate['Time'], df_rate['Rate'], s=1, c='b', label='Rate')

    # Plot shut-in and flowing intervals
    for ax, intervals, color, label in [(ax1, sel_shutin, 'green', 'Shut-in'), 
                                        (ax2, sel_shutin, 'green', 'Shut-in'),
                                        (ax1, sel_flowing, 'blue', 'Flowing'),
                                        (ax2, sel_flowing, 'blue', 'Flowing')]:
        for i, (start, end) in intervals[['start/hr', 'end/hr']].iterrows():
            ax.axvspan(start, end, alpha=0.2, color=color, label=label if i == 0 else "")

    # Add legends
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    # Set axis labels
    ax1.set_ylabel('Pressure [bar]')
    ax1.set_xlabel('Time [hr]')
    ax2.set_ylabel('Rate [STM3/D]')
    ax2.set_xlabel('Time [hr]')

    # Adjust global font settings (optional, move outside if used globally)
    plt.rcParams.update({'font.size': 35, 'font.family': 'Calibri'})

    # Adjust space between subplots
    plt.subplots_adjust(hspace=0.4)

    # Show the plot
    plt.show()


def plot_TI_family(*logs):
    # Create the figure and axes objects
    fig, ax = plt.subplots(figsize=(24, 12))
    dot_size = 40

    # Use a colormap
    num_logs = len(logs)
    cmap = cm.get_cmap('viridis')

    # Iterate over the logs and plot them
    for i, log in enumerate(logs):
        # Calculate the color index based on the position in the log list
        color = cmap(i / (num_logs - 1))  # Normalize index to use full range of colormap
        # Plotting 'pn_saphir' with circle markers
        ax.scatter(log.Delta_Time, log.pn_saphir, s=dot_size, marker='o',
                   label=f'Pressure Transient#{i+1}', color=color)
        # # Plotting 'derivative_saphir' with cross markers
        ax.scatter(log.Delta_Time, log.derivative_saphir, s=dot_size, marker='x',
                   label=f'Derivative Transient#{i+1}', color=color)

    # Set the x and y scales to log
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Set the x and y axis labels and adjust font size
    ax.set_xlabel('Time [hr]', fontsize=20)
    ax.set_ylabel('Pressure & Derivative [bar]', fontsize=20)

    # Set font size for ticks
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Turn on the grid
    ax.grid(True, which='both', ls='-', color='0.65')

    # Put the legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., markerscale=5, fontsize=16)

    # Adjust layout to not cut off labels
    plt.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

    # Show the plot
    plt.show()

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
    
    return df_rate




def preprocess_and_save_data(df_bhp, df_rate, data_path='./data', start_time=None, end_time=None, save_to_csv=False):
    """
    Preprocesses BHP (Bottom Hole Pressure) and rate data, with an option to save the processed data to CSV files.

    This function performs the following steps:
    1. Filters the data to include only rows where the 'Time' column is between `start_time` and `end_time`.
       - If `start_time` is not provided, it defaults to the maximum of the minimum times in `df_bhp` and `df_rate`.
       - If `end_time` is not provided, it defaults to the minimum of the maximum times in `df_bhp` and `df_rate`.
    2. Resets the time to start from zero by subtracting the initial time value.
    3. Optionally saves the processed BHP and rate data to CSV files in the specified `data_path`.

    Parameters:
    - df_bhp (pd.DataFrame): DataFrame containing BHP data with a 'Time' column.
    - df_rate (pd.DataFrame): DataFrame containing rate data with a 'Time' column.
    - data_path (str, optional): Directory path where the CSV files will be saved. Defaults to './data'.
    - start_time (float or int, optional): Start time threshold. Only rows with 'Time' >= `start_time` are retained.
                                          If not provided, defaults to the maximum of the minimum times in `df_bhp` and `df_rate`.
    - end_time (float or int, optional): End time threshold. Only rows with 'Time' <= `end_time` are retained.
                                         If not provided, defaults to the minimum of the maximum times in `df_bhp` and `df_rate`.
    - save_to_csv (bool, optional): Whether to save the processed data to CSV files. Defaults to False.

    Returns:
    - pressure (pd.DataFrame): Processed BHP data.
    - rate (pd.DataFrame): Processed rate data.
    """
    # Set start_time to the maximum of the initial times in df_bhp and df_rate
    if start_time is None:
        start_time = max(df_bhp['Time'].min(), df_rate['Time'].min())
    
    # Set end_time to the minimum of the final times in df_bhp and df_rate
    if end_time is None:
        end_time = min(df_bhp['Time'].max(), df_rate['Time'].max())
    
    # Filter data to include only rows where Time is between start_time and end_time
    df_bhp = df_bhp[(df_bhp['Time'] >= start_time) & (df_bhp['Time'] <= end_time)].reset_index(drop=True)
    df_rate = df_rate[(df_rate['Time'] >= start_time) & (df_rate['Time'] <= end_time)].reset_index(drop=True)

    # Reset time to start from zero
    df_bhp['Time'] = df_bhp['Time'] - df_bhp['Time'].iloc[0]
    df_rate['Time'] = df_rate['Time'] - df_rate['Time'].iloc[0]

    # Copy data to new DataFrames
    pressure = df_bhp.copy()
    rate = df_rate.copy()

    # Save processed data to CSV files if save_to_csv is True
    if save_to_csv:
        os.makedirs(data_path, exist_ok=True)  # Create directory if it doesn't exist
        pressure.to_csv(f"{data_path}/pressure.csv")
        rate.to_csv(f"{data_path}/rate.csv")

    return pressure, rate



