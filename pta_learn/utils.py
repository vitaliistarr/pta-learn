import os
import numpy as np
import pandas as pd
from numba import jit

# Static methods
@jit(nopython=True, cache=True)
def get_window_frame_logic(window_number, time, window_step, window_length):
    len_time = len(time)
    last_time = time[-1]

    window_start = time[0] + window_step * (window_number - 1)
    window_end = window_start + window_length

    # Vectorized boundary detection
    i = np.searchsorted(time, window_start, side='right') - 1
    j = np.searchsorted(time[i:], window_end, side='right') + i

    # Handle edge case by checking against check_size
    check_size = last_time - window_step
    if j < len_time:
        last_in_window = time[j - 1]
    else:
        last_in_window = last_time

    if last_in_window > check_size:
        j = len_time

    return i, j, window_start, window_end

@jit(nopython=True, cache=True)
def slope_line(x, y, slope):

    # Calculate the means of x and y values
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    b = y_mean - slope * x_mean
    y_synth = slope * x + b

    return y_synth


def round_timeindex(s):
    s = s.copy()
    s.index = s.index.round('s')
    return s


def series_to_aw_csv(s, paw='', filename=None, **kwargs):
    '''Saves series in the csv format that supports import into AW dataset:
    1. creates a header with a dictionary with metadata, like:
    "# {'well': 'W1', 'label': 'BHP', ...}"
    2. writes the data
    Parameters
    -----------
    s : pd.Series
        series with timeindex

    paw : str
        path to _aw folder

    filename : str
        should start with "_".
        default: "_well_label.csv"

    kwargs - str descriptors from _info.csv:
    * well (default: "DUMMY")
    * label (default: s.name)
    * long_label
    etc.
    '''
    kwargs['well'] = kwargs.get('well', 'DUMMY')
    kwargs['label'] = kwargs.get('label', s.name)
    if 'long_label' in kwargs:
        kwargs['long label'] = kwargs.pop('long_label')

    well, label = kwargs.get('well'), kwargs.get('label')
    if filename is None: filename = f'_{well}_{label}.csv'
    p = filename if paw == '' else f'{paw}/{filename}'

    with open(p, 'w') as f:
        f.write(f"# {kwargs}\n")

    if p[-4:] != '.csv': p += '.csv'
    s.to_csv(p, mode='a', header=False)
    print(f'"{p}" saved')


def fetch_vector_light(p, t1, t2):

    if not os.path.exists(p):
        print(f'File not found: {p}')
        return None

    if p[-3:] == 'pkl':
        s = pd.read_pickle(p)
    elif p[-3:] == 'csv':
        s = pd.read_csv(p) # tailor to the specific csv-format
    else:
        raise ValueError('Unknown format')

    if (t1 is not None) & (t2 is not None):
        s = s[t1:t2]
    elif (t1 is not None) & (t2 is None):
        s = s[t1:]
    elif (t1 is None) & (t2 is not None):
        s = s[:t2]
    else:
        pass

    return s