import numpy as np
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