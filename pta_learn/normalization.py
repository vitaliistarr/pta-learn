import numpy as np
import pandas as pd

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