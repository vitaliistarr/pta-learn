import numpy as np
import pandas as pd



def find_no0_index_from_bottom(arr):
    """
    Finds the index of the first non-zero element from the bottom of the array.
    This is for the purpose for cal Bourdet Derivative for Shut-in transient, because reference rate is the last non-zero rate value before shutin.


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




