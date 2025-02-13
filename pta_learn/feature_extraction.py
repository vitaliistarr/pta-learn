import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import TheilSenRegressor, LinearRegression
from skopt import gp_minimize
from skopt.space import Real as Real_value
from numba import jit

from .utils import get_window_frame_logic, slope_line

class Logtime_window:
    """
    Create Log time window object for rolling window calculations.

    This class allows to perform rolling window calculations.
    The class provides indices and data values of the selected window in a Log-Log response.

    Attributes
    --------
    time: np.array
        Log time values.
    pressure: np.array
        Log pressure derivative values.
    window_length: float
        The size of window in log time units.
    window_step: float
        The size of step in log time units.
    number: int
        Total number of windows in a log-log response. Calculated by the class while initialized.

    Methods
    --------
    number_of_windows():
        Calculate the total number of windows.
    check_window_length():
        Check window_length parameter.
    get_window_frame(window_number):
        Get window indices for a selected window number.
    get_window(window_number):
        Get data in the selected window.
    """
    def __init__(
            self,
            time: np.array,
            pressure: np.array,
            window_length: float,
            window_step: float
    ):
        """
        Initialize the class.

        Parameters
        --------
        time: np.array
            Log time values.
        pressure: np.array
            Log pressure derivative values.
        window_length: float
            The size of window in log time units.
        window_step: float
            The size of step in log time units.
        """
        self.time = time
        self.pressure = pressure
        self.window_length = window_length
        self.window_step = window_step
        self.check_window_length()
        self.number = self.number_of_windows()

        # Compile get_window_frame_logic function in numba
        i,j,start,end = get_window_frame_logic(window_number=5,
                                               time=np.log10(np.arange(1,100)),
                                               window_step=0.05,
                                               window_length=0.5)
        if i > 0:
            print('Function compiled')

    def number_of_windows(self):
        """
        Calculate the total number of windows.

        This function calculates the total number of windows based window length and window step.

        Returns
        --------
        number_of_windows: int
            Total number of windows in a log-log response.
        """
        # If window length covers the transient - return number_of_window = 1
        if self.time[-1] - self.time[0] < self.window_length:
            number_of_windows = 1
        else:
            # '+2' because window range starts from zero + last shorter window with missing values
            number_of_windows = (self.time[-1] - self.window_length - self.time[0]) / self.window_step
            number_of_windows = int(number_of_windows) + 2  # Round to upper int.

        return number_of_windows

    def check_window_length(self):
        """
        Check window_length parameter.

        This function checks if the passed window_length parameter covers at least 2 data points.
        In order to construct a synthetic slope and calculate the distance.

        Returns
        --------
        ValueError if the condition wasn't passed.
        """
        if len(self.time) < 2:
            raise ValueError(f'Data can not be interpreted. '
                             f'Size of data passed is {len(self.time)}')
        diff = np.diff(self.time)
        if self.window_length < diff.max():
            raise ValueError(f'window {self.window_length} is too short. '
                             f'Should be higher than minimum resolution {diff.max()}')

    def get_window_frame(self, window_number):
        """
        Get window indices for a selected window number.

        This function retrieves indices of the first and last points in a selected window.
        Further, pressure/derivative/time data can be extracted using these indices.

        Parameters
        --------
        window_number: int
            Number of the selected window.

        Returns
        --------
        i: int
            First index.
        j: int
            Last index.
        window_start: float
            First log time value in the window.
        window_end: float
            Last log time value in the window.
        """
        # Check window number
        if window_number > self.number:
            raise ValueError(f'Irrelevant window number: {window_number}. '
                             f'There are {self.number} windows total. '
                             f'Choose any of (0, {self.number - 1}) range. ')

        i, j, window_start, window_end = get_window_frame_logic(window_number=window_number,
                                                                time=self.time,
                                                                window_step=self.window_step,
                                                                window_length=self.window_length)

        x_window, y_window = self.time[i:j], self.pressure[i:j]

        return i, j, window_start, window_end, x_window, y_window


class PTAClassifier(Logtime_window):
    """
    Create PTAClassifier object for a singe pressure transient analysis.

    This class allows to perform single transient analysis, based on the time, pressure and pressure derivative data.
    This class inherits functions from a parent Logtime_window class to perform rolling window calculations.

    Attributes
    --------
    window_length: float
        The size of window in log time units.
    window_step: float
        The size of step in log time units.
    metric: str
        Similarity measure to use. Available: norm. euclidean, cosine, frechet and hausdorff measures.
    filter_value: float
        Parameter to filter_out Wellbore storage flow regime, mostly dominated by wellbore dynamics
        and not suitable for rolling window-based analysis.
    fragment_window: float
        Parameter that defines the minimum length of flow regime to detect (in log time scale).
    slope_threshold: float
        Slope threshold for Boundary flow regime detection.
    dsw: int
        Data smoothing window. The window, which is used to smooth a pressure derivative for a more accurate analysis.
        dsw is integer value, which is measured by number of points to consider.
    rsw: float
        Regime smoothin window. The window, which is used to smooth short flow regimes in a processed transient.
        Optional parameter.
    smooth: boolean
        If True, smooths short flow regimes in a processed transient, based on rsw attribute.

    Methods
    --------
    fit():
        Fit the PTAClassifier object.
    predict():
        Predict flow regimes.
    fit_predict():
        Fit and predict as a one function.
    predict_optimize():
        Optimize parameters and predict flow regimes.
    get_params():
        Get the current PTAClassifier parameters (commonly used after optimization step).
    to_csv():
        Export results to csv format.
    to_excel():
        Export results to excel format.
    to_pickle():
        Export results to pickle format.
    """
    def __init__(
            self,
            window_length=0.5,
            window_step=0.02,
            metric='euclidean',
            filter_value=0,
            fragment_window=0.5,
            slope_threshold=0.7,
            dsw=25,
            rsw=0.15,
            bdw=10,
            smooth=False,
    ):
        ''' User input data'''
        self.window_length = window_length
        self.window_step = window_step
        self.filter_value = self.check_get_filter_value(filter_value)
        self.fragment_window = self.check_get_fragment_window(fragment_window)
        self.slope_threshold = slope_threshold
        self.smooth = smooth

        ''' Program included data'''
        self.metrics = ['euclidean', 'cosine', 'frechet', 'hausdorff']
        self.functions = ['Zero slope', 'Linear-up 1', 'Linear-up 2', 'Linear-down 1', 'Linear-down 2']
        self.boundary = 5
        self.wbs = 6
        self.slope = [0, 0.176, 0.364, -0.176, -0.364]  # 0, 10, 20, -10, -20
        # 10deg = 0.176; 20deg = 0.364; 30deg = 0.57; 40deg = 0.84; 50deg = 1.2; 35deg = 0.7; 55deg = 1.5
        self.names = ['Radial', 'Linear-up', 'Linear-down', 'Boundary', 'WBS']
        self.color = ['green', 'maroon', 'orange', 'navy', 'blue']
        self.output_data = pd.DataFrame()
        self.distance_data = pd.DataFrame()
        self.clf_fitted = False
        self.output_compiled = False

        ''' Check if everything is correct '''
        self.metric = self.check_get_metric(metric=metric)          # Check the metric passed
        self.dsw = self.check_get_data_smoothing_window(dsw)        # Check the data smoothing window passed
        self.rsw = self.check_get_regime_smoothing_window(rsw)      # Check the regime smoothing window passed
        self.bdw = self.check_get_boundary_detection_window(bdw)    # Check the boundary detection window passed

        # Compile get_window_frame_logic function in numba
        _,_,_,_ = get_window_frame_logic(window_number=5,
                                                  time=np.log10(np.arange(1, 100)),
                                                  window_step=0.05,
                                                  window_length=0.5)

        _ = slope_line(np.log10(np.arange(1, 100)), np.log10(np.arange(1, 100)), 1)


    def check_get_metric(self, metric):
        """
        Check if the passed metric is supported by the model

        Parameters
        --------
        metric: str
            Metric to be checked.

        Returns
        --------
        metric: str
            Output metric if correct.
        """
        if metric not in self.metrics:
            raise NameError(f'Unknown metric: {metric}. '
                            f'Choose any of the following: {self.metrics}')
        else:
            return metric

    def check_get_data_smoothing_window(self, data_smoothing_window):
        """
        Check if the passed data smoothing window (dsw) is adequate.

        The function checks if the passed data_smoothing_window is odd and is in [0, 50) range.

        Parameters
        --------
        data_smoothing_window: int
            Smoothing window to be checked.

        Returns
        --------
        data_smoothing_window: int
            Smoothing window if correct.
        """
        if isinstance(data_smoothing_window, int):
            if (data_smoothing_window > 0) and (data_smoothing_window <= 50):
                if not data_smoothing_window % 2 == 0:
                    return data_smoothing_window
                else:
                    raise ValueError(f'Data smoothing window should be odd, integer and > 0. '
                                     f'Incorrect value passed: {data_smoothing_window}')
            elif data_smoothing_window <= 0:
                raise ValueError(f'Data smoothing window should odd, integer and > 0. '
                                 f'Incorrect value passed: {data_smoothing_window}')
            elif data_smoothing_window > 50:
                raise ValueError(f'Data smoothing window passed is too big: {data_smoothing_window}.')
        else:
            TypeError(f'Expected data smoothing window format: int. '
                      f'The given type: {type(data_smoothing_window)}')

    def check_get_regime_smoothing_window(self, regime_smoothing_window):
        """
        Check if the passed regime smoothing window (rsw) is adequate.

        The function checks if the passed regime_smoothing_window is at least 0.1 in log time units.

        Parameters
        --------
        regime_smoothing_window: float
            Smoothing window to be checked.

        Returns
        --------
        regime_smoothing_window: float
            Smoothing window if correct.
        """
        if regime_smoothing_window >= 0.1:
            return regime_smoothing_window
        else:
            raise ValueError(f'Regime smoothing window should be at least higher than 0.1 LogTime. '
                             f'Value passed: {regime_smoothing_window}')

    def check_get_boundary_detection_window(self, boundary_detection_window):
        """
        Check if the passed boundary detection window (bdw) is adequate.

        The function checks if the passed boundary_detection_window is int and >= 2

        Parameters
        --------
        boundary_detection_window: int
            Boundary detection window to be checked.

        Returns
        --------
        boundary_detection_window: float
            Boundary detection window if correct.
        """
        if isinstance(boundary_detection_window, int):
            if boundary_detection_window >= 2:
                return boundary_detection_window
            else:
                raise ValueError(f'Boundary detection window should be greater or at least equal 2. '
                                 f'The given value: {boundary_detection_window}')
        else:
            TypeError(f'Expected boundary detection window type: int. '
                      f'The given type is: {type(boundary_detection_window)}')

    def check_get_filter_value(self, filter_value):
        """
        Check if the passed filter value is adequate.

        The function checks if the passed filter_value is in a recommended log time range [-1.5, 1].

        Parameters
        --------
        filter_value: float
            Filter value to be checked.

        Returns
        --------
        filter_value: float
            Filter value if correct.
        """
        if (filter_value >= -1.5) and (filter_value <= 1):
            return filter_value
        else:
            raise ValueError(f'Recommended range for filter value is [-1.5, 1]. '
                             f'Filter value passed: {filter_value}')

    def check_get_fragment_window(self, fragment_window):
        """
        Check if the passed fragment window is adequate.

        The function checks if the passed fragment_window is in a recommended log time range [0.3, 1].

        Parameters
        --------
        fragment_window: float
            Fragment window to be checked.

        Returns
        --------
        fragment_window: float
            Fragment window if correct.
        """
        if (fragment_window >= 0.3) and (fragment_window <= 1):
            return fragment_window
        else:
            raise ValueError(f'Recommended range for fragment window is [0.3, 1]. '
                             f'Fragment window passed: {fragment_window}')

    def fragmentation_check_is_reasonable(self):
        """
        Check if the fragmentation check of an output regime array is reasonable.

        First, this function checks if self.time attribute has at least 2 points.
        Second, checks if the duration of the pressure transient is longer than fragment_window.

        Returns
        --------
        Condition: True or False.
        """
        if len(self.time) >= 2:
            aft_length = self.time[-1] - self.time[0]  # Length in log time
            if aft_length > self.fragment_window:
                return True
            else:
                return False
        else:
            return False

    def fragmentation_check(self, regime):
        """
        Check the output regime array for fragmented flow regimes.

        The function check if flow regimes shorter than fragment_window are present in the output regime array.

        Parameters
        --------
        regime: np.array
            Regime array to be checked.

        Returns
        --------
        indices: list
            Indices of the fragmented flow regimes detected. [] if nothing detected.
        first_established_regime: int
            First flow regime detected.
        """
        first_established_regime = None
        indices = []
        if self.fragmentation_check_is_reasonable():  # perform check if needed
            first_established_regime = regime[0]
            current_period = [0]  # start with the first index as the current period
            for i in range(1, len(regime)):
                if regime[i] == regime[i - 1]:  # part of the same period
                    current_period.append(i)
                else:  # Check if the period is shorter than smoothing window
                    if abs(self.time[current_period[-1]] - self.time[current_period[0]]) <= self.fragment_window:
                        for index in current_period:
                            indices.append(index)
                    current_period = [i]  # start of a new period
            # Check the last period
            if abs(self.time[current_period[-1]] - self.time[current_period[0]]) <= self.fragment_window:
                for index in current_period:
                    indices.append(index)
        else:
            indices = [1]

        return indices, first_established_regime

    def get_data_for_detection(self):
        """
        Get data for window-based detection.

        The function takes data points which can be analyzed with window-based detection.
        The function filters out WBS regime with filter_value and Boundary regime if detected.
        Relevant data is being passed to self.time and self.pressure attributes of PTAClassifier class.
        Data is being smoothed with data_smoothing() method.

        See also
        ________
        detect_boundary: Detects boundary regime aftering the data is fitted.
        data_smoothing: Applies savitsky-golay smoothing filter.
        """
        # Data not including boundary
        self.time = self.time_log_scale[self.inliers == 1]
        self.pressure = self.pressure_derivative_log_scale[self.inliers == 1]
        # Get rid of initial WBS
        self.pressure = self.pressure[self.time >= self.filter_value]
        self.time = self.time[self.time >= self.filter_value]

        # Smooth pressure data
        self.time, self.pressure = self.data_smoothing(self.time, self.pressure)
        # Regime detection flag
        self.regime_detection_possible = True if len(self.time) >= 2 else False

    def get_min_window(self):
        """
        Get minimum window possible.

        This function retakes inliers with get_inliers() method to ensure processing of the relevant self.time.
        Minimum window is calculated based on log time resolution and window step.

        Returns
        --------
        min_window_length: float
            Minimum window possible.
        """
        self.get_data_for_detection()
        if len(self.time) < 2:
            raise ValueError(f'Data can not be interpreted. '
                             f'Size of data passed is {len(self.time)}')
        else:
            diff = np.diff(self.time)
            min_window_length = diff.max() + 3*self.window_step

        return min_window_length

    def combine_regimes(self, regime, add_wbs_boundary=True):
        """
        Combine regimes into the selected flow regime classification.

        This function combines regimes detected with different reference slopes into the selected classification.
        If add_wbs_boundary flag is True, the function adds WBS and Boundary regimes,
        which can't be detected by window-based detection.

        Parameters
        --------
        regime: np.array
            Regime array to be processed.
        add_wbs_boundary: boolean
            If True - add WBS and Boundary to regime array. Usually done before plotting.

        Returns
        --------
        r: np.array
            Combined and processed regime array.
        """
        r = regime
        for i in range(len(r)):
            # Linear-up
            if r[i] == 2:
                r[i] = 1
            # Linear-down
            elif r[i] == 3:
                r[i] = 2
            elif r[i] == 4:
                r[i] = 2
            # Boundary from smoothing
            elif r[i] == 5:
                r[i] = 3

        if add_wbs_boundary:
            # Add WBS before Filter value
            head = self.time_log_scale[self.time_log_scale < self.filter_value]
            wbs = np.ones(shape=len(head)).astype('int')
            wbs[:] = 4
            r = np.concatenate((wbs, r))

            # Add short boundary period at the end (if present)
            tail = self.time_log_scale[self.inliers == 2]
            boundary = np.ones(shape=len(tail)).astype('int')
            boundary[:] = 3
            r = np.concatenate((r, boundary))

        return r

    def data_smoothing(self, time, pressure):
        """
        Smooth data with Savgol filter.

        This function smooths time and pressure derivative data to improve window-based detection.
        The function takes time and pressure derivative in log-log scale.
        The function is used in get_inliers() function for window-based detection
        and in detect_boundary() function for Bourdary regime detection.

        Parameters
        --------
        time: np.array
            Time array in log scale.
        pressure: np.array
            Pressure derivative array in log scale.

        Returns
        --------
        time: np.array
            Processed time array in log scale.
        pressure: np.array
            Processed pressure derivative array in log scale.
        """
        if len(pressure) < 100:
            if len(pressure) < 30:
                return time, pressure
            self.dsw = 5
            pressure = savgol_filter(pressure, self.dsw, 2)
        else:
            pressure = savgol_filter(pressure, self.dsw, 2)

        return time, pressure

    def data_resampling(self, time, pressure, derivative):
        """
        Resample input data to a log space.

        This function resample input pressure and pressure derivative data to a log space.
        The function recalculate vectors to 500 log space data points.
        Important step to ensure accurate regime detection.

        Parameters
        --------
        time: np.array
            1dim array of time values
        pressure: np.array
            1dim array of pressure values
        derivative: np.array
            1dim array of pressure derivative values

        Returns
        --------
        time_resampled: np.array
            1dim resampled time
        pressure_resampled: np.array
            1dim resampled pressure
        derivative_resampled: np.array
            1dim resampled pressure derivative
        """
        log_time = np.log10(time)
        log_time = np.nan_to_num(log_time, nan=0, neginf=0)
        time_resampled = np.logspace(min(log_time),max(log_time),500)
        pressure_func = interp1d(time, pressure, kind='linear', fill_value='extrapolate')
        derivative_func = interp1d(time, derivative, kind='linear', fill_value='extrapolate')
        pressure_resampled = pressure_func(time_resampled)
        derivative_resampled = derivative_func(time_resampled)

        def replace_invalid_values(array):
            # Replace inf and -inf with NaN, then fill NaN with nearest values
            array[np.isinf(array)] = np.nan
            array = np.nan_to_num(array, nan=np.nan)  # Ensure NaN is preserved

            # Fill NaNs by interpolating with nearest values
            not_nan = ~np.isnan(array)
            indices = np.arange(len(array))
            array[np.isnan(array)] = np.interp(indices[np.isnan(array)], indices[not_nan], array[not_nan])

            return array
        pressure_resampled = replace_invalid_values(pressure_resampled)
        derivative_resampled = replace_invalid_values(derivative_resampled)

        return time_resampled, pressure_resampled, derivative_resampled

    def regime_smoothing(self, regime):
        """
        Smooth short flow regimes in an output regime array.

        This function smooths flow regimes in an output regime array, which are shorter than a selected threshold.
        First, the function detects short flow regimes, then interpolates between longer ones.

        Parameters
        --------
        regime: np.array
            Regime array to be processed.

        Returns
        --------
        regime: np.array
            Processed regime array.
        """
        # Get indices of short periods
        indices_to_revise = []
        current_period = [0]
        for i in range(1, len(regime)):
            # part of the same period
            if regime[i] == regime[i - 1]:
                current_period.append(i)
            else:
                if abs(self.time[current_period[-1]] - self.time[current_period[0]]) <= self.rsw:
                    for index in current_period:
                        indices_to_revise.append(index)
                current_period = [i]
        if abs(self.time[current_period[-1]] - self.time[current_period[0]]) <= self.rsw:
            for index in current_period:
                indices_to_revise.append(index)

        # Assign nan to indices (that should be revised)
        regime = regime.astype('float') # convert to float values to use pandas fillna
        for itr in indices_to_revise:
            regime[itr] = np.nan

        # Interpolate, forward and back fill nan values in regime array
        s = pd.Series(regime)
        s.interpolate(method='nearest', inplace=True)
        s.fillna(method='bfill', inplace=True)
        if self.time_log_scale[-1] > self.time[-1]:  # Check if Boundary was detected
            s.fillna(self.boundary, inplace=True)
        else:
            s.fillna(method='ffill', inplace=True)
        regime = s.values
        regime = regime.astype('int')

        return regime

    def detect_boundary(self, window=10):
        """
        Detect boundary regime.

        This function detects a sharp and short slope at the end of the pressure transient.
        The function assigns result to self.inliers attribute.

        Parameters
        --------
        window: int
            Number of points to process.
        """
        t, p = self.data_smoothing(self.time_log_scale, self.pressure_derivative_log_scale)
        slope = []
        for i in range(window, len(t)):
            x = t[i - window:i]
            y = p[i - window:i]
            ts_reg = TheilSenRegressor().fit(x.reshape(-1,1), y.ravel())
            slope.append(abs(ts_reg.coef_))
        slope = np.array(slope).ravel()
        boundary = np.zeros(shape=len(slope) - window)
        for i in range(len(boundary)):
            if slope[i:i + window].mean() > self.slope_threshold:
                boundary[i] = 1
        b = 0
        for i in range(len(boundary)):
            i_rev = len(boundary) - 1 - i
            if i == 0:
                if boundary[i_rev] == 1:
                    b += 1
                else:
                    break
            else:
                if boundary[i_rev] == boundary[i_rev + 1]:
                    b += 1
                else:
                    b += window - 1  # + the rest points in the window
                    break

        if b != 0:
            self.inliers[-b:] = 2

    @staticmethod
    def distance_calculation_logic(number, functions, slope, time, pressure, window_step, window_length):
        """    Optimized distance calculation with numba compilation

        """
        # Pre-allocate output array
        dist = np.zeros((number, len(functions)))

        for n in range(number):
            i, j, _, _ = get_window_frame_logic(n + 1, time, window_step, window_length)

            time_window = time[i:j]
            pressure_window = pressure[i:j]
            window_len = len(time_window)

            s_pressures = np.array([slope_line(time_window, pressure_window, s) for s in slope])
            diffs = s_pressures - pressure_window.reshape(1, -1)
            dist[n] = np.sqrt(np.sum(diffs * diffs, axis=1)) / window_len

        return dist

    def distance_calculation(self):
        """
        Calculate distances to all reference slopes.

        This is the core function, which calculates distances to all reference slopes, which are set up in __init__().
        Distance is the selected metric, which is also set up by __init__().
        Distance is calculated for each window in a log scale.

        Returns
        --------
        dist: np.array
            2dim array of n x f shape. Where n - window, f - reference slope.
        """

        dist = self.distance_calculation_logic(number=self.number,
                                               functions=self.functions,
                                               slope=self.slope,
                                               time=self.time,
                                               pressure=self.pressure,
                                               window_step=self.window_step,
                                               window_length=self.window_length)

        return dist

    @staticmethod
    @jit(nopython=True)
    def _process_window(i, j, regime_number, cum_distances, dist_n):
        """Numba-optimized helper function to process each window"""
        for index in range(len(regime_number)):
            if i <= index < j:
                regime_number[index] += 1
                cum_distances[index] += dist_n
        return regime_number, cum_distances

    def regime_detection(self, add_wbs_boundary=False, smooth=False):
        """
        Detect flow regimes for each data point.

        This is the core function, which assign flow regime to each data point, depending on the distance metric.
        The function calls distance_calculation() function and interprets its results.
        Interpretation (assignment of a flow regime to a data point) involves calculation of the average distance
        metric for each data point. Minimum average distance indicates the detected flow regime.

        Parameters
        --------
        add_wbs_boundary: boolean
            If True, add wbs and boundary while combining regimes.
        smooth: boolean
            If True, calls regime_smoothing() function to smooth short regimes.

        Returns
        --------
        regime: np.array
            Regime array with detected flow regimes.
        avg_dist: np.array
            2dim array of NxR size (N - number of data points, R - number of regimes) with average distance metric
            after a sliding window processing.

        See also
        ________
        get_inliers: detection of data point which can be processed with window-based analysis.
        distance_calculation: Calculation of a distance metric for each log time window.
        combine_regimes: Combine window-based detected regimes into the actual classification model.
        regime_smoothing: Smooth short regimes (shorter than 0.1 log time).
        """
        self.get_data_for_detection()

        if not self.regime_detection_possible:
            # Early return for insufficient data
            regime = np.array([0] if len(self.time) == 1 else [])
            avg_dist = np.ones((1, len(self.functions)))
            if add_wbs_boundary:
                regime = self.combine_regimes(regime, add_wbs_boundary=True)
            return regime.astype('int'), avg_dist

            # Initialize arrays
        self.check_window_length()
        self.number = self.number_of_windows()
        dist = self.distance_calculation()

        n_points = len(self.time)
        n_functions = len(self.functions)
        regime_number = np.zeros((n_points, n_functions))
        cum_distances = np.zeros((n_points, n_functions))

        # Process windows in parallel using vectorized operations
        for n in range(self.number):
            i, j, _, _, _, _ = self.get_window_frame(window_number=n)
            # Use numba-optimized function for the inner loop
            regime_number, cum_distances = self._process_window(
                i, j, regime_number, cum_distances, dist[n]
            )

            # Calculate average distances efficiently
        with np.errstate(divide='ignore'):  # Suppress divide by zero warnings
            avg_dist = np.where(
                regime_number > 0,
                cum_distances / regime_number,
                np.inf
            )

            # Get regime with minimum average distance
        regime = np.argmin(avg_dist, axis=1)

        # Apply smoothing if requested
        if smooth:
            regime = self.regime_smoothing(regime)

            # Combine regimes and add boundaries if needed
        if add_wbs_boundary:
            regime = self.combine_regimes(regime, add_wbs_boundary=True)

        return regime.astype('int'), avg_dist


    def objective(self, params, min_filter_value=-0.5, max_filter_value=0.5):
        """
        Objective function for model optimization.

        This function evaluates output regime results and creates 3dim optimization surface.
        The optimum detection is found at the objective function minimum.
        Fragmented responses are penalized by the objective function by relatively high values.
        The function takes window length and filter value as input parameters.
        The function's output is based on the relatively weighted average distance and filter value.
        The function's minimum value provides the best trade-off between average distance (detection accuracy)
        and filter value.

        Parameters
        --------
        params: list
            Input parameters as a python list. First value - window length, second - filter value.
        min_filter_value: float
            Lower boundary for filter value parameter. Sets a reasonable optimization range.
        max_filter_value: float
            Higher boundary for filter value parameter. Sets a reasonable optimization range.

        Returns
        --------
        response: float
            Objective function response.

        See also
        --------
        regime_detection: flow regime assignment for each data point in a pressure transient being processed.
        fragmentation_check: check if the output of regime_detection() is fragmented (short flow regimes).
        """
        self.window_length = params[0]
        self.filter_value = params[1]

        regime, avg_dist = self.regime_detection()
        indices, first_established_regime = self.fragmentation_check(regime)

        min_dist = []
        for dist in avg_dist:
            min_dist.append(min(dist))
        min_dist = np.array(min_dist)
        min_dist = min_dist[~np.isposinf(min_dist)]

        # Exponentially weighted average
        x = np.linspace(0, 1, len(min_dist))
        weights = np.exp(-2 * x)
        avg_min_dist = np.average(min_dist, weights=weights)

        # Scale filter value to the distance distribution
        filter_scaled = self.scaler.transform([[self.filter_value]])[0][0]
        # Construct the response
        a = 0.9992
        if indices == []:
            response = a * avg_min_dist + (1 - a) * filter_scaled
        else:
            response = 0.1
        return response

    def adjust_filter_value(self, fer, fil):
        """
        Adjust filter value after bayesian optimization.

        This function adjust filter value after bayesian optimization, which could be misinterpreted.
        This function is used within bayesian_optimization() function.
        The function use quite short window length to catch the transition from WBS to the first established regime.
        Regime array should be without WBS and Boundary (combine_regimes not applied).

        Parameters
        --------
        fer: int
            First established regime.
        fil: float
            Filter value to adjust.

        Returns
        --------
        true_fil: float
            Adjusted filter value.

        See also
        --------
        regime_detection: flow regime assignment for each data point in a pressure transient being processed.
        bayesian_optimization: pre-optimizes window length and filter value parameters.
        get_min_window: calculate the minimum window possible.
        """
        self.filter_value = -1.5
        self.window_length = 0.15
        try:
            regime_ad, avg_dist_ad = self.regime_detection()
        except ValueError:
            self.window_length = self.get_min_window()
            regime_ad, avg_dist_ad = self.regime_detection()
        # print(f'Used window: {self.window_length}')

        # Get Before and After Filter Regime (using initial filter value)
        bfr = regime_ad[self.time < fil]
        afr = regime_ad[self.time >= fil]
        fer_ad = afr[0]  # first established regime

        # True filter value
        true_fil = fil

        # Propagating back from current filter value untill the regime is changed
        if fer_ad == fer:  # Only if first established regimes are the same
            for i in range(len(bfr)):
                i_rev = len(bfr) - 1 - i
                if i == 0:
                    if bfr[i_rev] == fer_ad:
                        pass
                    else:
                        true_fil = self.time[i_rev]
                        break
                else:
                    if bfr[i_rev] == bfr[i_rev + 1] and bfr[i_rev] == fer_ad:
                        pass
                    else:
                        true_fil = self.time[i_rev]
                        break

        return true_fil

    def bayesian_optimization(self, adjust_filter=True, max_window_length=1, min_filter_value=-0.5,
                              max_filter_value=0.5, deep_search=False):
        """
        Optimize window length and filter value parameters.

        This function utilizes Gaussian process to find the minimum of the objective function.
        The minimum of objective function gives the pre-optimized filter value and window length parameters.
        Further, adjust_filter_value() function is applied to adjust the filter value.

        Parameters
        --------
        adjust_filter: boolean
            Adjust filter value if needed.
        max_window_length: float
            Higher boundary for window length. Sets a reasonable optimization range.
        min_filter_value: float
            Lower boundary for filter value parameter. Sets a reasonable optimization range.
        max_filter_value: float
            Higher boundary for filter value parameter. Sets a reasonable optimization range.
        deep_search: boolean
            If True performs a deeper search for optimal filter value and window length.

        Returns
        --------
        regime: np.array
            Optimized detection of flow regimes.
        avg_dist: np.array
            2dim array of N x F shape.
            Where N - number of data measurements in a transient, F - number of flow regimes in the classification.
        res: scipy.OptimizeResult
            OptimizeResult class from scipy.optimize package.

        See also
        --------
        regime_detection: flow regime assignment for each data point in a pressure transient being processed.
        objective: The core objective function for the Bayesian optimization process.
        adjust_filter_value: adjust filter value to catch the exact transition from WBS to a first regime.
        fragmentation_check: check if the output of regime_detection() is fragmented (short flow regimes).
        get_min_window: calculate the minimum window possible.
        """
        self.scaler = MinMaxScaler(feature_range=(0.1, 1))
        self.scaler.fit([[min_filter_value], [max_filter_value]])
        self.filter_value = min_filter_value
        min_window_length = self.get_min_window()

        space = [
            Real_value(min_window_length, max_window_length, name='window_length'),
            Real_value(min_filter_value, max_filter_value, name='filter_value')
        ]
        if deep_search:
            n_calls=50
            n_initial_points=30
        else:
            n_calls=20
            n_initial_points=15
        res = gp_minimize(
            self.objective,
            space,
            n_calls=n_calls,
            random_state=0,
            n_initial_points=n_initial_points,
            acq_func='EI'
        )

        window, fil = res.x

        # Filter value adjustment
        if adjust_filter:
            try:
                self.window_length = window
                self.filter_value = fil
                regime1, _ = self.regime_detection(add_wbs_boundary=False, smooth=True)
                fer = regime1[0]
                fil = self.adjust_filter_value(fer, fil)
            except IndexError:
                pass

        # Binary search with early stopping
        self.filter_value = fil
        left = self.get_min_window()
        right = max_window_length
        tolerance = 0.01
        max_iterations = 10
        iteration = 0

        while right - left > tolerance and iteration < max_iterations:
            self.window_length = (left + right) / 2
            regime2, _ = self.regime_detection()
            indices, _ = self.fragmentation_check(regime2)

            if indices:
                left = self.window_length
            else:
                right = self.window_length
                window = self.window_length
            iteration += 1

        self.window_length = window if 'window' in locals() else res.x[0]

        # Final regime detection
        regime, avg_dist = self.regime_detection(add_wbs_boundary=True, smooth=True)

        return regime, avg_dist, res



    ''' User interface endpoints: 
        1. fit()
        2. predict()
        3. fit_predict()
        4. predict_optimize()
        5. get_params()
        6. to_csv()
        7. to_excel()
        8. to_pickle()
    '''
    def compile_output(self, regime, avg_dist):
        """
        Create compiled output.

        This function compiles the output object which could be further exported in different formats.
        The function creates self.output_data, self.fig and self.distance data objects as an output result of the model.

        Parameters
        --------
        regime: np.array
            Regime array of predicted flow regimes assigned to each data point.
        avg_dist: np.array
            2dim array of N x F shape.
            Where n - number of data measurements in a transient, F - number of flow regimes in the classification.
        """
        output_data = np.concatenate((self.time_lin_scale.reshape(-1, 1),
                                      self.pressure_lin_scale.reshape(-1, 1),
                                      self.pressure_derivative_lin_scale.reshape(-1, 1),
                                      self.time_log_scale.reshape(-1, 1),
                                      self.pressure_log_scale.reshape(-1, 1),
                                      self.pressure_derivative_log_scale.reshape(-1, 1),
                                      regime.reshape(-1, 1)), axis=1)
        output_data_columns = ['Time', 'Pressure', 'Pressure_derivative',
                               'Log_time', 'Log_pressure', 'Log_pressure_derivative',
                               'Regime']
        self.output_data = pd.DataFrame(output_data, columns=output_data_columns)
        self.distance_data = pd.DataFrame(avg_dist, columns=self.functions)
        self.output_compiled = True

    def fit(self, data):
        """
        Fit the classifier.

        This function fits the PTAClassifier object with data to analyze.

        Parameters
        --------
        data: np.array
            2dim np.array with 3 parameters (each as a column, linear scale): time, pressure, pressure derivative.
        """
        # Get linear scale data
        self.time_lin_scale = data[:, 0].ravel()
        self.pressure_lin_scale = data[:, 1].ravel()
        self.pressure_derivative_lin_scale = data[:, 2].ravel()

        # Resample data for log scale analysis
        self.time_lin_scale, self.pressure_lin_scale,\
            self.pressure_derivative_lin_scale = self.data_resampling(time=self.time_lin_scale,
                                                                      pressure=self.pressure_lin_scale,
                                                                      derivative=self.pressure_derivative_lin_scale)

        # Calculate log time and log pressure
        self.time_log_scale = np.log10(self.time_lin_scale)
        self.pressure_log_scale = np.log10(self.pressure_lin_scale)
        self.pressure_derivative_log_scale = np.log10(self.pressure_derivative_lin_scale)
        self.time_log_scale = np.nan_to_num(self.time_log_scale, nan=0, neginf=0)
        self.pressure_log_scale = np.nan_to_num(self.pressure_log_scale, nan=0, neginf=0)
        self.pressure_derivative_log_scale = np.nan_to_num(self.pressure_derivative_log_scale, nan=0, neginf=0)

        # Create inliers array and mark points (1 - inlier, 2 - boundary)
        self.inliers = np.ones(shape=len(self.time_log_scale))
        # Detect boundary right after fitting data
        self.detect_boundary(window=self.bdw)
        self.clf_fitted = True

    def predict(self):
        """
        Predict flow regimes.

        This function predicts flow regimes for each data point measurement provided.
        Once flow regimes are predicted, the function compiles the output attributes and returns them as a result.

        Returns
        --------
        self.output_data: pd.DataFrame
            DataFrame object all the pressure transient data (both in linear and log scale) processed.
        self.distance_data: pd.DataFrame
            2dim array of N x F shape.
            Where N - number of data measurements in a transient, F - number of flow regimes in the classification.
        """
        if self.clf_fitted == True:
            regime, avg_dist = self.regime_detection(add_wbs_boundary=True, smooth=False)
            self.compile_output(regime, avg_dist)
        else:
            raise AttributeError(f'Classifier is not fitted. Call the fit() function first.')

        return self.output_data, self.distance_data

    def fit_predict(self, data):
        """
        Fit PTAClassifier and predict flow regimes.

        This function enables to fit the PTAClassifier object and predict flow regimes in one line of code.

        Parameters
        --------
        data: np.array
            2dim np.array with 3 parameters (each as a column, linear scale): time, pressure, pressure derivative.

        Returns
        --------
        self.output_data: pd.DataFrame
            DataFrame object all the pressure transient data (both in linear and log scale) processed.
        self.distance_data: pd.DataFrame
            2dim array of N x F shape.
            Where N - number of data measurements in a transient, F - number of flow regimes in the classification.
        """
        self.fit(data)
        regime, avg_dist = self.regime_detection(add_wbs_boundary=True, smooth=False)
        self.compile_output(regime, avg_dist)

        return self.output_data, self.distance_data

    def predict_optimize(self, adjust_filter=True,
                         max_window_length=1,
                         min_filter_value=-0.5,
                         max_filter_value=0.5,
                         deep_search=False):
        """
        Optimize parameters and predict flow regimes.

        This function optimizes model parameters and use them to predict flow regimes.

        Parameters
        --------
        adjust_filter: boolean
            If True, calls adjust_filter function, which performs additional search for filter value with a shorter window length.
        max_window_length: float
            Higher boundary for window length. Sets a reasonable optimization range.
        min_filter_value: float
            Lower boundary for filter value parameter. Sets a reasonable optimization range.
        max_filter_value: float
            Higher boundary for filter value parameter. Sets a reasonable optimization range.
        deep_search: boolean
            If True performs a deeper search for optimal filter value and window length.

        Returns
        --------
        self.output_data: pd.DataFrame
            DataFrame object all the pressure transient data (both in linear and log scale) processed.
        self.distance_data: pd.DataFrame
            2dim array of N x F shape.
            Where N - number of data measurements in a transient, F - number of flow regimes in the classification.
        res: scipy.optimize.OptimizeResult
            Result of GaussianProcess optimization
        """
        if self.clf_fitted == True:
            regime, avg_dist, res = self.bayesian_optimization(adjust_filter=adjust_filter,
                                                               max_window_length=max_window_length,
                                                               min_filter_value=min_filter_value,
                                                               max_filter_value=max_filter_value,
                                                               deep_search=deep_search)
            self.compile_output(regime, avg_dist)
        else:
            raise AttributeError(f'Classifier is not fitted. Call the fit() function first.')

        return self.output_data, self.distance_data, res

    def plot(self, industry_chart = False, figsize = (10,5)):
        """
        Plot PTAClassifier model results.

        This function plots detected flow regime. Industry chart flag provides option to add pressure on the graph.

        Parameters
        --------
        industry_chart: boolean
            If True, outputs commonly used industry chart with pressure and derivative curve.

        Returns
        --------
        self.fig: matplotlib.pyplot.figure
            Output figure.
        """
        if self.output_compiled:
            self.fig = plt.figure(figsize=figsize)
            for r in np.unique(self.output_data.Regime.astype('int')):
                plt.scatter(self.time_log_scale[self.output_data.Regime == r],
                            self.pressure_derivative_log_scale[self.output_data.Regime == r],
                            c=self.color[r], label=f'{self.names[r]} Derivative')
                if industry_chart:
                    plt.scatter(self.time_log_scale[self.output_data.Regime == r],
                                self.pressure_log_scale[self.output_data.Regime == r],
                                c=self.color[r], label=f'{self.names[r]} Pressure', marker='x')
            plt.title(f'Window length = {np.round(self.window_length, 2)}, window step = {self.window_step}')
            plt.xlabel('Log Time')
            plt.ylabel('Log Pressure and Log Derivative' if industry_chart else 'Log Derivative')
            plt.grid(); plt.legend()
            return self.fig
        else:
            raise AttributeError('Output data is not compiled. Call predict() or predict_optimize() first.')

    def get_params(self):
        """
        Get PTAClassifier parameters.

        This function returns parameters, which are currently used by PTAClassifier.

        Returns
        --------
        self.params: dict
            Dict object of the current parameters.
        """
        self.params = {
            'Window length': f'{np.round(self.window_length, 2)}',
            'Window step': f'{np.round(self.window_step, 2)}',
            'Filter value': f'{np.round(self.filter_value, 2)}',
            'Fragment window': f'{np.round(self.fragment_window, 2)}'
        }

        return self.params

    def to_csv(self, name):
        """
        Export results to csv format.

        This function exports self.output_data attribute to the csv format

        Parameters
        --------
        name: str
            Name of the file, without the format extension.
        """
        if self.output_compiled == True:
            self.output_data.to_csv(f'{name}.csv')
        else:
            raise AttributeError('Output data is not compiled. Call predict() or predict_optimize() first.')

    def to_excel(self, name, grouped_by_regime=False):
        """
        Export results to excel format.

        This function exports self.output_data attribute to the excel format

        Parameters
        --------
        name: str
            Name of the file, without format extension.
        """
        if self.output_compiled == True:
            writer = pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl')
            if grouped_by_regime:
                for i, r in enumerate(self.names):
                    detected_regime = self.output_data[self.output_data['Regime'] == i]
                    detected_regime.to_excel(writer, sheet_name=f'{r}')
            else:
                self.output_data.to_excel(writer)
            writer.close()
        else:
            raise AttributeError('Output data is not compiled. Call predict() or predict_optimize() first.')

    def to_pickle(self, name):
        """
        Export results to pickle format.

        This function exports self.output_data attribute to the pickle format

        Parameters
        --------
        name: str
            Name of the file, without the format extension.
        """
        if self.output_compiled == True:
            self.output_data.to_pickle(f'{name}.pkl')
        else:
            raise AttributeError('Output data is not compiled. Call predict() or predict_optimize() first.')