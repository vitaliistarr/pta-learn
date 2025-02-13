import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px

from scipy.interpolate import interp1d
from itertools import combinations
import time as time_lib
from sklearn.linear_model import LinearRegression

from .feature_extraction import Logtime_window, PTAClassifier


class PatternRecognition():
    """
        Create PatternRecognition class for multi-transient analysis.

        This class allows to perform multi-transient analysis and provides several methods for pattern recognition:
        1. Deterministic PTAClassifier model
        The current model is build on PTA_classifie feature extraction model and effectively uses predict_optimize()
        function to recognize stable patterns.

        Attributes
        --------
        regime_names: list
            List of regime names. Not defined by user.
        regime_color: list
            List of regime color for plotting. Not defined by user.
        pattern_color: list
            List of colors for detected stable patterns.
        stable_pattern_intervals: pd.DataFrame
            Table of recognized stable patterns.

        Methods
        --------
        fit():
            Fit the PatternRecognition object.
        detect_features():
            Use PTAClassifier.predict_optimize() function to process each pressure transient.
        combine_distances():
            Process the output distance array (from .predict_optimize() function).
        get_interval():
            Detect interval of a stable pattern
        compute_confidence():
            Convert the output distance array to a confidence in the recognized stable pattern interval.
        get_stable_pattern():
            Get and plot stable pattern.
        to_excel():
            Export results to excel.
    """
    def __init__(
            self
    ):
        self.regime_names = ['Radial', 'Linear-up', 'Linear-down', 'Boundary', 'WBS']
        self.regime_color = ['green', 'maroon', 'orange', 'navy', 'blue']
        self.pattern_color = {'Radial': 'olive', 'Linear-up': 'black', 'Linear-down': 'indigo'}
        self.stable_pattern_intervals = pd.DataFrame(index=['start', 'end', 'confidence'],
                                                     columns=['Radial', 'Linear-up', 'Linear-down'])

    def fit(self, dataframes):
        """
        Fit the classifier.

        This function fits the PatternRecognition object with data to analyze.

        Parameters
        --------
        dataframes: list
            A list of dataFrame objects with following columns: time, pressure, pressure derivative.
        """
        if isinstance(dataframes, list):
            self.data = dataframes
        else:
            raise TypeError(f'Expected format: list(pd.DataFrame, pd.DataFrame ...). '
                            f'Given format: {type(dataframes)}')

    def detect_features(self, adjust_filter=True, max_window_length=1, min_filter_value=-0.5, max_filter_value=0.5,
                        industry_chart=False, figsize = (10,5), deep_search=False):
        """
        Process each transient with PTAClassifier and predict flow regimes.

        This function effectively uses inbuilt PTAClassifier optimizated method to predict flow regimes in each
        pressure transient. The function creates 4 output PatternRecognition class attributes:
            1. output (results for each pressure transient)
            2. distance (underlying detected distances)
            3. figures (plots of each processed pressure transient)
            4. params (optimized hyperparameters for each pressure transient)

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
            If True performs a deeper search for optimal filter value and window length parameters of PTAClassifier.
        """
        self.output = []
        self.distance = []
        self.params = []
        self.figures = []

        clf = PTAClassifier()
        for d in self.data:
            clf.fit(d.values)
            result, dist, res = clf.predict_optimize(adjust_filter=adjust_filter,
                                                     max_window_length=max_window_length,
                                                     min_filter_value=min_filter_value,
                                                     max_filter_value=max_filter_value,
                                                     deep_search=deep_search)
            fig = clf.plot(industry_chart=industry_chart, figsize=figsize)
            self.figures.append(fig)
            self.output.append(result)
            self.distance.append(dist)
            self.params.append(clf.get_params())

    def combine_distances(self):
        """
        Process distance output from PTAClassifier.

        This function combines reference slopes distances calculated by PTAClassifier for each flow regime.
        As a result this function creates combined_dist attribute in PatternRecognition object.
        """
        def find_two_lowest_values(arr):
            min1 = np.inf
            min2 = np.inf
            index1 = None
            index2 = None
            for i, num in enumerate(arr):
                if num < min1:
                    min2 = min1
                    index2 = index1
                    min1 = num
                    index1 = i
                elif num < min2:
                    min2 = num
                    index2 = i

            return min1, min2, index1, index2

        self.combined_dist = []

        for dist in self.distance:
            combined_dist = np.full((dist.shape[0], 3), np.inf)
            for i, d in enumerate(dist.values):
                min1, min2, index1, index2 = find_two_lowest_values(d)

                if index1 in [0]:
                    combined_dist[i, 0] = min1
                elif index1 in [1, 2]:
                    combined_dist[i, 1] = min1
                elif index1 in [3, 4]:
                    combined_dist[i, 2] = min1

                if index2 in [0]:
                    combined_dist[i, 0] = min2
                elif index2 in [1, 2]:
                    combined_dist[i, 1] = min2
                elif index2 in [3, 4]:
                    combined_dist[i, 2] = min2

            self.combined_dist.append(combined_dist)

    def get_interval(self, minimum_interval_length = 0.3, max_number_of_transients_to_drop = 2):
        """
        Detect and get stable pattern intervals for each flow regime.

        This function analyzes results of PTAClassifier by resampling and cross-checking detected regimes.
        Stable pattern is detected if all the input transients contain the same regime at the resampled time.
        Stable patterns shorter than minimum_interval_length are filtered out.
        The function takes the longest stable pattern for each type of flow regime (Radial, Linear-up, Linear-down).
        Confidence in a detected stable pattern interval is calculated by Bayesian inference of all interval likelihoods
        from each pressure transient.
        As a result, the function compiles stable_pattern_intervals and confidence attributes of PatternRecognition
        object.

        Parameters
        --------
        minimum_interval_length: float
            Minimum logarithmic time duration of stable pattern to detect.

        See also:
        --------
        combine_distances: process output distance object from PTAClassifier.
        compute_confidence: convert underlying feature similarities into a certain confidence/likelihood.
        """
        def extract_true_intervals(arr):
            intervals = []
            start = None
            for i, value in enumerate(arr):
                if value and start is None:
                    start = i
                elif not value and start is not None:
                    intervals.append((start, i - 1))
                    start = None
            # Check if the last interval continues until the end of the array
            if start is not None:
                intervals.append((start, len(arr) - 1))

            return intervals

        def detect_patterns(data, regime):
            detected_patterns = np.array([False] * len(data), dtype=bool)
            for i,row in enumerate(data.values):
                if all(x == regime for x in row):
                    detected_patterns[i] = True

            return detected_patterns

        def select_transients_to_process(arr, num_extracted):
            variations_extracted = []
            variations_left = []
            for indices in combinations(range(len(arr)), num_extracted):
                extracted_values = [arr[i] for i in indices]
                left_values = [arr[i] for i in range(len(arr)) if i not in indices]
                variations_extracted.append(extracted_values)
                variations_left.append(left_values)

            return variations_extracted, variations_left

        self.combine_distances()
        self.confidence = dict()
        self.pattern_recognized = False
        num_of_transients_to_drop = 0
        t = time_lib.time()
        # Find max pattern duration for each regime. Stores data in NxR matrix: N-iteration, R-regime
        while not self.pattern_recognized:

            print(f'Drop number = {num_of_transients_to_drop}')
            dropped_transients, variation_of_selected_transients = select_transients_to_process(self.output, num_of_transients_to_drop)
            max_durations = np.zeros(shape=(len(variation_of_selected_transients), 3))
            start_end = np.zeros(shape=(len(variation_of_selected_transients), 3, 2))

            for iteration, selected_transients in enumerate(variation_of_selected_transients):

                t_min = max([result['Time'].values[0] for result in selected_transients])
                t_max = min([result['Time'].values[-1] for result in selected_transients])
                time = np.logspace(np.log10(t_min), np.log10(t_max), num=500)
                resampled_results = pd.DataFrame()
                for t_num, result in enumerate(selected_transients):
                    synth_regime_func = interp1d(result['Time'].values, result['Regime'].values, kind='linear',
                                                 fill_value='extrapolate')
                    resampled_results[f'Transient_{t_num + 1}'] = synth_regime_func(time)

                for i,r in enumerate(self.stable_pattern_intervals.columns):
                    detected_patterns = detect_patterns(resampled_results, i)
                    if np.any(detected_patterns):
                        true_intervals = extract_true_intervals(detected_patterns)
                        durations = []
                        for true_interval in true_intervals:
                            duration = np.log10(time[true_interval[1]]) - np.log10(time[true_interval[0]]) # Get logtime duration
                            durations.append(duration)

                        start = time[true_intervals[np.argmax(durations)][0]]
                        end = time[true_intervals[np.argmax(durations)][1]]
                        if abs(np.log10(end) - np.log10(start)) >= minimum_interval_length:
                            print(f'Stable pattern {(i, r)} detected, duration: {max(durations)}')
                            max_durations[iteration,i] = max(durations)
                            start_end[iteration,i] = np.array([start,end])
                            self.pattern_recognized = True

            if self.pattern_recognized:
                print(f'Pattern recognized, max_durations array {max_durations}')
                for i, r in enumerate(self.stable_pattern_intervals.columns):
                    if max(max_durations[:,i]) > 0: # Proceed with detected regimes
                        iter_num = np.argmax(max_durations[:,i]) # Take interation with longest pattern duration
                        start_detected = start_end[iter_num, i, 0]
                        end_detected = start_end[iter_num, i, 1]
                        conf = self.compute_confidence(regime_number=i, start_time=start_detected, end_time=end_detected)
                        self.stable_pattern_intervals[r] = np.array([start_detected, end_detected, np.prod(conf)])
                        self.confidence[r] = conf

            elif not self.pattern_recognized:
                if num_of_transients_to_drop < max_number_of_transients_to_drop:
                    num_of_transients_to_drop += 1
                else:
                    print(f'Stable pattern was not detected: min_pattern {minimum_interval_length} logtime; max_transients_to_drop {max_number_of_transients_to_drop}.')
                    self.pattern_recognized = True

        print(f'time to recognize interval {time_lib.time() - t}')

    def correlation_analysis(self, corr_coef, filenames):

        self.correlation_thresholds = np.round(np.linspace(corr_coef, 1, 5), 2)
        self.correlation_thresholds[-1] = 0.99
        starts = np.zeros(shape=len(self.correlation_thresholds)).astype('int') #Start window indices
        ends = np.zeros(shape=len(self.correlation_thresholds)).astype('int')   #End window indices
        self.lines = np.zeros(shape=(len(self.correlation_thresholds), 2)).astype('int')
        self.corr_window_found = np.array([False]*len(self.correlation_thresholds))

        # Function returns window indices with the highest average correlation. End window index is included
        def get_correlated_interval(data, threshold):
            current_region_start = None
            current_region_end = None
            highest_avg_corr = 0
            for window_number, correlation_coefficient in enumerate(data):
                if correlation_coefficient > threshold:
                    if current_region_start is None:
                        current_region_start = window_number
                    current_region_end = window_number
                else:
                    if current_region_start is not None:
                        if np.average(data[current_region_start:current_region_end+1]) > highest_avg_corr:
                            highest_corr_region_start = current_region_start
                            highest_corr_region_end = current_region_end
                            highest_avg_corr = np.average(data[highest_corr_region_start:highest_corr_region_end+1])
                        current_region_start = None
                        current_region_end = None
            # Check one last time after the loop
            if current_region_start is not None:
                if np.average(data[current_region_start:current_region_end+1]) > highest_avg_corr:
                    highest_corr_region_start = current_region_start
                    highest_corr_region_end = current_region_end
                    highest_avg_corr = np.average(data[highest_corr_region_start:highest_corr_region_end+1])

            if highest_avg_corr > 0:
                return highest_corr_region_start, highest_corr_region_end, True
            else:
                return 0, 0, False

        # Get max min time values for resampling
        t_min = max([d.iloc[:,0].values[0] for d in self.data])
        t_max = min([d.iloc[:,0].values[-1] for d in self.data])
        self.resampled_corr_data = pd.DataFrame()

        # Resampled data before computing correlation
        self.resampled_corr_data['Time'] = np.log10(np.logspace(np.log10(t_min), np.log10(t_max), num=500))
        for i,data in enumerate(self.data):
            log_time = np.log10(data.iloc[:,0].values)
            log_time = np.nan_to_num(log_time, nan=0)
            log_pressure_deriv = np.log10(data.iloc[:,2].values)
            log_pressure_deriv = np.nan_to_num(log_pressure_deriv, nan=0)
            f = interp1d(log_time, log_pressure_deriv, kind='linear', fill_value="extrapolate")
            self.resampled_corr_data[filenames[i]] = f(self.resampled_corr_data['Time'].values)

        # Run logtime window
        ltw = Logtime_window(time=self.resampled_corr_data['Time'].values,
                             pressure=self.resampled_corr_data.iloc[:, 1].values,
                             window_length=0.5,
                             window_step=0.02)
        number = ltw.number_of_windows()
        corr = np.zeros(number)
        data_point_start_indices = np.zeros(number) # Data point indices of each window
        data_point_end_indices = np.zeros(number)
        for n in range(number):
            i, j, _, _, x, y = ltw.get_window_frame(n)
            corr[n] = abs(self.resampled_corr_data.iloc[i:j, 1:].corr()).min().min()
            data_point_start_indices[n] = i
            data_point_end_indices[n] = j
        data_point_start_indices = data_point_start_indices.astype('int')
        data_point_end_indices = data_point_end_indices.astype('int')

        # Get start and end indices of correlation thresholds
        for i, ct in enumerate(self.correlation_thresholds):
            if i == 0:
                # find start end window in the initial correlation array data
                starts[i], ends[i], self.corr_window_found[i] = get_correlated_interval(corr, threshold=ct)
            else:
                # find higher correlation inside previously identified window (if identified before)
                if self.corr_window_found[i-1]:
                    s, e, self.corr_window_found[i] = get_correlated_interval(corr[starts[i - 1]:ends[i - 1] + 1], threshold=ct)
                    starts[i], ends[i] = starts[i - 1] + s, starts[i - 1] + e

        # Take periods where correlation window is detected
        starts = starts[self.corr_window_found]
        ends = ends[self.corr_window_found]
        self.lines = self.lines[self.corr_window_found]
        self.correlation_thresholds = self.correlation_thresholds[self.corr_window_found]
        for i in range(len(self.lines)):
            self.lines[i] = [data_point_start_indices[starts[i]], data_point_end_indices[ends[i]]]
        self.lines[:,1] -= 1 # Subtract last indices by 1

        # Define the regime
        if self.corr_window_found.any():
            beta = []
            x = self.resampled_corr_data.Time.values[self.lines[0][0]:self.lines[0][1]]
            for i in range(1,len(self.data)+1):
                y = self.resampled_corr_data.iloc[:,i].values[self.lines[0][0]:self.lines[0][1]]
                lr_reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
                beta.append(lr_reg.coef_[0][0])
            if -0.1 <= np.average(beta) <= 0.1:
                self.corr_defined_regime = 'Radial'
            elif 0.1 < np.average(beta) < 0.8:
                self.corr_defined_regime = 'Linear-up'
            elif -0.8 < np.average(beta) < -0.1:
                self.corr_defined_regime = 'Linear-down'
            else:
                self.corr_defined_regime = 'Not defined'

    def compute_confidence(self, regime_number, start_time, end_time, confidence_type = 'feature_based'):
        """
        Calculate confidence in a detected stable pattern.

        This function converts underlying similarity between data and reference slopes into a likelihood of a flow
        regime being in the selected interval.

        Parameters
        --------
        regime_number: float
            Numeric value of regime (0 - Radial, 1 - Linear-up, 2 - Linear-down, 3 - Boundary, 4 - WBS).
        start_time: float
            Logarithmic start time value.
        end_time: float
            Logarithmic end time value.

        Returns
        --------
        confidence: np.array
            1dim array of regime likelihoods in each pressure transient.
        """
        confidence = np.ones(shape=len(self.output))

        if confidence_type == 'feature_based':
            for i, result in enumerate(self.output):
                total_time = np.log10(end_time) - np.log10(start_time)
                regime_time_array = result.query(f'Regime == {regime_number} & Time >= {start_time} & Time <= {end_time}')['Time'].values.ravel()
                if len(regime_time_array) > 1:
                    regime_time = np.log10(regime_time_array[-1]) - np.log10(regime_time_array[0])
                else:
                    regime_time = 0
                confidence[i] = regime_time/total_time

        elif confidence_type == 'distance_based':
            for i,combined_dist in enumerate(self.combined_dist):
                cd = 1 / combined_dist
                row_sums = cd.sum(axis=1)
                probability = cd / row_sums[:, np.newaxis]
                time = self.output[i].loc[(self.output[i]['Regime'] != 3) & (self.output[i]['Regime'] != 4), ['Time']].values.ravel()
                confidence[i] = probability[:,regime_number][(time >= start_time) & (time <= end_time)].mean()

        return confidence

    def get_stable_pattern(self, minimum_interval_length = 0.3, figsize=(10,5)):
        """
        Get deterministic pattern recognition results.

        This function calls get_interval() method, plot and output results of the analysis.

        Parameters
        --------
        minimum_interval_length: float
            Minimum logarithmic time duration of stable pattern to detect.

        Returns
        --------
        pattern_recognition_plot: matplotlib.pyplot.figure
            Pattern recognition plot.
        """
        # If not found, recursively eliminate till 2 trnasients are left
        self.get_interval(minimum_interval_length = minimum_interval_length,
                          max_number_of_transients_to_drop = len(self.output)-2)

        regime = np.array([])
        time = np.array([])
        pressure_derivative = np.array([])

        for result in self.output:
            regime = np.concatenate((regime, result.Regime.values))
            time = np.concatenate((time, result.Log_time.values))
            pressure_derivative = np.concatenate((pressure_derivative, result.Log_pressure_derivative.values))

        self.pattern_recognition_plot = plt.figure(figsize=figsize)
        regime = regime.astype('int')
        for r in np.unique(regime):
            plt.scatter(time[regime == r], pressure_derivative[regime == r], c=self.regime_color[r], label=self.regime_names[r])

        # All patterns in one plot
        if self.pattern_recognized:
            for i,regime in enumerate(self.stable_pattern_intervals.columns):
                if not np.any(self.stable_pattern_intervals[regime].isna().values):
                    start = np.log10(self.stable_pattern_intervals[regime].values[0])
                    end = np.log10(self.stable_pattern_intervals[regime].values[1])
                    conf = self.stable_pattern_intervals[regime].values[2]
                    plt.plot([start, start], [min(pressure_derivative), max(pressure_derivative)],
                             label=f'{self.regime_names[i]} stable pattern',
                             color=self.pattern_color[regime], linewidth=3)
                    plt.plot([end, end], [min(pressure_derivative), max(pressure_derivative)],
                             color=self.pattern_color[regime], linewidth=3)
                    print(f'{regime} stable pattern was found at [{np.round(start,2)}, {np.round(end,2)}] '
                          f'log time range with a confidence of {np.round(conf,2)}.')
        else:
            print('Stable pattern was not recognized.')

        plt.title(f'Pattern recognition plot')
        plt.xlabel('Log Time'), plt.ylabel('Log Derivative')
        plt.grid(); plt.legend()

        return self.pattern_recognition_plot

    def get_correlation_plot(self):

        self.data_for_plotting = 10**self.resampled_corr_data
        def create_colormap(N):
            colors = px.colors.sequential.Viridis
            # Interpolate the color scale to get N colors
            if N == 1:
                interpolated_colors = [colors[-1]]
            else:
                interpolated_colors = [colors[i * (len(colors) - 1) // (N - 1)] for i in range(N)]
            return interpolated_colors

        # Plot the data curve
        fig = go.Figure()
        min_y = self.data_for_plotting.iloc[:,1:].min().min()
        max_y = self.data_for_plotting.iloc[:,1:].max().max()
        for column in self.data_for_plotting.columns[1:]:
            fig.add_trace(go.Scatter(x=self.data_for_plotting.Time, y=self.data_for_plotting[column],
                                     mode='markers', name=column, legendgroup='Data'))
        # custom color map
        custom_colors = create_colormap(len(self.correlation_thresholds))

        # Fill the regions between the vertical lines with colors
        fig.add_shape(
            go.layout.Shape(type="rect", x0=self.data_for_plotting.Time[self.lines[-1][0]],
                            x1=self.data_for_plotting.Time[self.lines[-1][1]], y0=min_y, y1=max_y,
                            fillcolor=custom_colors[-1], opacity=0.5,
                            name=f'Corr. coef = {self.correlation_thresholds[-1]}'))
        left_x = self.lines[:, 0]
        right_x = self.lines[:, 1]
        for i in range(len(left_x) - 1):
            fig.add_shape(
                go.layout.Shape(type="rect", x0=self.data_for_plotting.Time[left_x[i]],
                                x1=self.data_for_plotting.Time[left_x[i + 1]], y0=min_y, y1=max_y,
                                fillcolor=custom_colors[i], opacity=0.5,
                                name=f'Corr. coef = {self.correlation_thresholds[i]}'))
            fig.add_shape(
                go.layout.Shape(type="rect", x0=self.data_for_plotting.Time[right_x[i + 1]],
                                x1=self.data_for_plotting.Time[right_x[i]], y0=min_y, y1=max_y,
                                fillcolor=custom_colors[i], opacity=0.5))
        for i, correlation_threshold in enumerate(self.correlation_thresholds):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color=custom_colors[i]),
                                     name=f'Corr. coef = {correlation_threshold}', legendgroup=f'Stable pattern'))
        fig.update_xaxes(title_text='Time, hr', type='log')
        fig.update_yaxes(title_text='Pressure derivative, psi', type='log')

        return fig

    def to_excel(self, name, grouped_by_regime=False):
        """
        Export results to Excel format.

        This function exports self.output_data attribute to the excel format

        Parameters
        --------
        name: str
            Name of the file, without the format extension.
        """
        with pd.ExcelWriter(f'{name}.xlsx', engine='openpyxl') as writer:
            if grouped_by_regime:
                for i, r in enumerate(self.regime_names):
                    regime = pd.DataFrame()
                    for result in self.output:
                        detected_regime = result[result['Regime'] == i]
                        regime = pd.concat([regime, detected_regime], axis=0)
                    regime.to_excel(writer, sheet_name=f'{r}')
            else:
                for i,result in enumerate(self.output):
                    result.to_excel(writer, sheet_name=f'Transient_{i+1}')
                if self.pattern_recognized:
                    self.stable_pattern_intervals.to_excel(writer, sheet_name='Stable pattern interval')