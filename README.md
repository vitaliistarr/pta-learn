# pta-learn

A Python library for automated Pressure Transient Analysis (PTA) workflows. The library provides tools to identify shut-in and flowing transients, detect PTA flow regime features and recognize stable patterns in time-lapse pressure transient responses. 
Feature extraction and pattern recognition modules are based on the methodology described in the peer-reviewed paper: [Feature extraction and pattern recognition in time-lapse pressure transient responses](https://doi.org/10.1016/j.geoen.2024.213160).
Shut-in pressure transient identification module is implemented using the methodology detailed in the conference paper [TPMR - A Novel Method for Automated Identification of Well Pressure Transients](https://doi.org/10.3997/2214-4609.202310910). 
Similarly, the flowing transient identification module employs the approach described in the conference paper [LMIR - A New Method for Automated Identification of Multi-Rate Pressure Transients](https://doi.org/10.3997/2214-4609.202410313).

Usage examples provided in:

<a href="https://colab.research.google.com/drive/1ioJiNM5xpNyP1NoVpBrQp1qr1u94Knlz?usp=sharing"> PTA flow regime feature extraction example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1_ASQ8nmRewhCZmNSMPcs3WBmBFFGiSs6?usp=sharing"> Time-lapse PTA pattern recognition example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1z9B7RzGkWfQEpRWUygTIJvKtjnSeexE4?usp=sharing"> Shutin Transient Identification by TPMR method example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1WFA9hKydEoxk1Z60_NFCQQJwhT9QwMEw?usp=sharing"> Flowing Transient Identification by LMIR method <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1SsXoKafnEJWafGUk8FHa_d7oF6ODljHI?usp=sharing"> Integrated Transient Identification Workflow example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1lVzSIklC-51Nzqehp4lOFctKCjhEbX3Z?usp=sharing"> Loglog family ploting Workflow example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a>

## Installation

Install the package using pip:

```bash
pip install pta-learn
```


## Usage

### Feature Extraction

- Automated flow regime feature detection
- Parameter optimization
- Visualization tools
- Multiple export formats (Excel, CSV)


The `PTAClassifier` class provides automated flow regime detection for single pressure transient analysis:

```python
from pta_learn import PTAClassifier
import numpy as np

# Prepare your data as a 2D numpy array with columns: [time, pressure, pressure_derivative]
data = np.array([
    [time_values],
    [pressure_values],
    [pressure_derivative_values]
]).T

# Initialize classifier
clf = PTAClassifier()

# Option 1: Fit and predict separately
clf.fit(data)
output, distance = clf.predict()

# Option 2: Fit and predict in one step
output, distance = clf.fit_predict(data)

# Option 3: Optimize parameters and predict
output, distance, optimization_result = clf.predict_optimize()

# Plot results
fig = clf.plot(industry_chart=True)

# Export results
clf.to_excel("results")  # or
clf.to_csv("results")
```

### Pattern Recognition
- Multi-transient analysis
- Stable pattern recognition
- Confidence estimation
- Result visualization

The `PatternRecognition` class enables multi-transient analysis to identify stable patterns:

```python
from pta_learn import PatternRecognition
import pandas as pd

# Prepare list of dataframes, each containing [time, pressure, pressure_derivative]
transient1 = pd.DataFrame({
    'time': time_values1,
    'pressure': pressure_values1,
    'pressure_derivative': pressure_derivative_values1
})
transient2 = pd.DataFrame({...})
data = [transient1, transient2]

# Initialize pattern recognition
pr = PatternRecognition()

# Fit data
pr.fit(data)

# Detect features
pr.detect_features()

# Get stable pattern
fig = pr.get_stable_pattern()

# Export results
pr.to_excel("pattern_results")
```

### TPMR method
- Shutin Transient Identification
- Result visualization

The `TPMR` function enables shut-in transient identication in time series sensors.
eg downhole pressure gauges. Please refer to `TPMR` and `plot_target` functions input/output description in the pta_learn module.

```python
from pta_learn import TPMR
from pta_learn.ti_misc import plot_target
import pandas as pd

# Prepare pressure and rate dataframes, pressure data is the one used by the method, and rate data is to verify the result.
pressure = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Pressure': pressure_values,
    'Time': time_values
})
rate = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Rate': rate_values,
    'Time': time_values
})

# Set parameters in the function
# p(float) stands for prominence, and it can be understood as the relative pressure drop in a shut-in transient
p = 10

# interval(float) is the minimum time duration in a shut-in transient
interval_shutin = 20

# run the function
shutin_bp_all, shutin_bp_interval, shutin_transient_all, shutin_transient_interval = TPMR(df_bhp, p, interval_shutin)

# plot the result
plot_target(df_bhp, df_rate ,shutin_transient_interval, TI_empty)
```

### LMIR method
- Flowing Transient Identification
- Result visualization

The `LMIR` function enables flowing transient identification in time series sensors, eg Step Rate Tests in downhole pressure gauges. Please refer to `LMIR` and `plot_target` functions input/output description in the pta_learn module. 

```python
from pta_learn import LMIR
from pta_learn.ti_misc import plot_target
import pandas as pd

# Prepare pressure and rate dataframes, pressure data is the one used by the method, and rate data is to verify the result.
pressure = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Pressure': pressure_values,
    'Time': time_values
})
rate = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Rate': rate_values,
    'Time': time_values
})

# Set parameters in the function
# order (int, optional): The number of adjacent points on each side of a data point to compare when identifying local minima.
order = 10
# - start_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the start of the transient.
# Example: If start_filter_hours = 5, any breakpoint detected within the first 5 hours will be removed.

# - end_filter_hours (int, optional): Filters out breakpoints detected within the specified number of hours from the end of the transient.
# Example: If end_filter_hours = 5, any breakpoint detected within the last 5 hours will be removed.

# run the function
flowing_period, multirate_bp_period,filtered_minima_df,para = LMIR(df_bhp,order = order)

# plot the result
plot_target(df_bhp, df_rate,TI_empty,flowing_period)
```

### Integrated transient identification workflow
- Identify both shut-in transients and flowing transient
- Result visualization

The Integrated transient identification workflow `ti_worflow` function enables shut-in and flowing transients identication in time series sensors, eg downhole pressure gauges. Please refer to `ti_workflow` and `plot_target` functions input/output description in the pta_learn module.

```python
from pta_learn import ti_workflow
from pta_learn.ti_misc import plot_target
import pandas as pd

# Prepare pressure and rate dataframes, pressure data is the one used by the method, and rate data is to verify the result.
pressure = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Pressure': pressure_values,
    'Time': time_values
})
rate = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Rate': rate_values,
    'Time': time_values
})

# Set parameters in the function

# parameters for TPMR method (p)
p = 10
# parameters for LMIR method (order, start_filter_hours , end_filter_hours)
order = 10
# paremeters for workflow (interval_shutin, interval_flowing)
interval_shutin = 10
interval_flowing = 10


# run the function
shutin,flowing,TI,TI_ft,all_breakpoints,w_rate,para = ti_workflow(df_bhp, df_rate, p, interval_shutin, interval_flowing, order = order)

# plot the result
plot_target(df_bhp, df_rate,shutin,flowing)
```

### Loglog family ploting Workflow
- Identify both shut-in transients and flowing transient in PTA
- Rate rebuilt based on breakpoints detected by transient idenfication.
- Superposition time calculation in PTA
- Bourdet derivative calculation in PTA
- Normalization calculation in PTA
- Result visualization

The `Loglog family ploting Workflow` function enables plotting loglog family of Bourdet derivatives from identified transients. Please refer to `cal_loglog_shut`, `cal_loglog_inj` and `plot_TI_family` functions input/output description in the pta_learn module.

```python
from pta_learn import ti_workflow
from pta_learn.bourdet_derivative import cal_loglog_shut, cal_loglog_inj
from pta_learn.normalization import normal_calc
from pta_learn.ti_misc import plot_TI_family
import pandas as pd

# Prepare pressure and rate dataframes, pressure data is the one used by the method, and rate data is to verify the result.
pressure = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Pressure': pressure_values,
    'Time': time_values
})
rate = pd.DataFrame({
    'Timestamp': timestamp_values,
    'Rate': rate_values,
    'Time': time_values
})

# Transient identification workflow

# parameters for TPMR method (p)
p = 10
# parameters for LMIR method (order, start_filter_hours , end_filter_hours)
order = 10
# paremeters for workflow (interval_shutin, interval_flowing)
interval_shutin = 10
interval_flowing = 10

# run the function
shutin,flowing,TI,TI_ft,all_breakpoints,w_rate,para = ti_workflow(df_bhp, df_rate, p, interval_shutin, interval_flowing, order = order)


# Call the cal_loglog_shutin function
# L is the smoothing factor by default.
L = 0.1
log, rate_ave, all_bps = cal_loglog_shut(df_bhp, df_rate, Sel_shutin, all_breakpoints,w_rate,index,L)
log, rate_ave, all_bps = cal_loglog_inj(df_bhp, df_rate, Sel_flowing, all_breakpoints,w_rate,index,L)

# normalize the derivative
# ref is the index of transient used as the reference for normalization
ref = 0
loglog_normalized = []
for i in range(len(loglogs)):
    loglog_nor = normal_calc(loglogs[i],rate_lists[i],ref,rate_lists)
    loglog_normalized.append(loglog_nor)

# plot the loglog family
fig = plot_TI_family(*loglog_normalized)
```

## Citation

If you use this library in your research, please cite:

```
@article{starikov2024feature,
  title={Feature extraction and pattern recognition in time-lapse pressure transient responses},
  author={Starikov, V. and Shchipanov, A. and Demyanov, V. and Muradov, K.},
  journal={Geoenergy Science and Engineering},
  volume={242},
  pages={213160},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.geoen.2024.213160},
  url={https://www.sciencedirect.com/science/article/pii/S294989102400530X}
}

@conference{starikov2023unsupervised,
  title={Unsupervised Classification of Flow Regime Features in Pressure Transient Responses},
  author={Starikov, V. and Demyanov, V. and Muradov, K. and Shchipanov, A.},
  booktitle={Fifth EAGE Conference on Petroleum Geostatistics},
  year={2023},
  month={Nov},
  pages={1-5},
  publisher={European Association of Geoscientists & Engineers},
  doi={10.3997/2214-4609.202335019}
}

@conference{Boyu2023tpmr,
  title={TPMR - A Novel Method for Automated Identification of Well Pressure Transients},
  author={Cui,B. and Zhang,N. and Shchipanov,A. and Rong,C. and Demyanov,V.},
  booktitle={84th EAGE Annual Conference & Exhibition},
  year={2023},
  month={Jun},
  pages={1-5},
  publisher={European Association of Geoscientists & Engineers},
  doi={https://doi.org/10.3997/2214-4609.202310910}
}

@conference{Boyu2024lmir,
  title={LMIR - A New Method for Automated Identification of Multi-Rate Pressure Transients},
  author={Cui,B. and Shchipanov,A. and Zhang,N. and Demyanov,V. and Rong,C.},
  booktitle={85th EAGE Annual Conference & Exhibition},
  year={2024},
  month={Jun},
  pages={1-5},
  publisher={European Association of Geoscientists & Engineers},
  doi={https://doi.org/10.3997/2214-4609.202410313}
}



```
## Acknowledgements

This research code was developed within the AutoWell research and development project funded by the Research Council of Norway and the industry partners including ConocoPhillips Skandinavia, Sumitomo Corporation Europe Norway Branch, Harbour Energy and Aker BP (grant no. 326580, PETROMAKS2 programme)​

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

