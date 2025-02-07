# pta-learn

A Python library for automated feature extraction and pattern recognition for Pressure Transient Analysis (PTA) workflows. The library provides tools to detect flow regime features and recognize stable patterns in time-lapse pressure transient responses. The library is based on the methodology described in the peer-reviewed paper: [Feature extraction and pattern recognition in time-lapse pressure transient responses](https://doi.org/10.1016/j.geoen.2024.213160)

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
clf.to_excel("results.xlsx")  # or
clf.to_csv("results.csv")
```

### Pattern Recognition
- Multi-transient analysis
- Stable pattern detection
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
pr.to_excel("pattern_results.xlsx")
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
```
## Acknowledgements

This research code was developed within the AutoWell research and development project funded by the Research Council of Norway and the industry partners including ConocoPhillips Skandinavia, Sumitomo Corporation Europe Norway Branch, Harbour Energy and Aker BP (grant no. 326580, PETROMAKS2 programme)​

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

