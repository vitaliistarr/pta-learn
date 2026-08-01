# pta-learn

A Python library for automated Pressure Transient Analysis (PTA) workflows. The library provides tools to identify shut-in and flowing transients, detect PTA flow regime features and recognize stable patterns in time-lapse pressure transient responses. 
Feature extraction and pattern recognition modules are based on the methodology described in the peer-reviewed paper: [Feature extraction and pattern recognition in time-lapse pressure transient responses](https://doi.org/10.1016/j.geoen.2024.213160).
Shut-in pressure transient identification module is implemented using the methodology detailed in the conference paper: [TPMR - A Novel Method for Automated Identification of Well Pressure Transients](https://doi.org/10.3997/2214-4609.202310910). 
Similarly, the flowing transient identification module employs the approach described in the conference paper: [LMIR - A New Method for Automated Identification of Multi-Rate Pressure Transients](https://doi.org/10.3997/2214-4609.202410313).

## Installation

Install the package using pip:

```bash
pip install pta-learn
```


## Usage

### Pressure Transient Identification
<a href="https://colab.research.google.com/drive/1SsXoKafnEJWafGUk8FHa_d7oF6ODljHI?usp=sharing"> Integrated Transient Identification Workflow example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1z9B7RzGkWfQEpRWUygTIJvKtjnSeexE4?usp=sharing"> Shutin Transient Identification by TPMR method example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1WFA9hKydEoxk1Z60_NFCQQJwhT9QwMEw?usp=sharing"> Flowing Transient Identification by LMIR method <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>

### Pressure Transient Quality Screening
<a href="https://colab.research.google.com/drive/1VQ4WcATTnOtiuFuHI0SsGkaVx852eRlt?usp=sharing"> Pressure Transient Quality Scores <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1VWowB60ohT1Qp5qy01mmVvTTHfH6ViZN?usp=sharing"> Rate Absolute Relative Error (RARE) calculation example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>


### Bourdet Derivative and Loglog Plot Calculation
<a href="https://colab.research.google.com/drive/1lVzSIklC-51Nzqehp4lOFctKCjhEbX3Z?usp=sharing"> Loglog family ploting Workflow example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/14DUNrKVd90Jzc0dfJHHC3Jg37MIQcuox?usp=sharing"> Derivative Uncertainty Envelope (DUE) calculation example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>

### Pattern Recognition in Time-lapse Pressure Transient Responses
<a href="https://colab.research.google.com/drive/1ioJiNM5xpNyP1NoVpBrQp1qr1u94Knlz?usp=sharing"> PTA flow regime feature extraction example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>
<a href="https://colab.research.google.com/drive/1_ASQ8nmRewhCZmNSMPcs3WBmBFFGiSs6?usp=sharing"> Time-lapse PTA pattern recognition example <img src="https://colab.research.google.com/assets/colab-badge.svg" height=16px></a> <br>


## Citation

If you use this library in your research, please cite:

### Pressure Transient Identification
```
@article{shchipanov2025new,
  title={New transient identification methods for automated pre-processing of pressure measurements with permanent well gauges},
  author={Cui, B. and Shchipanov, A. and Demyanov, V. and Zhang, N. and Rong, C.},
  journal={Geoenergy Science and Engineering},
  volume={257},
  pages={214203},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.geoen.2025.214203},
  url={https://www.sciencedirect.com/science/article/pii/S2949891025005615}
}
```

### Pattern Recognition in Time-Lapse PTA
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

### Derivative Uncertainty Envelope (DUE)
```
@inproceedings{mugisha2026breakthrough,
  title={A Breakthrough in Early Detection and Monitoring of Induced Fracture with Automated Interpretation of Step-Rate Tests},
  author={Mugisha, J. and Shchipanov, A. Midtb{\o} {\O}verland, A. Starikov, V. and Muradov, K.},
  booktitle={Paper presented at the SPE Europe Energy Conference and Exhibition},
  address={Istanbul, Turkey},
  year={2026},
  month={June},
  number={SPE-233202-MS},
  organization={Society of Petroleum Engineers},
  doi={10.2118/233202-MS},
  url={https://doi.org/10.2118/233202-MS}
}
```

## Acknowledgements

This research code was developed within the following projects:

- AutoWell research and development project funded by the Research Council of Norway and the industry partners including
ConocoPhillips Skandinavia AS, Sumitomo Corporation Europe Norway Branch, Harbour Energy Norge AS and Aker BP ASA
(grant no. 326580, PETROMAKS2 programme).
- AutoWell Phase 2, a joint industry research and development project funded by ConocoPhillips Skandinavia AS,
Aker BP ASA, Harbour Energy Norge AS and TotalEnergies EP Norge AS.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

