# ENSO-MC 1.0.0
This project is built and trained on Ubuntu 18.04, with python3.6 and CUDA 10.0
## 0. Enviroment
| Package  | Version |
| ------------- | ------------- |
| tensorflow-gpu  | 2.4.0  |
| numpy  | 1.19.5  |
| Cartopy  | 0.18.0  |
| loguru  | 0.5.3  |
| pandas  | 0.25.3  |
| scikit-learn  | 0.19.0  |
| xarray  | 0.16.2  |

## 1. Prepare the dataset
The complete dataset can be downloaded from [CISL of NCAR](https://rda.ucar.edu/datasets/ds277.3/), [IRI/LDEO Climate Data Library](https://iridl.ldeo.columbia.edu/SOURCES/.CARTON-GIESE/.SODA/.v2p2p4/), [NOAA/OAR/ESRL PSL](https://psl.noaa.gov/data/gridded/) and [ECMWF](https://apps.ecmwf.int/datasets/data/interim-full-moda/levtype=sfc/).
And the scripts in `./data` are provided for downloading from the above websites.
```
python ./data/*_data/download_*.py
```
Convert the data from .nc to .npz format for the input of model.
```
python ./predictorTest/multivar/GODAS(SODA)/data2npz.py
```

## 2. Train the model
```
python train_distributed.py
```

## 3. Test
```
python ./predictorTest/multivar/testDMS.py
```
or
```
python ./rollingPredict/testRolling.py
```

## 4. Precursor analysis
```
python ./explain/explainer.py
```

## 5. Sensitive areas identification
Get the composite saliency map:
```
python ./sensitiveExp/sensitive_area.py
```
Compare the sensitivity of six regions:
```
python ./sensitiveExp/sensitivity_compare.py
```


