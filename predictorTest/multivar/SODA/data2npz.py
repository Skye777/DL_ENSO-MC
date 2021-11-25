"""
@author: Skye Cui
@file: data2npz.py
@time: 2021/5/11 9:28
@description: 
"""
import xarray
from PIL import Image
import netCDF4 as nc
import numpy as np
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

sst_l = nc.Dataset(f"{hp.reanalysis_dataset_dir}/meta-data/HadISST_sst.nc").variables['sst'][12:1668, 50:130, 300:360]
sst_r = nc.Dataset(f"{hp.reanalysis_dataset_dir}/meta-data/HadISST_sst.nc").variables['sst'][12:1668, 50:130, 0:100]
sst = np.flip(np.concatenate((sst_l, sst_r), axis=2), 1).filled()
sst[sst == -1.0e+30] = 0

ssh = xarray.open_dataset(f'{hp.soda_dataset_dir}/interp-data/ssh.nc', cache=True, decode_times=False).fillna(0).variables['ssh']
taux = xarray.open_dataset(f'{hp.soda_dataset_dir}/interp-data/taux.nc', cache=True, decode_times=False).fillna(0).variables['taux']
tauy = xarray.open_dataset(f'{hp.soda_dataset_dir}/interp-data/tauy.nc', cache=True, decode_times=False).fillna(0).variables['tauy']
t300 = xarray.open_dataset(f'{hp.soda_dataset_dir}/interp-data/t300.nc', cache=True, decode_times=True).variables['temp']
# print(sst.min(), sst.max())
# print(np.array(ssh).min(), np.array(ssh).max())
# print(np.array(taux).min(), np.array(taux).max())
# print(np.array(tauy).min(), np.array(tauy).max())
# print(sst.shape)
# print(ssh.shape)
# print(taux.shape)
# print(tauy.shape)
# print(np.array(t300).min(), np.array(t300).max())
# print(t300.shape)

data = {'sst': np.array(sst)}
np.savez(f"{hp.multivar}/SODA/npz_data/sst.npz", **data)

data = {'ssh': np.array(ssh)}
np.savez(f"{hp.multivar}/SODA/npz_data/ssh.npz", **data)

data = {'taux': np.array(taux)}
np.savez(f"{hp.multivar}/SODA/npz_data/taux.npz", **data)

data = {'tauy': np.array(tauy)}
np.savez(f"{hp.multivar}/SODA/npz_data/tauy.npz", **data)

data = {'t300': np.array(t300)}
np.savez(f"{hp.multivar}/SODA/npz_data/t300.npz", **data)

# var = np.load('D:/Python/transformer/predictorTest/sst-resolve.npz')['sst']
# print(var.min(), var.max())
