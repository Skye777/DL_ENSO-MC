"""
@author: Skye Cui
@file: data2npz.py
@time: 2021/6/8 16:59
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

# sst = xarray.open_dataset(f'{hp.godas_dataset_dir}/interp-data/sst.nc')['sst'].loc['1981-12-01':'2019-08-01', :, :].fillna(0)
# sshg = xarray.open_dataset(f'{hp.godas_dataset_dir}/interp-data/sshg.nc')['sshg'].loc['1981-12-01':'2019-08-01', :, :].fillna(0)
# uwind = xarray.open_dataset(f'{hp.godas_dataset_dir}/interp-data/uwind.nc')['u10'].loc['1981-12-01':'2019-08-01', :, :]
# vwind = xarray.open_dataset(f'{hp.godas_dataset_dir}/interp-data/vwind.nc')['v10'].loc['1981-12-01':'2019-08-01', :, :]
t300 = xarray.open_dataset(f'{hp.godas_dataset_dir}/interp-data/t300.nc')['pottmp'].loc['1981-12-01':'2019-08-01', :, :].fillna(0)

# print(np.array(sst).min(), np.array(sst).max())
# print(np.array(sshg).min(), np.array(sshg).max())
# print(np.array(uwind).min(), np.array(uwind).max())
# print(np.array(vwind).min(), np.array(vwind).max())
# print(sst.shape)
# print(sshg.shape)
# print(uwind.shape)
# print(vwind.shape)
print(np.array(t300).min(), np.array(t300).max())
print(t300.shape)

# data = {'sst': np.array(sst)}
# np.savez(f"{hp.multivar}/GODAS/npz_data/sst.npz", **data)
#
# data = {'sshg': np.array(sshg)}
# np.savez(f"{hp.multivar}/GODAS/npz_data/sshg.npz", **data)
#
# data = {'u10': np.array(uwind)}
# np.savez(f"{hp.multivar}/GODAS/npz_data/uwind.npz", **data)
#
# data = {'v10': np.array(vwind)}
# np.savez(f"{hp.multivar}/GODAS/npz_data/vwind.npz", **data)

data = {'t300': np.array(t300)}
np.savez(f"{hp.multivar}/GODAS/npz_data/t300.npz", **data)
