"""
@author: Skye Cui
@file: data_read.py
@time: 2021/6/7 22:05
@description: 
"""
import xarray
import matplotlib.pyplot as plt
import numpy as np
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

# dataset_s = self.config.dataset_s
# dataset_e = self.config.dataset_e
llat, rlat, llon, rlon = [-40, 40, 120, 280]
blev = 300

dataset = xarray.open_dataset(f'{hp.godas_dataset_dir}/meta-data/pottmp/pottmp.1980.nc', cache=True, decode_times=True)
print(dataset)
# data = dataset['tauy']
lc = dataset.coords['lon']
la = dataset.coords['lat']
lev = dataset.coords['level']

data = dataset.loc[
    dict(level=lev[lev <= blev], lon=lc[(lc >= llon) & (lc <= rlon)], lat=la[(la >= llat) & (la <= rlat)])]['pottmp'].mean(dim='level')-273.15
print(data)
data.isel(time=0).plot()
plt.tight_layout()
plt.show()

# lat_len, lon_len = 80, 160
# lon = np.linspace(120.25, 279.75, lon_len)
# lat = np.linspace(-39.75, 39.75, lat_len)
# interp_data = data.interp(lat=lat, lon=lon)
# print(interp_data)
# # print(interp_data.shape)
# interp_data['tauy'].isel(time=0).plot()
# plt.tight_layout()
# plt.show()
# # interp_data.to_netcdf('interp_data.nc')
