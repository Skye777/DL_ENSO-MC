"""
@author: Skye Cui
@file: data_read.py
@time: 2021/6/7 22:05
@description: 
"""
import xarray
import matplotlib.pyplot as plt
import numpy as np

# dataset_s = self.config.dataset_s
# dataset_e = self.config.dataset_e
llat, rlat, llon, rlon = [-40, 40, 120, 280]

dataset = xarray.open_dataset('meta-data/vwind.mon.mean1980-2019.nc', cache=True, decode_times=True)
print(dataset)
# data = dataset['tauy']
lc = dataset.coords['longitude']
la = dataset.coords['latitude']

data = dataset.loc[dict(longitude=lc[(lc >= llon) & (lc <= rlon)], latitude=la[(la >= llat) & (la <= rlat)])]
data['v10'].isel(time=0).plot()
plt.tight_layout()
plt.show()

lat_len, lon_len = 80, 160
lon = np.linspace(120.5, 279.5, lon_len)
lat = np.linspace(-39.5, 39.5, lat_len)
interp_data = data.interp(latitude=lat, longitude=lon)
print(interp_data)
# print(interp_data.shape)
interp_data['v10'].isel(time=0).plot()
plt.tight_layout()
plt.show()
# interp_data.to_netcdf('interp_data.nc')

