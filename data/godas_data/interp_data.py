"""
@author: Skye Cui
@file: interp_data.py
@time: 2021/6/8 15:30
@description: 
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

llat, rlat, llon, rlon = [-40, 40, 120, 280]
dic = [
    [f'{hp.godas_dataset_dir}/meta-data/sst.mnmean.nc', f'{hp.godas_dataset_dir}/interp-data/sst.nc', 'sst', 'lat'],
    [f'{hp.godas_dataset_dir}/meta-data/sshg.mon.mean1980-2020.nc', f'{hp.godas_dataset_dir}/interp-data/sshg.nc', 'sshg', 'lat'],
    [f'{hp.godas_dataset_dir}/meta-data/uwind.mon.mean1980-2019.nc', f'{hp.godas_dataset_dir}/interp-data/uwind.nc', 'u10', 'latitude'],
    [f'{hp.godas_dataset_dir}/meta-data/vwind.mon.mean1980-2019.nc', f'{hp.godas_dataset_dir}/interp-data/vwind.nc', 'v10', 'latitude'],
]


def intercept(base_file, coords):
    dataset = xarray.open_dataset(base_file, cache=True, decode_times=True)
    print(dataset)
    if coords == 'lat':
        lc = dataset.coords['lon']
        la = dataset.coords['lat']
        data = dataset.loc[dict(lon=lc[(lc >= llon) & (lc <= rlon)], lat=la[(la >= llat) & (la <= rlat)])]
    else:
        lc = dataset.coords['longitude']
        la = dataset.coords['latitude']
        data = dataset.loc[dict(longitude=lc[(lc >= llon) & (lc <= rlon)], latitude=la[(la >= llat) & (la <= rlat)])]
    return data


def interpolation(data, coords):
    lat_len, lon_len = 80, 160
    lon = np.linspace(120.25, 279.75, lon_len)
    lat = np.linspace(-39.75, 39.75, lat_len)
    if coords == 'lat':
        interp_data = data.interp(lat=lat, lon=lon)
    else:
        interp_data = data.interp(latitude=lat, longitude=lon)
    print(interp_data)
    return interp_data


if __name__ == '__main__':
    for dataset in dic:
        data = intercept(dataset[0], dataset[3])
        interp_data = interpolation(data, dataset[3])
        interp_data.to_netcdf(dataset[1])
