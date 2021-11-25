"""
@author: Skye Cui
@file: pottmp_file_merge.py
@time: 2021/6/13 18:52
@description: 
"""
import os
import xarray
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()

data_attribute = [f'meta-data/pottmp/', 'pottmp.nc', 'pottmp', f'{hp.godas_dataset_dir}/interp-data/t300.nc']

llat, rlat, llon, rlon = [-40, 40, 120, 280]
blev = 300


def intercept(base_file):
    dataset = xarray.open_dataset(base_file, cache=True, decode_times=True)
    print(dataset)
    lc = dataset.coords['lon']
    la = dataset.coords['lat']
    lev = dataset.coords['level']

    data = dataset.loc[dict(level=lev[lev <= blev], lon=lc[(lc >= llon) & (lc <= rlon)], lat=la[(la >= llat) & (la <= rlat)])]['pottmp'].mean(dim='level')-273.15
    return data


def interpolation(attr):
    data = xarray.open_dataset(os.path.join(attr[0], attr[1]), cache=True, decode_times=True)
    lat_len, lon_len = 80, 160
    lon = np.linspace(120.5, 279.5, lon_len)
    lat = np.linspace(-39.5, 39.5, lat_len)
    interp_data = data.interp(lat=lat, lon=lon)
    print(interp_data)
    return interp_data


def merge_nc_files(attr):
    basefile = attr[0]
    savepath = os.path.join(basefile, attr[1])
    urls = sorted(os.listdir(basefile))
    urls = [os.path.join(basefile, i) for i in urls]
    datasets = [intercept(url) for url in urls]
    merged = xarray.concat(datasets, 'time')
    merged.to_netcdf(savepath)


def read_data():
    file = os.path.join(f'{hp.godas_dataset_dir}/interp-data', 't300.nc')
    sea_data = xarray.open_dataset(file, cache=True, decode_times=True)
    print(sea_data)
    sea_data['pottmp'].sel(time='2020-12-01').plot()
    plt.tight_layout()
    plt.show()


def main():
    # merge_nc_files(data_attribute)
    interp_data = interpolation(data_attribute)
    # interp_data['pottmp'].sel(time='2020-12-01').plot()
    # plt.tight_layout()
    # plt.show()
    interp_data.to_netcdf(data_attribute[3])


if __name__ == "__main__":
    read_data()
    # main()
