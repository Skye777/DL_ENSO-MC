"""
@author: Skye Cui
@file: hc_file_merge.py
@time: 2021/6/13 14:11
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

data_attribute = [f'soda_temp/', 'soda_avg_temp.nc', 'temp', f'{hp.soda_dataset_dir}/interp-data/t300.nc']

dataset_s = '1871-01-15'
dataset_e = '2008-12-15'
llat, rlat, llon, rlon = [-40, 40, 120, 280]


def intercept(base_file):
    dataset = xarray.open_dataset(base_file, cache=True, decode_times=True)
    print(dataset)
    lc = dataset.coords['longitude']
    la = dataset.coords['latitude']

    data = dataset.loc[dict(time=slice(dataset_s, dataset_e), longitude=lc[(lc >= llon) & (lc <= rlon)], latitude=la[(la >= llat) & (la <= rlat)])]
    return data


def interpolation(data):
    lat_len, lon_len = 80, 160
    lon = np.linspace(120.25, 279.75, lon_len)
    lat = np.linspace(-39.75, 39.75, lat_len)
    interp_data = data.interp(latitude=lat, longitude=lon)
    print(interp_data)
    return interp_data


def merge_nc_files(attr):
    basefile = attr[0]
    savepath = os.path.join(basefile, attr[1])
    var = attr[2]
    urls = sorted(os.listdir(basefile))
    urls = [os.path.join(basefile, i) for i in urls]
    datasets = [xarray.open_dataset(url, cache=True, decode_times=True).fillna(0)[var].mean(dim='LEV') for url in urls]
    merged = xarray.concat(datasets, 'time')
    merged.to_netcdf(savepath)


def read_data():
    file = os.path.join(f'{hp.soda_dataset_dir}/interp-data/', 't300.nc')
    sea_data = xarray.open_dataset(file, cache=True, decode_times=True)
    print(sea_data)
    sea_data['temp'].isel(time=0).plot()
    plt.tight_layout()
    plt.show()
    # print(sea_data.sst)


def main():
    # merge_nc_files(data_attribute)
    data = intercept(base_file=os.path.join(data_attribute[0], data_attribute[1]))
    interp_data = interpolation(data)
    interp_data.to_netcdf(data_attribute[3])


if __name__ == "__main__":
    read_data()
    # main()
