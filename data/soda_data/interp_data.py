"""
@author: Skye Cui
@file: interp_data.py
@time: 2021/6/8 9:44
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
dic = [
    [f'{hp.soda_dataset_dir}/meta-data/soda_ssh.nc', f'{hp.soda_dataset_dir}/interp-data/ssh.nc'],
    [f'{hp.soda_dataset_dir}/meta-data/soda_taux.nc', f'{hp.soda_dataset_dir}/interp-data/taux.nc'],
    [f'{hp.soda_dataset_dir}/meta-data/soda_tauy.nc', f'{hp.soda_dataset_dir}/interp-data/tauy.nc']
]


def intercept(base_file):
    dataset = xarray.open_dataset(base_file, cache=True, decode_times=False)
    print(dataset)
    lc = dataset.coords['lon']
    la = dataset.coords['lat']

    data = dataset.loc[dict(lon=lc[(lc >= llon) & (lc <= rlon)], lat=la[(la >= llat) & (la <= rlat)])]
    return data


def interpolation(data):
    lat_len, lon_len = 80, 160
    lon = np.linspace(120.25, 279.75, lon_len)
    lat = np.linspace(-39.75, 39.75, lat_len)
    interp_data = data.interp(lat=lat, lon=lon)
    print(interp_data)
    return interp_data


if __name__ == '__main__':
    for dataset in dic:
        data = intercept(dataset[0])
        interp_data = interpolation(data)
        interp_data.to_netcdf(dataset[1])
