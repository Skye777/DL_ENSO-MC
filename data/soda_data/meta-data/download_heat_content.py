"""
@author: Skye Cui
@file: download_heat_content.py
@time: 2021/6/12 19:45
@description: 
"""
import urllib.request

START_YEAR = 1884
END_YEAR = 2010
uri = ('http://apdrc.soest.hawaii.edu/erddap/griddap/'
       'hawaii_soest_c71f_e12b_37f8.nc'
       '?temp'
       '[({}-01-15):1:({}-12-15)]'
       '[(5.01):1:(268.46)]'
       '[(-75.25):1:(89.25)][(0.25):1:(359.75)]').format

for i in range(START_YEAR, END_YEAR+1, 1):
    url = uri(i, i)
    print("Downloading Year {} data...".format(i))
    urllib.request.urlretrieve(url=url, filename="soda_temp/temp.{}.nc".format(i))
    print("Data in {} download completed!".format(i))
