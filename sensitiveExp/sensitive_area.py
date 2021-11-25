"""
@author: Skye Cui
@file: sensitive_area.py
@time: 2021/7/8 14:03
@description: 
"""
import os
import numpy as np
from component.plot_helper import plot_composite_saliency, plot_time_series_composite_saliency
from hparams import Hparams

hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()


def composition(var):
    comp_saliency = []
    files = [f for f in os.listdir(hp.saliency_npz) if os.path.isfile(os.path.join(hp.saliency_npz, f))]
    for file in files:
        print("saliency:", file)
        saliency = np.load(f"{hp.saliency_npz}/{file}")[var]
        event_comp_saliency = np.sum(saliency, axis=0)
        comp_saliency.append(event_comp_saliency)
        # plot_one(event_comp_saliency)
    comp_saliency = np.sum(np.stack(comp_saliency, axis=0), axis=0)
    plot_composite_saliency(comp_saliency/comp_saliency.max(), ids=var)


def time_series_composition():
    comp_saliency = []
    for file in os.listdir(hp.saliency_npz):
        print("saliency:", file)
        saliency = np.load(f"{hp.saliency_npz}/{file}")['t300']
        comp_saliency.append(saliency)
        # plot_one(event_comp_saliency)
    comp_saliency = np.sum(np.stack(comp_saliency, axis=0), axis=0)
    comp_saliency = comp_saliency[[0, 3, 10, 11]]
    titles = ["12-month lead", "9-month lead", "2-month lead", "1-month lead"]
    plot_time_series_composite_saliency(comp_saliency, titles=titles)
    # for i in range(len(comp_saliency)):
    #     plot_composite_saliency(comp_saliency[i]/comp_saliency[i].max(), ids=str(i+1))


if __name__ == '__main__':
    # composition(var='sst')
    # composition(var='t300')
    # composition(var='ssh')
    time_series_composition()
