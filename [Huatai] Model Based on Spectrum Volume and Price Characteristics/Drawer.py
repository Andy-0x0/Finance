import os
import random
import datetime
import pandas as pd
from pandas.tseries.offsets import Day
import numpy as np
import matplotlib.pyplot as plt

import torch


class Drawer:
    def __init__(self, path_config, model_prototype, dataloader, reverse_para, display=True):
        self.state_dict_path = path_config['model']
        self.result_path = path_config['result']
        self.model = model_prototype
        self.dataloader = dataloader
        self.reverse_para = reverse_para
        self.display = display

        self.model.load_state_dict(torch.load(self.state_dict_path))
        self.model = self.model.to(device=torch.device('cpu'))
        self.model.eval()

    def draw(self, clip=500):
        pred_collector = []
        label_collector = []

        def reverse_normalization(line, method, fac1, fac2):
            if method == 'z-score':
                return (line * fac2) + fac1
            if method == 'min_max':
                return line * abs(fac1 - fac2) + fac1

        for idx, (data, label) in enumerate(self.dataloader):
            label = label.to(device=torch.device('cpu'))
            if type(data) is tuple or type(data) is list:
                data = map(lambda ele: ele.to(device=torch.device('cpu')), data)
                pred_label = self.model(* data)
            else:
                data = data.to(device=torch.device('cpu'))
                pred_label = self.model(data)

            if pred_label.shape[1] > 1:
                chip_p = pred_label.detach().squeeze().numpy()[:, -1]
                chip_l = label.detach().squeeze().numpy()[:, -1]
            else:
                chip_p = pred_label.detach().squeeze().numpy()
                chip_l = label.detach().squeeze().numpy()

            pred_collector.extend(reverse_normalization(chip_p,
                                                        self.reverse_para[0],
                                                        self.reverse_para[1],
                                                        self.reverse_para[2]))
            label_collector.extend(reverse_normalization(chip_l,
                                                         self.reverse_para[0],
                                                         self.reverse_para[1],
                                                         self.reverse_para[2]))

        if clip > 0:
            res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})
            res = res.iloc[:min(clip, len(res)), :]
        else:
            res = pd.DataFrame({'Prediction': pred_collector, 'Label': label_collector})

        res.plot(kind='line',
                 figsize=(16, 9),
                 alpha=0.8,
                 lw=0.8,
                 xlabel='Samples',
                 ylabel='Values',
                 title='Fit on Testing Data',
                 color=('red', 'blue'))
        plt.grid(axis='y')

        if self.display:
            plt.show()
            plt.close()
        else:
            plt.savefig(self.result_path)

        print(f'>>>[Drawer]: Done!')

