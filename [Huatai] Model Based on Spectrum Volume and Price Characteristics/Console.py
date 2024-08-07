from Preprocessor import Preprocessor
from Nets import GRU3Freq, init_xavier, seed_setting
from Losses import Corr3Loss
from Trainer import Trainer
from Drawer import Drawer

import torch

seed_setting(42)
TOGGLE = 1

pps = Preprocessor({
                        'minute': '2013-01-01_2024-01-01_1m_Info.csv',
                        'day': '2013-01-01_2024-01-01_1d_Info.csv',
                        'week': '2013-01-01_2024-01-01_1w_Info.csv'
                   },
                   (4, 1),
                   10,
                   'z-score')

dataloader_train, dataloader_test, norm_param = pps.get_train_test_dataloader()


model_obj = GRU3Freq(6, 30, 3)
loss_obj = Corr3Loss(0.01)
optim_obj = torch.optim.Adam(model_obj.parameters(), lr=0.005)


if __name__ == '__main__':
    if TOGGLE:
        trainer = Trainer(
                            {
                                'train': dataloader_train,
                                'test': dataloader_test
                            },
                            {
                                'model': model_obj,
                                'loss': loss_obj,
                                'optim': optim_obj,
                                'epochs': 80,
                                'checkpoint': 3,
                            },
                            mode='regression',
                            path='D:/FILE_PythonTemp/Temp/.venv/Quant/Models/',
                            display='E'
                         )
        model_prototype, state_dict_path = trainer.train()

    drawer = Drawer(
                      {
                        'model': 'D:/FILE_PythonTemp/Temp/.venv/Quant/Models/New_Model.params',
                        'result': 'D:/FILE_PythonTemp/Temp/.venv/Quant/Plots/',
                      },
                      model_obj,
                      dataloader_test,
                      norm_param,
                      True
                   )
    drawer.draw(-1)
