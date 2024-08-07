import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Preprocessor import Preprocessor
from Nets import GRU15MIN, GRUAttention, GRU15MinGRUDay, GRU2Face, init_xavier
from Trainer import Trainer
from Drawer import Drawer


class DataSet15Min(Dataset):
    def __init__(self, info_df):
        super(DataSet15Min, self).__init__()
        self.info_df = info_df

    def __getitem__(self, index):
        return (np.array(self.info_df.iloc[index, 0].tolist()).astype(np.float32),
                np.array(self.info_df.iloc[index, -1].tolist()).astype(np.float32).reshape(-1))

    def __len__(self):
        return len(self.info_df)


class DataSet15Min1D(Dataset):
    def __init__(self, info_df):
        super(DataSet15Min1D, self).__init__()
        self.info_df = info_df

    def __getitem__(self, index):
        return ((np.array(self.info_df.iloc[index, 0].tolist()).astype(np.float32),
                 np.array(self.info_df.iloc[index, 1].tolist()).astype(np.float32)),
                np.array(self.info_df.iloc[index, -1].tolist()).astype(np.float32).reshape(-1))

    def __len__(self):
        return len(self.info_df)


class Console:
    def __init__(self, choice, toggle):
        self.choice = choice
        self.toggle = toggle
        self.preprocessor = Preprocessor({'price': '2013-01-01_2024-01-01_1d_Info.csv', 'vwap': '2013-01-01_2024-01-01_1d_Vwap.csv'},
                                         {'price': '2013-01-01_2024-01-01_1m_Info.csv', 'vwap': '2013-01-01_2024-01-01_1m_Vwap.csv'},
                                         (4, 1),
                                         10,
                                         'z-score')

        self.config = [{
                            'model': GRU15MIN(6, 30, 3, 1).apply(init_xavier),
                            'loss': self._loss_corr,
                            'epochs': 50,
                            'total_dataset': self.preprocessor.get_train_test_data_15min,
                            'dataset_class': DataSet15Min,
                            'message': '>>>[Console]: Using Strategy <GRU [15min]>'
                        },
                        {
                            'model': GRUAttention(6, 30, 3, 1).apply(init_xavier),
                            'loss': self._loss_corr,
                            'epochs': 50,
                            'total_dataset': self.preprocessor.get_train_test_data_15min,
                            'dataset_class': DataSet15Min,
                            'message': '>>>[Console]: Using Strategy <GRU+Attention [15min]>'
                        },
                        {
                            'model': GRU15MinGRUDay(
                                                    {
                                                        'gru_in': 6,
                                                        'gru_hidden': 30,
                                                        'gru_layer': 3,
                                                    },
                                                    {
                                                        'gru_in': 6,
                                                        'gru_hidden': 30,
                                                        'gru_layer': 3,
                                                    },
                                                    1).apply(init_xavier),
                            'loss': self._loss_corr,
                            'epochs': 50,
                            'total_dataset': self.preprocessor.get_train_test_data_15min_1d,
                            'dataset_class': DataSet15Min1D,
                            'message': '>>>[Console]: Using Strategy <GRU [15min+1d]>'
                        },
                        {
                            'model': [GRU2Face(
                                             {
                                                 'gru_in': 6,
                                                 'gru_hidden': 30,
                                                 'gru_layer': 3,
                                             },
                                             {
                                                 'gru_in': 6,
                                                 'gru_hidden': 30,
                                                 'gru_layer': 3,
                                             },
                                             1, False).apply(init_xavier),
                                      GRU2Face(
                                             {
                                                 'gru_in': 6,
                                                 'gru_hidden': 30,
                                                 'gru_layer': 3,
                                             },
                                             {
                                                 'gru_in': 6,
                                                 'gru_hidden': 30,
                                                 'gru_layer': 3,
                                             },
                                             1, True).apply(init_xavier)],
                            'loss': self._loss_corr,
                            'epochs': [50, 50],
                            'total_dataset': self.preprocessor.get_train_test_data_15min_1d,
                            'dataset_class': DataSet15Min1D,
                            'message': '>>>[Console]: Using Strategy <GRU+Frozen [15min+1d]>'
                        }]

    def _loss_corr(self, output, target):
        temp = torch.concat((output, target), dim=0).reshape(2, -1)
        loss = -torch.corrcoef(temp)[0, 1]
        return loss + 2 * torch.nn.MSELoss()(output, target)

    def activate(self):
        config = self.config[self.choice]
        print(config['message'])

        train_dataset_raw, test_dataset_raw, norm_config = config['total_dataset']()
        train_dataset, test_dataset = config['dataset_class'](train_dataset_raw), config['dataset_class'](test_dataset_raw)
        train_dataloader, test_dataloader = (DataLoader(dataset=train_dataset, batch_size=50, shuffle=True),
                                             DataLoader(dataset=test_dataset, batch_size=50, shuffle=False))

        if self.toggle:
            model = config['model'] if type(config['model']) is not list else config['model'][0]
            epoch = config['epochs'] if type(config['epochs']) is not list else config['epochs'][0]
            trainer = Trainer({
                'train': train_dataloader,
                'test': test_dataloader
            },
                {
                    'model': model,
                    'loss': config['loss'],
                    'optim': torch.optim.Adam(model.parameters(), lr=0.001),
                    'epochs': epoch,
                    'checkpoint': 3
                },
                mode='regression' if self.choice != 3 else 'single',
                path='Models/',
                display='E')
            model_face1, model_face1_path = trainer.train()

            if self.choice == 3:
                print('>>>[Console]: Face 1 Model Trained')

                config['model'][1].load_state_dict(torch.load(model_face1_path))
                for name, param in config['model'][1].named_parameters():
                    if 'GruD' in name or 'BatchNormD' in name:
                        param.requires_grad = False

                trainer = Trainer({
                    'train': train_dataloader,
                    'test': test_dataloader
                },
                    {
                        'model': config['model'][1],
                        'loss': config['loss'],
                        'optim': torch.optim.Adam(filter(lambda p: p.requires_grad, config['model'][1].parameters()), lr=0.001),
                        'epochs': config['epochs'][1],
                        'checkpoint': 3
                    },
                    mode='regression',
                    path='Models/',
                    display='E')
                trainer.train()
                print('>>>[Console]: Face 2 Model Trained')

        drawer = Drawer({
            'model': 'Models/New_Model.params',
            'result': 'Imgs/New_Model_Test_Result.png'
        },
            config['model'] if type(config['model']) is not list else config['model'][-1],
            test_dataloader,
            norm_config,
            display=True)
        drawer.draw(200)

        print('>>>[Console]: Done!')

