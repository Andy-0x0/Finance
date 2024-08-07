import rqdatac

import pandas as pd
from pandas.tseries.offsets import Day, Week, MonthBegin
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

# rqdatac.init()    # Initialization When DownLoad Raw Data From RiceQuant
DATABASE_PATH = 'D:/FILE_PythonTemp/Temp/.venv/Quant/Collected/'

'''Parameter Instruction'''
'''
# info_dict_1d       ->          Dict 
    {
        'price'   ->   str (path for lookup csv containing price factors)
        'vwap'    ->   str (path for lookup csv containing vwap factors)
    },
# info_dict_1m       ->          Dict 
    {
        'price'   ->   str (Path for lookup csv containing price factors)
        'vwap'    ->   str (Path for lookup csv containing vwap factors)
    }
split_ratio          ->     tuple (the length of train dataset, the length of test dataset)
prediction_t         ->     int (the span for a return to be calculated (days))
normalization        ->     str['z-score' | 'min-max'] (method of normalization)
'''


class Preprocessor:
    def __init__(self, info_dict_1d, info_dict_1m, split_ratio=(4, 1), prediction_t=10, normalization='z-score'):
        self.info_1d_df = self._get_valid_dataframe(info_dict_1d['price'])
        self.vwap_1d_df = self._get_valid_dataframe(info_dict_1d['vwap'])
        self.info_1min_df = self._get_valid_dataframe(info_dict_1m['price'])
        self.vwap_1min_df = self._get_valid_dataframe(info_dict_1m['vwap'])

        self.prediction_t = prediction_t
        self.split_ratio = split_ratio
        self.normalization = normalization

    def _get_valid_dataframe(self, filename):
        info_df = pd.read_csv(DATABASE_PATH + filename)
        if 'datetime' in info_df.columns:
            info_df = info_df.set_index('datetime')
        elif 'date' in info_df.columns:
            info_df = info_df.set_index('date')
        else:
            print('Index Dose Not Include either date or datetime!')
            return
        info_df.index = pd.to_datetime(info_df.index)
        info_df = info_df.drop(labels='order_book_id', axis=1)
        return info_df

    def _get_target_idx(self, start_time, end_time, frequency):
        if frequency == 'd':
            target_idx = pd.date_range(start=start_time, end=end_time, freq='B').map(lambda x: str(x.date()))
            return sorted(list(set(target_idx) & set(self.info_1d_df.index.map(lambda x: str(x.date())))))
        elif frequency == 'w':
            target_idx = pd.date_range(start=start_time, end=end_time, freq='W-MON').map(lambda x: str(x.date()))
            return sorted(list(set(target_idx) & set(self.info_1d_df.index.map(lambda x: str(x.date())))))
        elif frequency == '15min':
            target_idx = pd.date_range(start=start_time, end=end_time, freq='15min')
            return sorted(list(set(target_idx) & set(self.info_1min_df.index)))
        else:
            target_idx = pd.date_range(start=start_time, end=end_time, freq='BM').map(lambda x: str(x.date()))
            return sorted(list(set(target_idx) & set(self.info_1d_df.index.map(lambda x: str(x.date())))))

    def _get_open_ser(self, start_time, end_time, frequency):
        if 'min' in frequency:
            temp = self.info_1min_df.loc[self._get_target_idx(start_time, end_time, frequency), 'open']
        else:
            temp = self.info_1d_df.loc[self._get_target_idx(start_time, end_time, frequency), 'open']
        return temp

    def _get_close_ser(self, start_time, end_time, frequency):
        if 'min' in frequency:
            temp = self.info_1min_df.loc[self._get_target_idx(start_time, end_time, frequency), 'close']
        else:
            temp = self.info_1d_df.loc[self._get_target_idx(start_time, end_time, frequency), 'close']
        return temp

    def _get_high_ser(self, start_time, end_time, frequency):
        if 'min' in frequency:
            temp = self.info_1min_df.loc[self._get_target_idx(start_time, end_time, frequency), 'high']
        else:
            temp = self.info_1d_df.loc[self._get_target_idx(start_time, end_time, frequency), 'high']
        return temp

    def _get_low_ser(self, start_time, end_time, frequency):
        if 'min' in frequency:
            temp = self.info_1min_df.loc[self._get_target_idx(start_time, end_time, frequency), 'low']
        else:
            temp = self.info_1d_df.loc[self._get_target_idx(start_time, end_time, frequency), 'low']
        return temp

    def _get_volume_ser(self, start_time, end_time, frequency):
        if 'min' in frequency:
            temp = self.info_1min_df.loc[self._get_target_idx(start_time, end_time, frequency), 'volume']
        else:
            temp = self.info_1d_df.loc[self._get_target_idx(start_time, end_time, frequency), 'volume']
        return temp

    def _get_vwap_ser(self, start_time, end_time, frequency):
        if 'min' in frequency:
            temp = self.vwap_1min_df.loc[self._get_target_idx(start_time, end_time, frequency)]
        else:
            temp = self.vwap_1d_df.loc[self._get_target_idx(start_time, end_time, frequency)]
        return temp

    def _get_xs(self, start_time, end_time, frequency):
        df_xs = pd.concat([self._get_open_ser(start_time, end_time, frequency),
                           self._get_high_ser(start_time, end_time, frequency),
                           self._get_low_ser(start_time, end_time, frequency),
                           self._get_close_ser(start_time, end_time, frequency),
                           self._get_vwap_ser(start_time, end_time, frequency),
                           self._get_volume_ser(start_time, end_time, frequency)],
                          axis=1)
        df_xs.columns = ['open', 'high', 'low', 'close', 'vwap', 'volume']
        return df_xs

    def _get_labels(self, start_time, end_time, frequency):
        if 'min' in frequency:
            df_temp = self.info_1min_df.loc[self._get_target_idx(start_time, end_time, frequency), 'close'].copy()
        else:
            df_temp = self.info_1d_df.loc[self._get_target_idx(start_time, end_time, frequency), 'close'].copy()

        df_label = ((df_temp.shift(self.prediction_t) - df_temp) / df_temp).bfill()
        df_label.name = 'return'
        return df_label

    def _normalization(self, train_data, test_data, method='z-score'):
        if method == 'min-max':
            for col in range(0, len(train_data.columns) - 1):
                chunk = np.concatenate(train_data.iloc[:, col].copy().tolist(), axis=0)
                min_line = chunk.min(axis=0)
                max_line = chunk.max(axis=0)
                train_data.iloc[:, col] = train_data.iloc[:, col].map(lambda x: (x - min_line) / (max_line - min_line))
                test_data.iloc[:, col] = test_data.iloc[:, col].map(lambda x: (x - min_line) / (max_line - min_line))

            label_max = train_data.iloc[:, -1].max(axis=0)
            label_min = train_data.iloc[:, -1].min(axis=0)
            train_data.iloc[:, -1] = (train_data.iloc[:, -1] - label_min) / (label_max - label_min)
            test_data.iloc[:, -1] = (test_data.iloc[:, -1] - label_min) / (label_max - label_min)
            return train_data, test_data, ['min_max', label_min, label_max]

        elif method == 'z-score':
            for col in range(0, len(train_data.columns) - 1):
                chunk = np.concatenate(train_data.iloc[:, col].copy().tolist(), axis=0)
                mean_line = chunk.mean(axis=0)
                std_line = chunk.std(axis=0)
                train_data.iloc[:, col] = train_data.iloc[:, col].map(lambda x: (x - mean_line) / std_line)
                test_data.iloc[:, col] = test_data.iloc[:, col].map(lambda x: (x - mean_line) / std_line)

            label_mean = train_data.iloc[:, -1].mean(axis=0)
            label_std = train_data.iloc[:, -1].std(axis=0)
            train_data.iloc[:, -1] = (train_data.iloc[:, -1] - label_mean) / label_std
            test_data.iloc[:, -1] = (test_data.iloc[:, -1] - label_mean) / label_std
            return train_data, test_data, ['z-score', label_mean, label_std]

        else:
            return train_data, test_data, ['z-score', 0, 1]

    # The Index This Method Produces is Marking the First Day For the 10-day-long Prediction (D+21)
    def get_train_test_data_15min(self):
        df_labels_d = self._get_labels(self.info_1d_df.index[0], self.info_1d_df.index[-1], 'd')
        df_xs_15min = self._get_xs(self.info_1d_df.index[0], self.info_1d_df.index[-1], '15min')

        # Create Rolling Window DataSets
        def rolling_dataset_15min(l_d, x_15min, given_steps_d, predicting_steps_d):
            collector_x = []
            collector_l = []
            for i in range(0, len(l_d) - (given_steps_d + predicting_steps_d)):
                start = str(x_15min.loc[str(l_d.index[i].date()), :].index[0])
                end = str(x_15min.loc[str(l_d.index[i + given_steps_d - 1].date()), :].index[-1])

                collector_x.append(x_15min.loc[start:end, :].values)
                collector_l.append(l_d.iloc[i + given_steps_d])

            index_d = l_d.index[given_steps_d: len(l_d) - predicting_steps_d]
            return pd.DataFrame({'x': collector_x, 'label': collector_l}, index=index_d)

        total_data = rolling_dataset_15min(df_labels_d, df_xs_15min, 20, 10)

        spliter = int((len(total_data) * self.split_ratio[0]) / (self.split_ratio[0] + self.split_ratio[1]))
        test_data = total_data.iloc[spliter:, :]
        train_data = total_data.iloc[:spliter, :]

        # Normalization
        train_data, test_data, norm_config = self._normalization(train_data, test_data, self.normalization)

        print(f'>>>[Preprocessor]: Done!')
        return train_data, test_data, norm_config

    # The Index This Method Produces is Marking the First Day For the 10-day-long Prediction (D+21)
    def get_train_test_data_15min_1d(self):
        df_xs_d = self._get_xs(self.info_1d_df.index[0], self.info_1d_df.index[-1], 'd')
        df_labels_d = self._get_labels(self.info_1d_df.index[0], self.info_1d_df.index[-1], 'd')
        df_xs_15min = self._get_xs(self.info_1d_df.index[0], self.info_1d_df.index[-1], '15min')

        def rolling_dataset_15min(l_d, x_15min, given_steps_d, predicting_steps_d):
            collector_x = []
            collector_l = []
            for i in range(0, len(l_d) - (given_steps_d + predicting_steps_d)):
                start = str(x_15min.loc[str(l_d.index[i].date()), :].index[0])
                end = str(x_15min.loc[str(l_d.index[i + given_steps_d - 1].date()), :].index[-1])

                collector_x.append(x_15min.loc[start:end, :].values)
                collector_l.append(l_d.iloc[i + given_steps_d])

            index_d = l_d.index[given_steps_d: len(l_d) - predicting_steps_d]
            return pd.DataFrame({'x': collector_x, 'label': collector_l}, index=index_d)

        def rolling_dataset_1d(l_d, x_d, given_steps_d, predicting_steps_d):
            collector_x = []
            collector_l = []
            for i in range(0, len(l_d) - (given_steps_d + predicting_steps_d)):
                start = i
                end = i + given_steps_d

                collector_x.append(x_d.iloc[start:end, :].values)
                collector_l.append(l_d.iloc[i + given_steps_d])

            index_d = l_d.index[given_steps_d: len(l_d) - predicting_steps_d]
            return pd.DataFrame({'x': collector_x, 'label': collector_l}, index=index_d)

        total_data_min = rolling_dataset_15min(df_labels_d, df_xs_15min, 20, 10).drop(labels='label', axis=1)
        total_data_day = rolling_dataset_1d(df_labels_d, df_xs_d, 40, 10)
        total_data = pd.concat([total_data_min, total_data_day], axis=1).dropna(axis=0)
        total_data.columns = ['min_x', 'day_x', 'l']

        spliter = int((len(total_data) * self.split_ratio[0]) / (self.split_ratio[0] + self.split_ratio[1]))
        test_data = total_data.iloc[spliter:, :]
        train_data = total_data.iloc[:spliter, :]

        # Normalization
        train_data, test_data, norm_config = self._normalization(train_data, test_data, self.normalization)

        print(f'>>>[Preprocessor]: Done!')
        return train_data, test_data, norm_config




