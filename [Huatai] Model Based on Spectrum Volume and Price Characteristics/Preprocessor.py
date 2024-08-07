import rqdatac

import pandas as pd
from pandas.tseries.offsets import Day, Week, MonthBegin
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# rqdatac.init()    # Initialization When DownLoad Raw Data From RiceQuant
DATABASE_PATH = 'D:/FILE_PythonTemp/Temp/.venv/Quant/Collected/'


'''Parameter Instruction'''
'''
info_dict_1d {
    'price'          ->     str (path for lookup csv containing price factors)
    'vwap'           ->     str (path for lookup csv containing vwap factors)
},
info_dict_1m {
    'price'          ->     str (Path for lookup csv containing price factors)
    'vwap'           ->     str (Path for lookup csv containing vwap factors)
}
split_ratio          ->     tuple (the length of train dataset, the length of test dataset)
prediction_t         ->     int (the span for a return to be calculated (days))
normalization        ->     str['z-score' | 'min-max'] (method of normalization)
'''


class Preprocessor:
    def __init__(self, info_dict, split_ratio=(4, 1), prediction_t=10, normalization='z-score'):
        self.info_1m_df = self._get_valid_dataframe(info_dict['minute'])
        self.info_1d_df = self._get_valid_dataframe(info_dict['day'])
        self.info_1w_df = self._get_valid_dataframe(info_dict['week'])

        temp = self.info_1d_df.copy()
        temp['date'] = temp.index
        self.info_1M_df = temp.resample('BME').agg({'open': 'first',
                                                    'high': 'max',
                                                    'low': 'min',
                                                    'close': 'last',
                                                    'volume': 'sum',
                                                    'date': 'max'}).set_index('date')

        self.vwap_1d_df = self._get_n_vwap(frequency='day')
        self.vwap_1w_df = self._get_n_vwap(frequency='week')
        self.vwap_1M_df = self._get_n_vwap(frequency='month')

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

    def _get_n_vwap(self, frequency='day'):
        def _get_daily_vwap(min_info_df, day_info_df):
            collector = []

            for curr_date, _ in day_info_df.iterrows():
                chunk = min_info_df.loc[str(curr_date.date()), :]

                price = (chunk.loc[:, 'close'] + chunk.loc[:, 'high'] + chunk.loc[:, 'low']) / 3
                weighted_price = (price * chunk.loc[:, 'volume']).sum(axis=0)
                total_volume = chunk.loc[:, 'volume'].sum(axis=0)

                vwap = weighted_price / total_volume
                collector.append(vwap)

            return pd.Series(collector, index=day_info_df.index, name='vwap').interpolate(
                method='akima').bfill().ffill()

        def _get_weekly_vwap(day_info_df, week_info_df):
            prev_date = day_info_df.index[1]
            day_index = sorted(list(day_info_df.index))
            collector = []

            for curr_date, _ in week_info_df.iterrows():
                total_volume = 0
                weighted_price = 0
                for date in day_index[day_index.index(prev_date):]:
                    if date <= curr_date:
                        price = (day_info_df.loc[date, 'close'] + day_info_df.loc[date, 'high'] + day_info_df.loc[
                            date, 'low']) / 3
                        weighted_price += price * day_info_df.loc[date, 'volume']
                        total_volume += day_info_df.loc[date, 'volume']
                    else:
                        prev_date = date
                        break

                vwap = weighted_price / total_volume
                collector.append(vwap)

            return pd.Series(collector, index=week_info_df.index, name='vwap').interpolate(
                method='akima').bfill().ffill()

        def _get_monthly_vwap(day_info_df, month_info_df):
            prev_date = day_info_df.index[1]
            day_index = sorted(list(day_info_df.index))
            collector = []

            for curr_date, _ in month_info_df.iterrows():
                total_volume = 0
                weighted_price = 0

                for date in day_index[day_index.index(prev_date):]:
                    if date <= curr_date:

                        price = (day_info_df.loc[date, 'close'] + day_info_df.loc[date, 'high'] + day_info_df.loc[
                            date, 'low']) / 3
                        weighted_price += price * day_info_df.loc[date, 'volume']
                        total_volume += day_info_df.loc[date, 'volume']
                    else:
                        prev_date = date
                        break

                vwap = weighted_price / total_volume
                collector.append(vwap)

            return pd.Series(collector, index=month_info_df.index, name='vwap').interpolate(method='akima').bfill().ffill()

        if frequency == 'day':
            return _get_daily_vwap(self.info_1m_df, self.info_1d_df)
        if frequency == 'week':
            return _get_weekly_vwap(self.info_1d_df, self.info_1w_df)
        else:
            return _get_monthly_vwap(self.info_1d_df, self.info_1M_df)

    def _get_xs(self, frequency='day'):
        if frequency == 'day':
            xs_df = pd.concat([self.info_1d_df.loc[:, 'open'],
                               self.info_1d_df.loc[:, 'high'],
                               self.info_1d_df.loc[:, 'low'],
                               self.info_1d_df.loc[:, 'close'],
                               self.info_1d_df.loc[:, 'volume'],
                               self.vwap_1d_df], axis=1)
        elif frequency == 'week':
            xs_df = pd.concat([self.info_1w_df.loc[:, 'open'],
                               self.info_1w_df.loc[:, 'high'],
                               self.info_1w_df.loc[:, 'low'],
                               self.info_1w_df.loc[:, 'close'],
                               self.info_1w_df.loc[:, 'volume'],
                               self.vwap_1w_df], axis=1)
        else:
            xs_df = pd.concat([self.info_1M_df.loc[:, 'open'],
                               self.info_1M_df.loc[:, 'high'],
                               self.info_1M_df.loc[:, 'low'],
                               self.info_1M_df.loc[:, 'close'],
                               self.info_1M_df.loc[:, 'volume'],
                               self.vwap_1M_df], axis=1)

        xs_df.columns = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        return xs_df

    def _get_labels(self):
        df_temp = self.info_1d_df.loc[:, 'close']

        df_label = ((df_temp.shift(self.prediction_t) - df_temp) / df_temp).bfill()
        df_label.name = 'label'
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

    # The Index This Method Produces is Marking the First Day For the 10-day-long Prediction
    def get_train_test_data(self, day_span=40, week_span=20, month_span=12):
        day_xs_df = self._get_xs(frequency='day')
        week_xs_df = self._get_xs(frequency='week')
        month_xs_df = self._get_xs(frequency='month')
        labels_ser = self._get_labels()

        global_start = max(day_xs_df.index[day_span - 1],
                           week_xs_df.index[week_span - 1],
                           month_xs_df.index[month_span - 1])
        labels_ser = labels_ser.loc[global_start:]

        day_index_list = sorted(list(day_xs_df.index))
        week_index_list = sorted(list(week_xs_df.index))
        month_index_list = sorted(list(month_xs_df.index))

        def _init_end_index(frequency='week'):
            if frequency == 'week':
                indexer = 0
                while global_start >= week_index_list[indexer + 1]:
                    indexer += 1
            else:
                indexer = 0
                while global_start >= month_index_list[indexer + 1]:
                    indexer += 1

            return indexer

        week_end_idx = _init_end_index('week')
        month_end_idx = _init_end_index('month')

        collector_day = []
        collector_week = []
        collector_month = []
        for curr_date, label in labels_ser.items():
            day_end_idx = day_index_list.index(curr_date)
            day_start_idx = day_end_idx - day_span + 1
            day_end = curr_date
            day_start = day_index_list[day_start_idx]
            collector_day.append(day_xs_df.loc[day_start:day_end, :].to_numpy())

            if week_index_list[week_end_idx + 1] <= curr_date:
                week_end_idx += 1
            week_start_idx = week_end_idx - week_span + 1
            week_end = week_index_list[week_end_idx]
            week_start = week_index_list[week_start_idx]
            collector_week.append(week_xs_df.loc[week_start:week_end, :].to_numpy())

            if month_index_list[month_end_idx + 1] <= curr_date:
                month_end_idx += 1
            month_start_idx = month_end_idx - month_span + 1
            month_end = month_index_list[month_end_idx]
            month_start = month_index_list[month_start_idx]
            collector_month.append(month_xs_df.loc[month_start:month_end, :].to_numpy())

        content_df = pd.DataFrame({'day': collector_day,
                                   'week': collector_week,
                                   'month': collector_month,
                                   'label': labels_ser.tolist()}, index=labels_ser.index)

        spliter = int((len(content_df) * self.split_ratio[0]) / (self.split_ratio[0] + self.split_ratio[1]))
        train_set = content_df.iloc[:spliter, :]
        test_set = content_df.iloc[spliter:, :]

        return self._normalization(train_set, test_set, 'z-score')

    def get_train_test_dataloader(self, day_span=40, week_span=20, month_span=12):
        train_df, test_df, norm_param = self.get_train_test_data(day_span, week_span, month_span)

        class DataSet(Dataset):
            def __init__(self, info_df):
                super(DataSet, self).__init__()
                self.info_df = info_df

            def __getitem__(self, index):
                return ((torch.from_numpy(np.array(self.info_df.iloc[index, 0].tolist()).astype(np.float32)),
                         torch.from_numpy(np.array(self.info_df.iloc[index, 1].tolist()).astype(np.float32)),
                         torch.from_numpy(np.array(self.info_df.iloc[index, 2].tolist()).astype(np.float32))),
                        torch.tensor(self.info_df.iloc[index, -1]).unsqueeze(-1))

            def __len__(self):
                return len(self.info_df)

        train_dataloader = DataLoader(DataSet(train_df),
                                      batch_size=64,
                                      shuffle=True)
        test_dataloader = DataLoader(DataSet(test_df),
                                     batch_size=64,
                                     shuffle=False)

        return train_dataloader, test_dataloader, norm_param


# Using Sample
# pps = Preprocessor({
#                         'minute': '2013-01-01_2024-01-01_1m_Info.csv',
#                         'day': '2013-01-01_2024-01-01_1d_Info.csv',
#                         'week': '2013-01-01_2024-01-01_1w_Info.csv'
#                    },
#                    (4, 1),
#                    10,
#                    'z-score')
#
# dataloader_train, dataloader_test, norm_param = pps.get_train_test_dataloader()
# for (day, week, month), label in dataloader_train:
#     print(day.shape)
#     print(week.shape)
#     print(month.shape)
#     print(label.shape)
#     print()
#     break

