import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import Sequential


# Global Random Seed Initialization
def seed_setting(seed=42):
    print(f'>>>[Nets]: Using Random Seed {seed}')
    g = torch.Generator()
    g.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_setting(42)


# Xavier Initialization for Full-Connections and CNNs
def init_xavier(layer_model_obj):
    if layer_model_obj is torch.nn.Linear or layer_model_obj is torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(layer_model_obj.weight)


# Linear Full Connection ===============================================================================================
class LFCNet(torch.nn.Module):
    def __init__(self, input_flat, label_flat, linear_list, activate_list):
        super(LFCNet, self).__init__()
        self.input_flat = input_flat
        self.label_flat = label_flat
        self.activate_list = activate_list
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_flat, int(input_flat * linear_list[0])))
        for i in range(1, len(linear_list)):
            if linear_list[i] == -1:
                self.layers.append(torch.nn.Linear(int(linear_list[i - 1] * input_flat), label_flat))
                break
            else:
                self.layers.append(torch.nn.Linear(int(linear_list[i - 1] * input_flat), int(linear_list[i] * input_flat)))

    def forward(self, x):
        x = x.view(-1, self.input_flat)
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                x = self.layers[i](x)
                break
            elif self.activate_list[i] == 'r':
                x = F.relu(self.layers[i](x))
            elif self.activate_list[i] == 's':
                x = F.sigmoid(self.layers[i](x))
            elif self.activate_list[i] == 't':
                x = F.tanh(self.layers[i](x))
            else:
                x = F.relu(self.layers[i](x))
        return x


# ResNet-18 ============================================================================================================
class _ResBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(_ResBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_channel)
        self.bn2 = torch.nn.BatchNorm2d(out_channel)
        self.conv1by1 = torch.nn.Conv2d(in_channel, out_channel, kernel_size=1)
        self.use1by1 = in_channel != out_channel

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.use1by1:
            x = self.conv1by1(x)
        y += x
        return F.relu(y)


class ResNet18:
    def __init__(self, in_channel, out_channel):
        self.in_channel = in_channel
        self.out_channel = out_channel

    def _build_res_block(self, in_channel, out_channel, layers):
        asemble_line = []
        for i in range(layers):
            if i == 0:
                asemble_line.append(_ResBlock(in_channel, out_channel))
            else:
                asemble_line.append(_ResBlock(out_channel, out_channel))
        return asemble_line

    def construct_net(self):
        ini_out_channel = 16 * self.in_channel
        ini_block = Sequential(
            torch.nn.Conv2d(self.in_channel, ini_out_channel, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(ini_out_channel),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        res_net = Sequential(
            ini_block,
            Sequential(* self._build_res_block(ini_out_channel, ini_out_channel, 2)),
            Sequential(* self._build_res_block(ini_out_channel, ini_out_channel * 2, 2)),
            Sequential(* self._build_res_block(ini_out_channel * 2, ini_out_channel * 4, 2)),
            Sequential(* self._build_res_block(ini_out_channel * 4, ini_out_channel * 8, 2)),
            torch.nn.AvgPool2d(kernel_size=1),
            torch.nn.Flatten(),
            torch.nn.Linear(25600, 4096 * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(4096 * 2, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64)
        )
        return res_net


# GRU (Seq-2-Seq) ===================================================================================================
class GRUSeq2Seq(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=1):
        super(GRUSeq2Seq, self).__init__()
        self.Gru = torch.nn.GRU(in_dims, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        o, _ = self.Gru(x)
        return o


# GRU (Seq-2-Point(s)) =================================================================================================
class GRUSeq2Point(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=1):
        super(GRUSeq2Point, self).__init__()
        self.Gru = torch.nn.GRU(in_dims, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        o, _ = self.Gru(x)
        o = o[:, -1, :]
        return o


# LSTM (Seq-2-Seq) =====================================================================================================
class LSTMSeq2Seq(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, out_dim):
        super(LSTMSeq2Seq, self).__init__()
        self.out_dim = out_dim
        self.Lstm = torch.nn.LSTM(in_dims, hidden_dim, num_layers=3, batch_first=True)

    def forward(self, x):
        o, _ = self.Lstm(x)
        return o


# LSTM (Seq-2-Point(s)) ================================================================================================
class LSTMSeq2Point(torch.nn.Module):
    def __init__(self, in_dims, hidden_dim, num_layers=1):
        super(LSTMSeq2Point, self).__init__()
        self.Lstm = torch.nn.LSTM(in_dims, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        o, _ = self.Lstm(x)
        o = o[:, -1, :]
        return o


# Attention (Stylized) =================================================================================================
class Attention(torch.nn.Module):
    def __init__(self, in_features):
        super(Attention, self).__init__()
        self.Linear = torch.nn.Linear(in_features, in_features)

    def forward(self, x):
        o = F.tanh(self.Linear(x))
        o = F.softmax(o, dim=1)
        return o


# Hybrid ===============================================================================================================
class ResNet18GRUF(torch.nn.Module):
    def __init__(self, in_channel, out_channel, in_dim, hidden_dim):
        super(ResNet18GRUF, self).__init__()
        self.ResNet = ResNet18(in_channel, out_channel).construct_net()
        self.GRUSeq2Seq = GRUSeq2Seq(in_dim, hidden_dim, 1)
        self.Dropout1 = torch.nn.Dropout(0.1)
        self.in_dim = in_dim

    def forward(self, x):
        with torch.no_grad():
            batch_size = x.shape[0]
            seq = x.shape[1]
            channel = x.shape[2]
            height = x.shape[3]
            width = x.shape[4]
        x = x.view(batch_size * seq, channel, height, width)
        x = self.ResNet(x)
        x = self.Dropout1(x)
        x = x.view(batch_size, seq, self.in_dim)
        x = self.GRUSeq2Seq(x)
        return x


class GRU15MIN(torch.nn.Module):
    def __init__(self, gru_in, gru_hidden, gru_layer, linear_out):
        super(GRU15MIN, self).__init__()
        self.Gru = GRUSeq2Point(gru_in, gru_hidden, gru_layer)
        self.BatchNorm = torch.nn.BatchNorm1d(gru_hidden)
        self.Linear = torch.nn.Linear(gru_hidden, linear_out)

    def forward(self, x):
        x = self.Gru(x)
        x = self.BatchNorm(x)
        x = self.Linear(x)
        return x


class GRUAttention(torch.nn.Module):
    def __init__(self, gru_in, gru_hidden, gru_layer, linear_out):
        super(GRUAttention, self).__init__()
        self.Gru = GRUSeq2Seq(gru_in, gru_hidden, gru_layer)
        self.AttLinear = Attention(gru_hidden)
        self.BatchNorm = torch.nn.BatchNorm1d(2 * gru_hidden)
        self.Linear = torch.nn.Linear(2 * gru_hidden, linear_out)

    def forward(self, x):
        x = self.Gru(x)
        a = self.AttLinear(x)

        o = torch.mul(x, a)
        o = torch.sum(o, dim=1)

        o = torch.concat((x[:, -1, :], o), dim=1)
        o = self.BatchNorm(o)
        o = self.Linear(o)
        return o


class GRU15MinGRUDay(torch.nn.Module):
    def __init__(self, min_config, day_config, linear_out):
        super(GRU15MinGRUDay, self).__init__()
        self.min_gru_in = min_config['gru_in']
        self.min_gru_hidden = min_config['gru_hidden']
        self.min_gru_layer = min_config['gru_layer']

        self.day_gru_in = day_config['gru_in']
        self.day_gru_hidden = day_config['gru_hidden']
        self.day_gru_layer = day_config['gru_layer']

        self.GruM = GRUSeq2Point(self.min_gru_in, self.min_gru_hidden, num_layers=self.min_gru_layer)
        self.GruD = GRUSeq2Point(self.day_gru_in, self.day_gru_hidden, num_layers=self.day_gru_layer)
        self.BatchNormM = torch.nn.BatchNorm1d(self.min_gru_hidden)
        self.BatchNormD = torch.nn.BatchNorm1d(self.day_gru_hidden)
        self.Linear = torch.nn.Linear(self.min_gru_hidden + self.day_gru_hidden, linear_out)

    def forward(self, m, d):
        m = self.GruM(m)
        m = self.BatchNormM(m)

        d = self.GruD(d)
        d = self.BatchNormD(d)

        x = torch.concat((m, d), dim=1)
        x = self.Linear(x)
        return x


class GRU2Face(torch.nn.Module):
    def __init__(self, min_config, day_config, linear_out, toggle=False):
        super(GRU2Face, self).__init__()
        self.min_gru_in = min_config['gru_in']
        self.min_gru_hidden = min_config['gru_hidden']
        self.min_gru_layer = min_config['gru_layer']

        self.day_gru_in = day_config['gru_in']
        self.day_gru_hidden = day_config['gru_hidden']
        self.day_gru_layer = day_config['gru_layer']

        self.toggle = toggle

        self.GruM = GRUSeq2Point(self.min_gru_in, self.min_gru_hidden, num_layers=self.min_gru_layer)
        self.GruD = GRUSeq2Point(self.day_gru_in, self.day_gru_hidden, num_layers=self.day_gru_layer)
        self.BatchNormM = torch.nn.BatchNorm1d(self.min_gru_hidden)
        self.BatchNormD = torch.nn.BatchNorm1d(self.day_gru_hidden)
        self.Linear = torch.nn.Linear(self.day_gru_hidden, linear_out)

    def forward(self, m, d):
        d = self.GruD(d)
        d = self.BatchNormD(d)

        if self.toggle:
            m = self.GruM(m)
            m = self.BatchNormM(m)

            x = m + d
        else:
            x = d

        x = self.Linear(x)
        return x


print(f'>>>[Nets]: Done!')
