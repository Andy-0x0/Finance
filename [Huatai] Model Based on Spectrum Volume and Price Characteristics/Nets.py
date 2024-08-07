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
class GRU3Freq(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers=3):
        super(GRU3Freq, self).__init__()
        self.Rnn = GRUSeq2Point(in_dim, hidden_dim, num_layers)

        self.BatchNormDay = torch.nn.BatchNorm1d(hidden_dim)
        self.BatchNormWeek = torch.nn.BatchNorm1d(hidden_dim)
        self.BatchNormMonth = torch.nn.BatchNorm1d(hidden_dim)

        self.LinearDay = torch.nn.Linear(hidden_dim, 1)
        self.LinearWeek = torch.nn.Linear(hidden_dim, 1)
        self.LinearMonth = torch.nn.Linear(hidden_dim, 1)

    def forward(self, d, w, m):
        d = self.Rnn(d)
        w = self.Rnn(w)
        m = self.Rnn(m)

        d = self.LinearDay(self.BatchNormDay(d))
        w = self.LinearWeek(self.BatchNormWeek(w))
        m = self.LinearMonth(self.BatchNormMonth(m))

        if not self.training:
            return d + w + m

        return d, w, m


print(f'>>>[Nets]: Done!')
