import torch
import torch.nn.functional as F
from torch.nn import Sequential


class CorrMSELoss(torch.nn.Module):
    def __init__(self, weight_corr, weight_value):
        super(CorrMSELoss, self).__init__()
        self.weight_corr = -abs(weight_corr) / (abs(weight_corr) + abs(weight_value))
        self.weight_value = abs(weight_value) / (abs(weight_corr) + abs(weight_value))

    def forward(self, pred, label):
        core = torch.concat((pred, label), dim=0).reshape(2, -1)
        corr = torch.corrcoef(core)[0, 1]
        value = torch.nn.MSELoss()(pred, label)
        return 10 * (self.weight_corr * corr + self.weight_value * value)


class Corr3Loss(torch.nn.Module):
    def __init__(self, weight_intersection):
        super(Corr3Loss, self).__init__()
        self.weight_intersection = weight_intersection

    def forward(self, factor1, factor2, factor3, label):
        core1 = torch.corrcoef(torch.concat((factor1, factor2), dim=0).reshape(2, -1))[0, 1]
        core2 = torch.corrcoef(torch.concat((factor2, factor3), dim=0).reshape(2, -1))[0, 1]
        core3 = torch.corrcoef(torch.concat((factor3, factor1), dim=0).reshape(2, -1))[0, 1]

        main1 = torch.corrcoef(torch.concat((factor1, label), dim=0).reshape(2, -1))[0, 1]
        main2 = torch.corrcoef(torch.concat((factor1, label), dim=0).reshape(2, -1))[0, 1]
        main3 = torch.corrcoef(torch.concat((factor1, label), dim=0).reshape(2, -1))[0, 1]

        return -(main1 + main2 + main3) + self.weight_intersection * (core1 + core2 + core3)
