import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

'''
loader_dict:    {
    train:      dataloader                      ->  the dataloader of train datasets
    test:       dataloader                      ->  the dataloader of test datasets
}

loader_dict:    {
    model:      ModelObj                        ->  the modelObj of training 
    optim:      optimObj                        ->  the optimObj of training
    loss:       lossObj                         ->  the lossObj of training
    epochs:     int                             ->  the epoch number of training
    checkpoint: int                             ->  the checkpoint for loss/batch update
}

batch_size:     int                             ->  the batch_size for training
'''


class Trainer:
    def __init__(self, loader_dict, train_dict, mode='none', path='fd_prediction/', display='E'):
        self.train_loader = loader_dict['train']
        self.test_loader = loader_dict['test']

        self.modelFunct = train_dict['model']
        self.optimizer = train_dict['optim']
        self.lossFunct = train_dict['loss']
        self.epochs = train_dict['epochs']
        self.checkpoint = train_dict['checkpoint']

        self.mode = mode
        self.path = path
        self.display = display

    def _device_decide(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def train(self):
        loss_collector_train = []
        outer_loss = 0
        outer_scaler = 0
        acc_collector_test = []
        self.modelFunct = self.modelFunct.to(self._device_decide())
        self.modelFunct.train()

        for epoch in range(self.epochs):
            if self.display == 'B':
                print(f'[Epoch]: {epoch + 1}/{self.epochs} ================')

            for batch_idx, (x, label) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                label = label.to(self._device_decide())
                if type(x) is tuple or type(x) is list:
                    x = map(lambda ele: ele.to(self._device_decide()), x)
                    pred_label = self.modelFunct(*x)
                    if type(pred_label) is tuple or type(pred_label) is list:
                        loss = self.lossFunct(* pred_label, label)
                    else:
                        loss = self.lossFunct(pred_label, label)
                else:
                    x = x.to(self._device_decide())
                    pred_label = self.modelFunct(x)
                    if type(pred_label) is tuple or type(pred_label) is list:
                        loss = self.lossFunct(* pred_label, label)
                    else:
                        loss = self.lossFunct(pred_label, label)

                loss.backward()

                self.optimizer.step()

                if self.display == 'E':
                    outer_loss += loss.item()
                    outer_scaler += len(label)
                else:
                    if (batch_idx + 1) % self.checkpoint == 0:
                        print(f'\t[Batch]: {batch_idx + 1} | [Loss]: {loss.item():.3f}')
                        loss_collector_train.append(loss.item())

            if self.display == 'E':
                print(f'[Epoch]: {epoch + 1}/{self.epochs} | ', end='')
                print(f'[Loss]: {outer_loss / outer_scaler:.3f}')
                loss_collector_train.append(outer_loss / outer_scaler)
                outer_loss, outer_scaler = 0, 0
            else:
                print()

            if self.mode == 'classification':
                acc = self._test_on_test_classification()
                acc_collector_test.append(acc)

        loss_df = pd.Series(loss_collector_train, index=np.arange(1, len(loss_collector_train) + 1), name='Loss')
        loss_df.plot(kind='line',
                     figsize=(16, 9),
                     lw=0.9,
                     xlabel='Epochs (Batches)',
                     ylabel='Loss Values',
                     title='Training Summary',
                     color='red',
                     )
        plt.show()
        plt.close()
        torch.save(self.modelFunct.state_dict(), self.path + 'New_Model.params')

        if self.mode == 'classification':
            plt.plot(np.linspace(1, len(loss_collector_train), self.epochs), acc_collector_test, color='blue',
                     label='Acc on Test')
            plt.show()
            plt.close()
        elif self.mode == 'regression':
            self._test_on_test_regression(200)

        print('>>>[Trainer]: Done!')
        return self.modelFunct, self.path + 'New_Model.params'

    def _test_on_test_classification(self):
        correct = 0
        total = 0
        for idx, (data, label) in enumerate(self.test_loader):
            data = data.to(device=self._device_decide())
            label = label.to(device=self._device_decide())
            pred_label = self.modelFunct(data)

            correct += (pred_label.argmax(dim=1) == label).sum(axis=0).to(device=torch.device('cpu'))
            total += label.numel()

        print(f'\t[Accuracy]  {100 * (correct / total):.2f}%')
        return correct / total

    def _test_on_test_regression(self, clip=200):
        pred_collector = []
        label_collector = []
        self.modelFunct = self.modelFunct.to(device=torch.device('cpu'))
        self.modelFunct.eval()

        for idx, (data, label) in enumerate(self.train_loader):
            label = label.to(device=torch.device('cpu'))
            if type(data) is tuple or type(data) is list:
                data = map(lambda ele: ele.to(device=torch.device('cpu')), data)
                pred_label = self.modelFunct(* data)
            else:
                data = data.to(device=torch.device('cpu'))
                pred_label = self.modelFunct(data)

            if pred_label.shape[1] > 1:
                chip_p = pred_label.detach().squeeze().numpy()[:, -1]
                chip_l = label.detach().squeeze().numpy()[:, -1]
            else:
                chip_p = pred_label.detach().squeeze().numpy()
                chip_l = label.detach().squeeze().numpy()

            pred_collector.extend(chip_p)
            label_collector.extend(chip_l)

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
                 title='Fit on Training Data',
                 color=('red', 'blue'))
        plt.show()
        plt.close()


