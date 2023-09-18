from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, Upsample, \
    LeakyReLU, Conv1d, Tanh
from torch.autograd import Variable
from torch.optim import Adam, SGD
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from scipy.signal import decimate
import numpy as np


class Wave_U_Net(nn.Module):
    def __init__(self, L, K, Fc, fd, fu):
        super(Wave_U_Net, self).__init__()
        self.L = L
        self.K = K
        self.Fc = Fc
        self.DS_layers = []
        nrLayers = 1
        for I in range(L):
            i = I + 1
            self.DS_layers.append(Sequential(
                Conv1d(nrLayers, Fc * i, (fd, fd)),
                LeakyReLU()
            ))
            nrLayers = Fc * i

        self.middleConv = Conv1d(nrLayers, Fc * (L + 1), (fd, fd))

        nrLayers = Fc * (L + 1)
        self.US_layers = []
        for I in range(L):
            i = L - I
            self.US_layers.append([Upsample(scale_factor=2),Sequential(
                Conv1d(nrLayers, Fc * i, (fu, fu)),
                LeakyReLU()
            )])
            nrLayers = Fc * i
        self.last_layers = Sequential(
            Conv1d(nrLayers, K, (1, 1)),
            Tanh()
        )

    def forward(self, x):
        DS_blocks = []
        for layer in self.DS_layers:
            x = layer(x)
            x = decimate(x, 2)
            DS_blocks.append(x)

        x = self.middleConv(x)

        for idx, layer in enumerate(self.US_layers):
            x = layer[0](x)
            x = np.concatenate([x, DS_blocks[self.L - idx]])
            x = layer[1](x)

        x = np.concatenate(x)
        x = self.last_layers(x)

        return x

    def change_device(self, device):
        for layer in self.DS_layers:
            layer = layer.to(device)

        self.middleConv = self.middleConv.to(device)

        for layer in self.US_layers:
            layer[0] = layer[0].to(device)
            layer[1] = layer[1].to(device)

        self.last_layers = self.last_layers.to(device)



class CNNModule():
    def __init__(self, batch_size, L, K, Fc, fd, fu, criterion=nn.CrossEntropyLoss(), learning_rate=0.01):
        # select gpu (Cuda) if available
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.batch_size = batch_size
        print("Using {}".format(self.device))
        self.model = Wave_U_Net(L, K, Fc, fd, fu)

        self.criterion = criterion

        # set to correct device
        self.model = self.model.to(self.device)
        self.model.change_device(self.device)
        self.criterion = self.criterion.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def train(self, num_epochs, train_dataset, test_dataset):

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size,
                                                  shuffle=False)

        iter = 0
        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Load images
                images = images.requires_grad_()
                images = images.to(self.device)
                labels = labels.to(self.device)
                # Clear gradients w.r.t. parameters
                self.optimizer.zero_grad()

                # Forward pass to get output/logits
                outputs = self.model(images)

                # Calculate Loss: softmax --> cross entropy loss
                loss = self.criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                self.optimizer.step()

                iter += 1

                if iter % 500 == 0:
                    # Calculate Accuracy
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    for images, labels in test_loader:
                        # Load images
                        images = images.requires_grad_()

                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        # Forward pass only to get logits/output
                        outputs = self.model(images)

                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)

                        # Total number of labels
                        total += labels.size(0)

                        # Total correct predictions
                        correct += (predicted == labels).sum()

                    accuracy = 100 * correct / total

                    # Print Loss
                    print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
