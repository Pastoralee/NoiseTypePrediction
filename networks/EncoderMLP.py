import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.convE = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU())

        self.poolE = nn.Sequential(nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size,
                              bias = False, stride = 2, padding = math.ceil((1 - 2 + (kernel_size-1))/2)),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU())

    def forward(self, inputs):
        x = self.convE(inputs)
        p = self.poolE(x)
        return p

class EncoderMLP(nn.Module):
    def __init__(self, conv_dim = 3):
        super(EncoderMLP, self).__init__()
        self.e1 = Encoder(2, 64, conv_dim)
        self.e2 = Encoder(64, 128, conv_dim)
        self.e3 = Encoder(128, 256, conv_dim)
        self.e4 = Encoder(256, 512, conv_dim)
        self.e5 = Encoder(512, 1024, conv_dim)

        self.fc = nn.Sequential(nn.Linear(53, 16),
                                nn.LeakyReLU(),
                                nn.Linear(16, 5),
                                nn.Softmax(dim=1))


        self.classifier = nn.Sequential(nn.Conv2d(1024, out_channels = 1, kernel_size = 1, padding = 0),
            nn.LeakyReLU())

    def forward(self, x, scalars):
        x = self.e1(x)
        x = self.e2(x)
        x = self.e3(x)
        x = self.e4(x)
        x = self.e5(x)
        x = self.classifier(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, scalars], dim=1)
        return self.fc(x)