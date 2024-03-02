import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.convB = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU())

    def forward(self, inputs):
        x = self.convB(inputs)
        return x

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
        return x, p

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.upD = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.convD = nn.Sequential(nn.Conv2d(in_channels=out_channels+out_channels, out_channels=out_channels, kernel_size=3, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias = False, padding = 'same'),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU())

    def forward(self, inputs, skip):
        x = self.upD(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.convD(x)
        return x

class UnetBiais(nn.Module):
    def __init__(self, conv_dim = 3):
        super(UnetBiais, self).__init__()
        self.e1 = Encoder(2, 64, conv_dim)
        self.e2 = Encoder(64, 128, conv_dim)
        self.e3 = Encoder(128, 256, conv_dim)
        self.e4 = Encoder(256, 512, conv_dim)

        self.bottleneck = Bottleneck(512, 1024, conv_dim)

        self.d1 = Decoder(1024, 512)
        self.d2 = Decoder(512, 256)
        self.d3 = Decoder(256, 128)
        self.d4 = Decoder(128, 64)

        self.scalar_fc = nn.Linear(4, 1024)

        self.classifier = nn.Sequential(nn.Conv2d(64, out_channels = 2, kernel_size = 1, padding = 0),
            nn.LeakyReLU())

        self.fc = nn.Sequential(nn.Flatten(),
            nn.Linear(2*224*224, 1280),
            nn.LeakyReLU(),
            nn.Linear(1280, 80),
            nn.LeakyReLU(),
            nn.Linear(80, 5),
            nn.Softmax(dim=1))

    def forward(self, x, scalars):
        up1, x = self.e1(x)
        up2, x = self.e2(x)
        up3, x = self.e3(x)
        up4, x = self.e4(x)

        x = self.bottleneck(x)
        scalar_biases = self.scalar_fc(scalars)
        scalar_biases = scalar_biases.view(-1, 1024, 1, 1)
        scalar_biases = scalar_biases.repeat(1, 1, x.size(2), x.size(3))
        x = x + scalar_biases

        x = self.d1(x, up4)
        x = self.d2(x, up3)
        x = self.d3(x, up2)
        x = self.d4(x, up1)
        
        x = self.classifier(x)
        return self.fc(x)