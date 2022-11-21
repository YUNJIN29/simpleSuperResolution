from apex.fp16_utils import tofp16, BN_convert_float
from torch import nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=3):
        super(SRCNN, self).__init__()
        self.act_fn = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(9, 9), padding=9 // 2)
        # self.conv2 = nn.Conv2d(64, 32, kernel_size=(5, 5), padding=5 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=(1, 1), padding=1 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=(5, 5), padding=5 // 2)
        self.pad = nn.ZeroPad2d(4)
        m = [
            self.conv1,
            self.act_fn,
            self.conv2,
            self.act_fn,
            self.conv3,
            self.act_fn
        ]
        self.sequential = nn.Sequential(*m)

    def forward(self, x):
        # x = self.pad(x)
        return self.sequential(x)


class Covgg7(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act_fn = nn.LeakyReLU(0.1, inplace=False)
        self.offset = 7
        self.pad = nn.ZeroPad2d(self.offset)
        m = [nn.Conv2d(3, 16, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(16, 32, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(32, 64, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(64, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 128, 3, 1, 0),
             self.act_fn,
             nn.Conv2d(128, 256, 3, 1, 0),
             self.act_fn,
             # in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=
             nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=3, bias=False),
             ]
        self.Sequential = nn.Sequential(*m)

    def forward(self, x):
        x = self.pad(x)
        return self.Sequential.forward(x)

