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
