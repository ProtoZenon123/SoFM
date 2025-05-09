import torch.nn as nn


class SiameseCnnContrastOrigin(nn.Module):
    def __init__(self, channels_img, num_features):
        super(SiameseCnnContrastOrigin, self).__init__()
        self.layer = nn.Sequential(
            # 3*100*100
            self._block(channels_img, 4, 3),
            self._block(4, 8, 3),
            self._block(8, 8, 3),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5),
        )

    def _block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def first_forward(self, input):
        output = self.layer(input)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input_01, input_02):
        output_01 = self.first_forward(input_01)
        output_02 = self.first_forward(input_02)
        return output_01, output_02
