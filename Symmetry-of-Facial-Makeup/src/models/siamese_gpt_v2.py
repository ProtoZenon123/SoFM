import torch.nn as nn


class SiameseCnnImproved(nn.Module):
    def __init__(self, channels_img, num_features):
        super(SiameseCnnImproved, self).__init__()

        # Convolutional Layers (increasing depth and using GAP)
        self.layer = nn.Sequential(
            self._block(channels_img, num_features, 7),  # 64x56x56
            self._block(num_features, num_features * 2, 5),  # 128x25x25
            self._block(num_features * 2, num_features * 2, 3),  # 128x11x11
            self._block(num_features * 2, num_features * 4, 3),  # 256x5x5
        )

        # Fully Connected Layer (flattening the output)
        self.fc1 = nn.Sequential(
            nn.Linear(num_features * 4 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 128),
        )

    def _block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def first_forward(self, input):
        output = self.layer(input)  # Convolutional Layers
        output = output.view(output.size(0), -1)  # Flatten for FC layers
        output = self.fc1(output)
        return output

    def forward(self, input_01, input_02):
        output_01 = self.first_forward(input_01)
        output_02 = self.first_forward(input_02)
        return output_01, output_02
