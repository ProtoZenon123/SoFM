import torch.nn as nn


class SiameseNetwork(nn.Module):
    def __init__(self, channels_img=3, num_features=64, embedding_dim=128):
        """
        Siamese Neural Network for learning similarity between paired facial images.

        Args:
        - channels_img (int): Number of image channels (3 for RGB).
        - num_features (int): Number of filters in the first CNN layer.
        - embedding_dim (int): Final feature embedding size.
        """
        super(SiameseNetwork, self).__init__()

        # **Feature Extraction Layers (CNN)**
        self.feature_extractor = nn.Sequential(
            self._conv_block(channels_img, num_features, kernel_size=10),  # 64 filters
            self._conv_block(
                num_features, num_features * 2, kernel_size=7
            ),  # 128 filters
            self._conv_block(
                num_features * 2, num_features * 2, kernel_size=4
            ),  # 128 filters
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4),  # 256 filters
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),  # Ensures output size is fixed for all inputs
        )

        # **Fully Connected Layers for Feature Embeddings**
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(4096, embedding_dim),  # Embedding layer (128-dim)
            nn.Tanh(),  # Keeps values in range [-1, 1] for better similarity learning
        )

    def _conv_block(self, in_channels, out_channels, kernel_size):
        """
        Defines a CNN block with Conv2D, BatchNorm, ReLU, and MaxPooling.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Reduces spatial size
        )

    def extract_features(self, x):
        """
        Extracts feature embeddings from an input image.
        """
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten before passing to FC layer
        x = self.fc1(x)
        return x

    def forward(self, img1, img2):
        """
        Computes similarity between two images.
        """
        emb1 = self.extract_features(img1)
        emb2 = self.extract_features(img2)
        return emb1, emb2  # Returns raw embeddings (not a classification)
