import torch
import torch.nn as nn
from torchvision import models

class sia_Resnet50(nn.Module):
    def __init__(self, channels_img, embedding_dim=128):
        super(sia_Resnet50, self).__init__()

        # Load the pretrained ResNet50 model
        self.resnet50 = models.resnet50(weights='IMAGENET1K_V1')  # Using the 'weights' argument instead of 'pretrained'

        # Remove the final classification layer (fc) and the avgpool layer from ResNet50
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])  # Removes the fully connected layer and the avgpool layer

        # Get the number of features from the output of the last Conv layer
        # After the last conv layer, the output shape is [batch_size, 2048, 1, 1]
        self.num_ftrs = 32768  # This corresponds to the output feature map size from ResNet50

        # Add custom layers for embedding extraction
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_ftrs, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, embedding_dim),  # Adjusting to the desired embedding dimension
            nn.ReLU(inplace=True)
        )

    def forward(self, input_01, input_02):
        # Extract embeddings for both inputs using ResNet50
        emb1 = self.extract_features(input_01)
        emb2 = self.extract_features(input_02)

        # Return embeddings for contrastive loss
        return emb1, emb2

    def extract_features(self, x):
        # Forward pass through ResNet50 feature extractor
        x = self.resnet50(x)  # Get features after the final conv layer (without FC layer)
        # Flatten the output to pass it through the fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the output of the conv layers (2048,)
        # Pass through the fully connected layers to get embeddings
        x = self.fc1(x)
        return x




class siameseCNN_Contrast_Inter(nn.Module):
    def __init__(self, channels_img, num_features):
        super(siameseCNN_Contrast_Inter, self).__init__()
        self.layer = nn.Sequential(
            #3*121*121
            self._block(channels_img, num_features, 10),#64*56*56
            self._block(num_features, num_features*2, 7),#128*25*25
            self._block(num_features*2, num_features*2, 4),#128*11*11
            nn.Conv2d(num_features*2, num_features*4, 4),#256*8*8
            nn.BatchNorm2d(num_features*4),
            nn.ReLU(inplace=True)
        )   
        self.fc1 = nn.Sequential(
            nn.Linear(256*8*8, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128)
            #tth11
            # nn.Linear(4096, 256)
        )
        
    def _block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2, stride=2)           
        )
    
    def first_forward(self, input):
        output = self.layer(input)
        output = output.view(-1, 256*8*8)
        output = self.fc1(output)
        return output
        
    def forward(self, input_01, input_02):
        output_01 = self.first_forward(input_01)
        output_02 = self.first_forward(input_02)
        return output_01, output_02


class siameseCNN_BCE(nn.Module):
    def __init__(self, channels_img, num_features):
        super(siameseCNN_BCE, self).__init__()
        self.layer = nn.Sequential(
            #3*121*121
            self._block(channels_img, num_features, 10),#64*56*56
            self._block(num_features, num_features*2, 7),#128*25*25
            self._block(num_features*2, num_features*2, 4),#128*11*11
            nn.Conv2d(num_features*2, num_features*4, 4),#256*8*8
            nn.BatchNorm2d(num_features*4),
            nn.ReLU()
        )   
        self.fc1 = nn.Sequential(
            nn.Linear(256*8*8, 4096),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
        
    def _block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)           
        )
    
    def first_forward(self, input):
        output = self.layer(input)
        output = output.view(-1, 256*8*8)
        output = self.fc1(output)
        return output
        
    def forward(self, input_01, input_02):
        output_01 = self.first_forward(input_01)
        output_02 = self.first_forward(input_02)
        output = torch.abs(output_01 - output_02)
        output = self.fc2(output)
        return output
     
class siameseCNN_Contrast_Origin(nn.Module):  
    def __init__(self, channels_img, num_features):
        super(siameseCNN_Contrast_Origin, self).__init__()
        self.layer = nn.Sequential(
            #3*100*100
            self._block(channels_img, 4, 3),
            self._block(4, 8, 3),
            self._block(8, 8, 3),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
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



### GPTT1
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
            self._conv_block(num_features, num_features * 2, kernel_size=7),  # 128 filters
            self._conv_block(num_features * 2, num_features * 2, kernel_size=4),  # 128 filters
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4),  # 256 filters
            nn.BatchNorm2d(num_features * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))  # Ensures output size is fixed for all inputs
        )

        # **Fully Connected Layers for Feature Embeddings**
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 8 * 8, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(4096, embedding_dim),  # Embedding layer (128-dim)
            nn.Tanh()  # Keeps values in range [-1, 1] for better similarity learning
        )

    def _conv_block(self, in_channels, out_channels, kernel_size):
        """
        Defines a CNN block with Conv2D, BatchNorm, ReLU, and MaxPooling.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces spatial size
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

##GPTT2

class siameseCNN_Improved(nn.Module):
    def __init__(self, channels_img, num_features):
        super(siameseCNN_Improved, self).__init__()
        
        # Convolutional Layers (increasing depth and using GAP)
        self.layer = nn.Sequential(
            self._block(channels_img, num_features, 7),     # 64x56x56
            self._block(num_features, num_features * 2, 5),  # 128x25x25
            self._block(num_features * 2, num_features * 2, 3), # 128x11x11
            self._block(num_features * 2, num_features * 4, 3),  # 256x5x5
        )
        
        # Fully Connected Layer (flattening the output)
        self.fc1 = nn.Sequential(
            nn.Linear(num_features * 4 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 128)
        )

    def _block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
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

   
if __name__ == '__main__':
    '''
        check the output of the model
    '''
    img1 = torch.rand(6, 3, 121, 121)
    img2 = torch.rand(6, 3, 121, 121)
    net = siameseCNN_Contrast_Inter(3, 64)
    output1, output2 = net(img1, img2)
    print('siameseCNN_Contrast_Inter:')
    print(output1.shape)
    print(output2.shape)
    
    img1 = torch.rand(6, 3, 100, 100)
    img2 = torch.rand(6, 3, 100, 100)
    net = siameseCNN_Contrast_Origin(3, 64)
    output1, output2 = net(img1, img2)
    print('siameseCNN_Contrast_Origin:')
    print(output1.shape)
    print(output2.shape)
    
    img1 = torch.rand(6, 3, 121, 121)
    img2 = torch.rand(6, 3, 121, 121)
    net = siameseCNN_BCE(3, 64)
    output1 = net(img1, img2)
    print('siameseCNN_BCE')
    print(output1.shape)

