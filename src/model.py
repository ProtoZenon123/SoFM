import torch
import torch.nn as nn

class siameseNN(nn.Module):
    def __init__(self, channels_img, num_features):
        super(siameseNN, self).__init__()
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
        
    
if __name__ == '__main__':
    '''
        test the output of the model
    '''
    img1 = torch.rand(6, 3, 121, 121)
    img2 = torch.rand(6, 3, 121, 121)
    nn = siameseNN(3, 64)
    output1 = nn(img1, img2)
    print(output1)


