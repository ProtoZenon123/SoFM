import torch
import torch.nn as nn

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
            nn.Linear(4096, 121)
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
    
if __name__ == '__main__':
    '''
        test the output of the model
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
