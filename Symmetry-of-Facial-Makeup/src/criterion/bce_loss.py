import torch.nn as nn

class BCELoss(nn.Module):
    """
    Binary Cross Entropy Loss
    """

    def __init__(self):
        super(BCELoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, output, label):
        return self.criterion(output, label)
    
def calculate_bce_loss(output, label):
    """
    Calculate the Binary Cross Entropy Loss
    """
    criterion = BCELoss()
    return criterion(output, label)