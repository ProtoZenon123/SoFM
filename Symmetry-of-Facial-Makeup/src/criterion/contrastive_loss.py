import os
from dotenv import load_dotenv
import torch.nn as nn
import torch
import torch.nn.functional as func

load_dotenv()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.float()
        euclidean_distance = func.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (label) * torch.pow(euclidean_distance, 2)
            + (1 - label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

def calculate_contrastive_loss(output1, output2, label):
    """
    Calculate the Contrastive Loss
    """
    margin = float(os.getenv("MARGIN"))
    criterion = ContrastiveLoss(margin=margin)
    return criterion(output1, output2, label)
