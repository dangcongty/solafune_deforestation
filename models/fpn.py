import os
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class FPN(nn.Module):
    def __init__(self, in_chan, num_classes):
        super().__init__()

        self.model = smp.create_model(
            arch='fpn',
            encoder_name='timm-efficientnet-b0',
            encoder_weights=None,
            in_channels=in_chan,
            classes=num_classes
        )

    def forward(self, image):
        return self.model(image)

if __name__ == '__main__':
    fpn = FPN(12, 4) #14 145 184
    