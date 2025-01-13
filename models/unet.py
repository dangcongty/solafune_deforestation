import os
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

sys.path.append(os.getcwd() + '/tools')
from tools.losses import *


class Unet(nn.Module):
    def __init__(self, in_chan, num_classes):
        super().__init__()

        self.model = smp.create_model(
            arch='UnetPlusPlus',
            encoder_name='tu-tf_efficientnetv2_s',
            encoder_weights='imagenet',
            in_channels=in_chan,
            classes=num_classes
        )

    def forward(self, image):
        return self.model(image)

if __name__ == '__main__':
    unet = Unet(12, 4)
    