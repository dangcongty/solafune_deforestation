import os
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, in_chan, num_classes):
        super().__init__()

        model = smp.create_model(
            arch='Unet',
            encoder_name='tu-efficientnetv2_rw_t',
            encoder_weights=None,
            in_channels=in_chan,
            classes=num_classes
        )
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.head = model.segmentation_head

        # self.reduce_channel = nn.Conv2d(16, 1, 1, 1)

    def forward(self, image):
        x_enc = self.encoder(image)
        feature = self.decoder(*x_enc)
        
        return self.head(feature), feature

if __name__ == '__main__':
    unet = Unet(12, 5)
    x = torch.rand((1, 12, 1024, 1024))
    y = unet(x)
    print(y.shape)