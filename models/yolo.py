
import os
import sys

sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import yaml
from torchinfo import summary
from ultralytics.utils.ops import make_divisible

from models.modules import *


class YOLOSeg(nn.Module):
    def __init__(self, config_file = 'models/yolo_config.yaml', scale = 'n'):
        super().__init__()
        with open(config_file.replace('_n', ''), 'r') as f:
            model_config = yaml.load(f, Loader=yaml.SafeLoader)

        depth, width, max_channels = model_config['scales'][scale]
        ch = [5]
        layers = []
        for idx, cfg in enumerate(model_config['backbone'] + model_config['head']):
            f, r, m, args = cfg
            r = max(round(r * depth), 1) if r > 1 else r  # depth gain
            _m = getattr(torch.nn, m[3:]) if "nn." in m else globals()[m]
            if _m in [Conv, C3k2, SPPF, C2PSA]:
                c2 = int(args[0]*width)
                c1 = ch[f]
                args = [c1, c2, *args[1:]]
                if m == 'C3k2':
                    args.insert(2, r)  # number of repeats
                    r = 1
                    args[3] = False if scale in "mlx" else args[3]
            elif _m is Concat:
                c2 = sum(ch[1:][x] for x in f)
            elif m == 'nn.Upsample':
                args[0] = None
            elif _m is Segment:
                args.append([ch[1:][x] for x in f])
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
                _m.legacy = True

            layer = nn.Sequential(*(_m(*args) for _ in range(r))) if r > 1 else _m(*args)  # module
            t = str(m)[8:-2].replace("__main__.", "")  # module type
            layer.np = sum(x.numel() for x in layer.parameters())  # number params
            layer.i, layer.f, layer.type = idx, f, t  # attach index, 'from' index, type

            ch.append(c2)
            layers.append(layer)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        '''
        input: 
            => image: B x 3 x H x W | B x 3 x 1024 x 1024
        outputs:
            => mask: B x 1 x (H/4 * W/4 + H/8 * W/8 + H/16 * W/16 + H/32 * W/32)
            B x 1 x 87040
        '''
        y, dt, embeddings = [], [], []  # outputs
        for idx, m in enumerate(self.model):
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x) 

        return x[0], x[1], x[-1]

if __name__ == '__main__':
    model = YOLOSeg()
    x = torch.rand((1, 12, 1024, 1024))
    summary(model.cuda(), (2, 12, 1024, 1024))
    # y = model(x)