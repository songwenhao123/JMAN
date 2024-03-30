import torch
import torch.nn as nn
from ..FusionNet.model_fusionnet import FusionNet
from .act import create_act_model


class LANet(FusionNet):
    def __init__(self, config, *args, **kwargs):
        super(LANet, self).__init__(*args, **kwargs)
        self.act_backbone = create_act_model(config, is_train=True)

    def forward(self, x, y=None):  # x= lms; y = pan
        if isinstance(x, list) and len(x) == 2 and y == None:
            x, y = x[0], x[1]
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        res = self.act_backbone(input)  # ResNet's backbone! Bsx32x64x64
        sr = x + res  # output:= lms + hp_sr
        return sr  # lms + outs