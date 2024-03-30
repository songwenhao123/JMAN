import torch.nn as nn
import torch.optim as optim
from .model_lanet import LANet
from ..FusionNet.fusionnet_main import SetCriterion

from UDL.pansharpening.models import PanSharpeningModel


class build_lanet(PanSharpeningModel, name='LANet'):
    def __call__(self, args):
        scheduler = None
        if any(["wv" in v for v in args.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4

        loss = nn.MSELoss(size_average=True).cuda()  # Define the Loss function
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        criterion = SetCriterion(losses, weight_dict)
        model = LANet(args, spectral_num, criterion).cuda()
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=0)  # optimizer 1: Adam

        return model, criterion, optimizer, scheduler
