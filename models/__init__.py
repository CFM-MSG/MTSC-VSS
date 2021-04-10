import torch
import torchvision
import torch.nn.functional as F

from .audio_net import Unet
from .va_net import ResnetDilated
from .criterion import BCELoss, L1Loss, L2Loss
from .vm_net import generate_model


def activate(x, activation):
    if activation == 'sigmoid':
        return torch.sigmoid(x)
    elif activation == 'softmax':
        return F.softmax(x, dim=1)
    elif activation == 'relu':
        return F.relu(x)
    elif activation == 'tanh':
        return F.tanh(x)
    elif activation == 'no':
        return x
    else:
        raise Exception('Unkown activation!')


class ModelBuilder():
    def __init__(self, args):
        self.args = args

    def build_sound(self):
        net_sound = Unet()
        return net_sound

    def build_frame(self, fc_dim=64, pool_type='avgpool'):
        pretrained = True
        original_resnet = torchvision.models.resnet18(pretrained)
        net = ResnetDilated(original_resnet, fc_dim=fc_dim, pool_type=pool_type, sa=self.args.s_a)

        for i in net.parameters():
            i.requires_grad = False
        return net

    def build_pretrained_3Dresnet_50(self):
        model = generate_model(50, n_classes=1039)
        for i in model.parameters():
            i.requires_grad = False
        return model

    def build_criterion(self, arch):
        if arch == 'bce':
            net = BCELoss()
        elif arch == 'l1':
            net = L1Loss()
        elif arch == 'l2':
            net = L2Loss()
        else:
            raise Exception('Architecture undefined!')
        return net

    # def build_criterion(self, arch):
    #     if arch == 'bce':
    #         net = BCELoss()
    #     elif arch == 'l1':
    #         net = L1Loss()
    #     elif arch == 'l2':
    #         net = L2Loss()
    #     else:
    #         raise Exception('Architecture undefined!')
    #     return net
