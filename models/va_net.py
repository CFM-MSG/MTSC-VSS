import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, fc_dim=64, pool_type='maxpool',
                 dilate_scale=16, conv_size=3, sa=False):
        super(ResnetDilated, self).__init__()
        from functools import partial

        self.pool_type = pool_type
        self.sa = sa

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        self.features = nn.Sequential(
            *list(orig_resnet.children())[:-2])

        self.fc = nn.Conv2d(
            512, fc_dim, kernel_size=conv_size, padding=conv_size // 2)

        self.soft = nn.Softmax(dim=2)

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward_multiframe(self, x, pool=True):
        (B, C, T, H, W) = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B * T, C, H, W)

        x = self.features(x)
        # x = self.fc(x)
        (_, C, H, W) = x.size()

        if not self.sa:
            x = x.view(B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
            if not pool:
                return x

            if self.pool_type == 'avgpool':
                x = F.adaptive_avg_pool3d(x, 1)
            elif self.pool_type == 'maxpool':
                x = F.adaptive_max_pool3d(x, 1)
            x = x.view(B, C)
        else:
            x = x.view(B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)
        return x

