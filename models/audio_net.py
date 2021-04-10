import torch
import torch.nn as nn
import torch.nn.functional as F


class Unet(nn.Module):
    def __init__(self, ngf=64):
        super(Unet, self).__init__()
        two_stream = TwoStreamBlock(ngf * 8, ngf * 8)
        unet_block = UnetBlock(ngf * 8, ngf * 8, upconv_in_dim=ngf * 8 * 2, submodule=two_stream)
        unet_block = UnetBlock(ngf * 8, ngf * 4, downconv_in_dim=None, submodule=unet_block)
        unet_block = UnetBlock(ngf * 4, ngf * 2, downconv_in_dim=None, submodule=unet_block)
        unet_block = UnetBlock(ngf * 2, ngf, downconv_in_dim=None, submodule=unet_block)
        unet_block = UnetBlock(ngf, 1, downconv_in_dim=1, submodule=unet_block, outermost=True)
        self.bn0 = nn.BatchNorm2d(1)
        self.unet_block = unet_block

    def forward(self, x, feat_motion, feat_appear):
        x = self.bn0(x.unsqueeze(1))
        x, feat_motion, feat_appear = self.unet_block(x, feat_motion, feat_appear)
        return x.squeeze(), feat_motion, feat_appear


class UnetBlock(nn.Module):
    def __init__(self, downconv_out_dim, upconv_out_dim, downconv_in_dim=None, upconv_in_dim=None,
                 submodule=None, outermost=False, innermost=False, noskip=False):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.noskip = noskip
        use_bias = False
        if downconv_in_dim is None:
            downconv_in_dim = upconv_out_dim
        if innermost:
            upconv_in_dim = downconv_out_dim
        elif upconv_in_dim is None:
            upconv_in_dim = downconv_out_dim * 2

        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = nn.BatchNorm2d(downconv_out_dim)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(upconv_out_dim)
        upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        if outermost:
            downconv = nn.Conv2d(downconv_in_dim, downconv_out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(upconv_in_dim, upconv_out_dim, kernel_size=3, padding=1)
            down = [downconv]
            up = [uprelu, upsample, upconv]
        elif innermost:
            downconv = nn.Conv2d(downconv_in_dim, downconv_out_dim, kernel_size=4, stride=2, padding=1,
                                 bias=use_bias)  # nice
            upconv = nn.Conv2d(upconv_in_dim, upconv_out_dim, kernel_size=3, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upsample, upconv, upnorm]
        else:
            downconv = nn.Conv2d(downconv_in_dim, downconv_out_dim, kernel_size=4, stride=2, padding=1, bias=use_bias)
            upconv = nn.Conv2d(upconv_in_dim, upconv_out_dim, kernel_size=3, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upsample, upconv, upnorm]
        self.down_net = nn.Sequential(*down)
        self.submodule = submodule
        self.up_net = nn.Sequential(*up)

    def forward(self, x, feat_motion, feat_appear):
        x_out = self.down_net(x)
        x_out, feat_motion, feat_appear = self.submodule(x_out, feat_motion, feat_appear)
        x_out = self.up_net(x_out)

        if self.outermost or self.noskip:
            return x_out, feat_motion, feat_appear
        else:
            return torch.cat([x, x_out], 1), feat_motion, feat_appear

class TwoStreamBlock(nn.Module):
    def __init__(self, downconv_out_dim, upconv_out_dim, feat_motion_dim=2048, feat_appear_dim=512):
        super(TwoStreamBlock, self).__init__()
        upconv_in_dim = downconv_out_dim
        downconv_in_dim = upconv_out_dim
        downconv = nn.Conv2d(downconv_in_dim, downconv_out_dim, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, False)
        uprelu = nn.ReLU(True)
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        upconv = nn.Conv2d(upconv_in_dim, upconv_out_dim, kernel_size=3, padding=1, bias=False)
        upnorm = nn.BatchNorm2d(upconv_out_dim)
        self.down_motion = nn.Sequential(*[downrelu, downconv])
        self.up_motion = nn.Sequential(*[uprelu, upsample, upconv, upnorm])

        downconv2 = nn.Conv2d(downconv_in_dim, downconv_out_dim, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu2 = nn.LeakyReLU(0.2, False)
        uprelu2 = nn.ReLU(True)
        upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        upconv2 = nn.Conv2d(upconv_in_dim, upconv_out_dim, kernel_size=3, padding=1, bias=False)
        upnorm2 = nn.BatchNorm2d(upconv_out_dim)
        self.down_appear = nn.Sequential(*[downrelu2, downconv2])
        self.up_appear = nn.Sequential(*[uprelu2, upsample2, upconv2, upnorm2])

        self.fc_motion = nn.Linear(feat_motion_dim, downconv_in_dim)
        self.fc_appear = nn.Linear(feat_appear_dim, downconv_in_dim)
        self.sig_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, feat_motion, feat_appear):
        feat_motion = self.fc_motion(feat_motion)
        feat_motion = feat_motion.unsqueeze(-1).unsqueeze(-1)
        feat_appear = feat_appear.permute(0, 2, 3, 4, 1).contiguous()
        feat_appear = self.fc_appear(feat_appear).mean(1)
        feat_appear = feat_appear.permute(0, 3, 1, 2).contiguous()
        map = torch.sum(torch.mul(feat_motion, feat_appear), 1, keepdim=True)
        map = self.sig_conv(map)
        map = self.sigmoid(map)
        feat_appear = self.max_pool(map * feat_appear).squeeze()

        x_motion = self.down_motion(x)
        x_motion = feat_motion + x_motion
        x_motion = self.up_motion(x_motion)

        x_appear = self.down_appear(x)
        x_appear = feat_appear.unsqueeze(-1).unsqueeze(-1) + x_appear
        x_appear = self.up_appear(x_appear)
        return torch.cat([x_appear, x_motion], 1), feat_motion, feat_appear
