# from model import common
# from model import common
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


def make_model(args, parent=False):
    return RCAN(args)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class SPM(nn.Module):
    def __init__(self, input_size=64, hidden_size=64):
        super(SPM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Add_Gates = nn.Conv2d(input_size + hidden_size, 2 * hidden_size, 3, padding=3 // 2)
        self.Dot_Gates = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.Out_Gates = nn.Conv2d(input_size + hidden_size, hidden_size, 3, 1, 1)
        self.out_conv = nn.Conv2d(hidden_size, input_size, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, input_, prev_hidden, prev_cell):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        # generate empty prev_state, if None is provided
        if prev_hidden is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = input_
            prev_cell = input_

        # print(prev_hidden.size())

        prev_hidden = prev_hidden.cuda().detach()
        prev_cell = prev_cell.cuda().detach()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        Add_gates = self.Add_Gates(stacked_inputs)
        remember_gate = torch.sigmoid(self.Dot_Gates(stacked_inputs))
        out = self.lrelu(self.Out_Gates(stacked_inputs))

        # chunk across channel dimension
        in_gate, cell_gate = Add_gates.chunk(2, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = prev_cell + (in_gate * cell_gate)
        # compute current cell and hidden state
        hidden = out * torch.tanh(cell)
        cell = remember_gate * cell
        # cell = cell + self.lrelu(self.enhence(prev_cell))
        out_hidden = self.lrelu(self.out_conv(hidden))
        return out_hidden, hidden, cell

class SPM_CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SPM_CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.SPM = SPM()
        self.fusionLS = nn.Conv2d(2*channel, channel, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x, last_h, last_c):
        y = self.avg_pool(x)
        x_LS, next_h, next_c = self.SPM(x, last_h, last_c)
        x = self.lrelu(self.fusionLS(torch.cat((x, x_LS), dim=1)))
        y = self.conv_du(y)
        return x * y, next_h, next_c


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SPM_RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(SPM_RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        self.CA = SPM_CALayer(n_feat, reduction)
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x, last_h, last_c):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res, next_h, next_c = self.CA(res, last_h, last_c)
        res += x
        return res, next_h, next_c

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, conv=default_conv):
        super(RCAN, self).__init__()

        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = 16
        scale = 4
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(255, rgb_mean, rgb_std)

        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]

        self.SPM_RCAB = SPM_RCAB(conv, n_feats, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1)
        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size)]

        self.add_mean = MeanShift(255, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, last_h, last_c):
        x = self.sub_mean(x)
        x = self.head(x)

        if last_h is None:
            res, next_h, next_c = self.SPM_RCAB(x, None, None)
        else:
            res, next_h, next_c = self.SPM_RCAB(x, last_h, last_c)
        res = self.body(res)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x, next_h, next_c

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
