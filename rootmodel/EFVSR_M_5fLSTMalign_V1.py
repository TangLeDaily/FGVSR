

import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable
from torch.nn import functional as F

from modules.modulated_deform_conv import _ModulatedDeformConv
from modules.modulated_deform_conv import ModulatedDeformConvPack


# Define some constants
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class BasicBlock(nn.Module):
    def __init__(self, inplanes = 64 * 4, planes = 64 * 4, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, padding=3 // 2)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=3 // 2)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)

class LSTM(nn.Module):
    """
    init: in_c, out_c
    input: tensor(x): b, in_c, h, w ; tensor(pre_x): b, out_c, h, w ; tensor(pre_c): b, out_c, h, w
    out: tensor(y & pre_h): b, out_c, h, w ; tensor(pre_c): b, out_c, h, w
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size=64, hidden_size=64):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, 3, padding=3 // 2)

    def forward(self, input_, prev_hidden, prev_cell):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_hidden is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_hidden = Variable(torch.zeros(state_size))
            prev_cell = Variable(torch.zeros(state_size))

        prev_hidden = torch.tensor(prev_hidden).cuda()
        prev_cell = torch.tensor(prev_cell).cuda()

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = f.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)
        return hidden, cell

class LSTM_enhence(nn.Module):
    def __init__(self):
        super(LSTM_enhence, self).__init__()
        self.LSTM = nn.ModuleDict()
        self.RB = nn.ModuleDict()
        for i in range(3):
            self.LSTM[str(i)] = LSTM()
            self.RB[str(i)] = ResidualBlockNoBN()
    def forward(self, input_feat, input_cell, ref_feat, neb_feat, single):
        hp_feat = []
        hn_feat = []
        c_feat = []
        if single:
            for i in range(3):
                if input_feat is None:
                    h_pre, c = self.LSTM[str(i)](neb_feat[i], None, None)
                else:
                    h_pre, c = self.LSTM[str(i)](neb_feat[i], input_feat[i], input_cell[i])
                h_now, c = self.LSTM[str(i)](ref_feat[i], h_pre, c)
                hp_feat.append(self.RB[str(i)](h_pre + neb_feat[i]))
                hn_feat.append(self.RB[str(i)](h_now + ref_feat[i]))
                c_feat.append(c)
            return hn_feat, c_feat, hn_feat, hp_feat
        else:
            for i in range(3):
                if input_feat is None:
                    h_now, c = self.LSTM[str(i)](ref_feat[i], None, None)
                else:
                    h_now, c = self.LSTM[str(i)](ref_feat[i], input_feat[i], input_cell[i])
                h_pre, c = self.LSTM[str(i)](neb_feat[i], h_now, c)
                hp_feat.append(self.RB[str(i)](h_pre + neb_feat[i]))
                hn_feat.append(self.RB[str(i)](h_now + ref_feat[i]))
                c_feat.append(c)
            return hp_feat, c_feat, hn_feat, hp_feat

class DCNAlignment(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(DCNAlignment, self).__init__()
        self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dcn = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=1,
                deformable_groups=deformable_groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        offset = torch.cat((nbr_feat_l, ref_feat_l), dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        nbr_feat_l = nbr_feat_l.contiguous()
        feat = self.lrelu(self.dcn(nbr_feat_l, offset))
        return feat

class LSTMAlign(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(LSTMAlign, self).__init__()
        self.DCN_align_1 = DCNAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        self.LSTM_1 = LSTM()
        self.DCN_align_2 = DCNAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        self.LSTM_2 = LSTM()
        self.DCN_align_3 = DCNAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        self.LSTM_3 = LSTM()

    def forward(self, x, last_h, last_c):
        # b, t, c, h, w = x.size()
        x_center = x[:, 0, :, :, :]
        neb_feat = []
        neb_feat_1 = []
        neb_feat_2 = []
        aligned_feat = []
        if last_h is None:
            h1 = None
            c1 = None
            h2 = None
            c2 = None
            h3 = None
            c3 = None
        else:
            h1 = last_h[0]
            c1 = last_c[0]
            h2 = last_h[1]
            c2 = last_c[1]
            h3 = last_h[2]
            c3 = last_c[2]
        new_h = []
        new_c = []
        for i in range(4):
            neb_feat.append(x[:, i+1, :, :, :])
            h1, c1 = self.LSTM_1(neb_feat[i], h1, c1)
            neb_feat_1.append(self.DCN_align_1(neb_feat[i], x_center))
            h2, c2 = self.LSTM_2(neb_feat_1[i], h2, c2)
            if i == 0:
                new_h.append(h1)
                new_c.append(c1)
                new_h.append(h2)
                new_c.append(c2)
        h1, c1 = self.LSTM_1(x_center, h1, c1)
        x_center_1 = h1
        h2, c2 = self.LSTM_2(x_center_1, h2, c2)
        x_center_2 = h2
        for i in range(4):
            neb_feat_2.append(self.DCN_align_2(neb_feat_1[i], x_center_1))
            h3, c3 = self.LSTM_3(neb_feat_2[i], h3, c3)
            if i == 0:
                new_h.append(h3)
                new_c.append(c3)
        h3, c3 = self.LSTM_3(x_center_2, h3, c3)
        x_center_3 = h3 + x_center
        aligned_feat.append(x_center_3)
        for i in range(4):
            aligned_feat.append(self.DCN_align_3(neb_feat_2[i], x_center_2))
        aligned_feat = torch.stack(aligned_feat, dim=1)

        return aligned_feat, new_h, new_c


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset_mask(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return _ModulatedDeformConv(x, offset, mask, self.weight, self.bias,
                                    self.stride, self.padding, self.dilation,
                                    self.groups, self.deformable_groups,
                                    self.im2col_step)


# DCNv2Pack = ModulatedDeformConvPack

class PCDAlignment(nn.Module):
    """Alignment module using Pyramid, Cascading and Deformable convolution
    (PCD). It is used in EDVR.

    Ref:
        EDVR: Video Restoration with Enhanced Deformable Convolutional Networks

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        deformable_groups (int): Deformable groups. Defaults: 8.
    """

    def __init__(self, num_feat=64, deformable_groups=8):
        super(PCDAlignment, self).__init__()

        # Pyramid has three levels:
        # L3: level 3, 1/4 spatial size
        # L2: level 2, 1/2 spatial size
        # L1: level 1, original spatial size
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        self.LSTM = nn.ModuleDict()

        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                 1)
            if i == 3:
                self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            else:
                self.offset_conv2[level] = nn.Conv2d(num_feat * 2, num_feat, 3,
                                                     1, 1)
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1,
                                                     1)
            self.dcn_pack[level] = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=1,
                deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1,
                                                  1)

        # Cascading dcn
        self.cas_offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.cas_offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.cas_dcnpack = DCNv2Pack(
            num_feat,
            num_feat,
            3,
            stride=1,
            padding=1,
            deformable_groups=deformable_groups)

        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self,nbr_feat_l, ref_feat_l):
        # Pyramids
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            if i == 3:
                offset = self.lrelu(self.offset_conv2[level](offset))
            else:
                offset = self.lrelu(self.offset_conv2[level](torch.cat(
                    [offset, upsampled_offset], dim=1)))
                offset = self.lrelu(self.offset_conv3[level](offset))

            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](
                    torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)

            if i > 1:  # upsample offset and features
                # x2: when we upsample the offset, we should also enlarge
                # the magnitude.
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)

        # Cascading
        offset = torch.cat([feat, ref_feat_l[0]], dim=1)
        offset = self.lrelu(
            self.cas_offset_conv2(self.lrelu(self.cas_offset_conv1(offset))))
        feat = self.lrelu(self.cas_dcnpack(feat, offset))
        return feat


class TSAFusion(nn.Module):
    """Temporal Spatial Attention (TSA) fusion module.

    Temporal: Calculate the correlation between center frame and
        neighboring frames;
    Spatial: It has 3 pyramid levels, the attention is similar to SFT.
        (SFT: Recovering realistic texture in image super-resolution by deep
            spatial feature transform.)

    Args:
        num_feat (int): Channel number of middle features. Default: 64.
        num_frame (int): Number of frames. Default: 5.
        center_frame_idx (int): The index of center frame. Default: 2.
    """

    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAFusion, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        """
        Args:
            aligned_feat (Tensor): Aligned features with shape (b, t, c, h, w).

        Returns:
            Tensor: Features after TSA with the shape (b, c, h, w).
        """
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, 0, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob
        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class EFVSR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 num_feat=64,
                 num_frame=5,
                 deformable_groups=8,
                 num_extract_block=5,
                 num_reconstruct_block=10):
        super(EFVSR, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # self.LSTM = LSTM(num_feat, num_feat)
        # extrat pyramid features
        self.feature_extraction = make_layer(
            ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.LSTMAlign = LSTMAlign(num_feat=num_feat, deformable_groups=deformable_groups)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame)
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)
        # upsample
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, last_h, last_c):
        b, t, c, h, w = x.size()
        x_center = x[:, 0, :, :, :].contiguous()

        # extract features for each frame
        # orign
        feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))
        feat = self.feature_extraction(feat)
        feat = feat.view(b, t, -1, h, w)
        feat, new_h, new_c = self.LSTMAlign(feat, last_h=last_h, last_c=last_c)
        out = self.fusion(feat)
        out = self.reconstruction(out)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        base = F.interpolate(x_center, scale_factor=4, mode='bilinear', align_corners=False)
        out += base

        return out, new_h, new_c

