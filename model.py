import math
import os

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.modulated_deform_conv import _ModulatedDeformConv
from modules.modulated_deform_conv import ModulatedDeformConvPack

# in: 64  out: 64  without BN
class RB_NoBN(nn.Module):
    def __init__(self, n_feats=64, kernel_size=3, bias=True, bn=False, act=nn.PReLU(), res_scale=1):
        super(RB_NoBN, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size//2, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x) #.mul(self.res_scale)
        res += x
        return res

# in: 64 out: 64  with IN
class RB_IN(nn.Module):
    def __init__(self, in_c=64):
        super(RB_IN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(in_c, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(in_c, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        output = torch.add(output, identity_data)
        return output

# in: x  out:y    one Conv with BN
class BasicBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True,bn=False, act=nn.PReLU()):
        m = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)
        super(BasicBlock, self).__init__(*m)

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
        self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1,1)
        self.dcn_pack = DCNv2Pack(
                num_feat,
                num_feat,
                3,
                stride=1,
                padding=1,
                deformable_groups=deformable_groups)
        self.feat_conv = nn.Conv2d(num_feat, num_feat, 3, 1,1)
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
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        # print(nbr_feat_l.size())
        # print(ref_feat_l.size())
        offset = torch.cat([nbr_feat_l, ref_feat_l], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        feat = self.dcn_pack(nbr_feat_l, offset)
        feat = self.lrelu(self.feat_conv(feat))
        # Cascading
        offset = torch.cat([feat, ref_feat_l], dim=1)
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

    def __init__(self, num_feat=64, num_frame=2):
        super(TSAFusion, self).__init__()
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_feat , num_feat, 3,1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, ref, neb):
        b, c, h, w = ref.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(ref.clone())
        embedding = self.temporal_attn2(neb)
        corr = torch.sum(embedding * embedding_ref, 1)
        corr_prob = corr.unsqueeze(1).expand(b, c, h, w)
        ref = ref * corr_prob
        feat = self.lrelu(self.feat_fusion(ref))

        return feat

class fusion(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(fusion, self).__init__()
        self.PCD = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        self.TSA = TSAFusion(num_feat=num_feat)

    def forward(self, ref, neb):
        start = ref
        feat = self.PCD(neb, ref)
        feat = self.TSA(ref, feat)
        feat = torch.add(feat, start)
        return feat

class cat_conv(nn.Module):
    def __init__(self, num_feat = 64):
        super(cat_conv, self).__init__()
        self.conv = nn.Conv2d(num_feat*2, num_feat, 3, 1, 1)

    def forward(self, ref, neb):
        start = ref
        feat = torch.concat([ref, neb], 1) # n c*2 h w
        feat = self.conv(feat)
        feat = torch.add(feat, start)
        return feat


# in:64 + 64 out；64
class NonLocalAttention(nn.Module):
    def __init__(self, channel=64, reduction=2, res_scale=1):
        super(NonLocalAttention, self).__init__()
        self.res_scale = res_scale
        self.conv_match1 = BasicBlock(channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_match2 = BasicBlock(channel, channel // reduction, 1, bn=False, act=nn.PReLU())
        self.conv_assembly = BasicBlock(channel, channel, 1, bn=False, act=nn.PReLU())

    def forward(self, inputa, inputb):

        x_embed_1 = self.conv_match1(inputa)
        x_embed_2 = self.conv_match2(inputb)
        x_assembly = self.conv_assembly(inputa)

        N, C, H, W = x_embed_1.shape
        x_embed_1 = x_embed_1.permute(0, 2, 3, 1).view((N, H * W, C))
        x_embed_2 = x_embed_2.view(N, C, H * W)
        score = torch.matmul(x_embed_1, x_embed_2)
        score = F.softmax(score, dim=2)
        x_assembly = x_assembly.view(N, -1, H * W).permute(0, 2, 1)
        x_final = torch.matmul(score, x_assembly)
        return x_final.permute(0, 2, 1).view(N, -1, H, W) + self.res_scale * inputa

# in:64 + file  out:64 + new_file
class GFeature(nn.Module):
    def __init__(self, RN_num = 5, channel=64, kernel_size=3):
        super(GFeature, self).__init__()
        m_body = [
            RB_NoBN(n_feats=channel, kernel_size=kernel_size, res_scale=1) for _ in range(RN_num)
        ]
        self.attn = fusion(num_feat=channel)
        self.body = nn.Sequential(*m_body)
    def get_feature(self, feat_ID):
        if not os.path.exists("data/feature/"):
            os.makedirs("data/feature/")
        if os.path.exists("data/feature/"+feat_ID+".npy"):
            np_feat = numpy.load("data/feature/"+feat_ID+".npy")
            tensor_feat = torch.from_numpy(np_feat)
            return True, tensor_feat
        else:
            return False, None
    def save_feature(self, feat_ID, feat):
        if not os.path.exists("data/feature/"):
            os.makedirs("data/feature/")
        np_feat = feat.cpu().detach().numpy()
        numpy.save("data/feature/"+feat_ID+".npy", np_feat)

    def forward(self, x, feat_ID):
        feat_ID = str(feat_ID[0])
        Nofirst, pre_feat = self.get_feature(feat_ID)
        if Nofirst:
            pre_feat = pre_feat.cuda()
            feat = self.attn(x, pre_feat)
            feat = self.body(feat)
            self.save_feature(feat_ID, feat)
            return feat
        else:
            feat = self.body(x)
            self.save_feature(feat_ID, feat)
            return feat

# in：64 out:3
class _NetG(nn.Module):
    def __init__(self, in_c=64, out_c=3, kernel_size=3):
        super(_NetG, self).__init__()

        # self.conv_input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4, bias=False)
        # self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(RB_IN, 16)

        self.conv_mid = nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn_mid = nn.InstanceNorm2d(in_c, affine=True)

        self.upscale4x = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c*4, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=in_c, out_channels=in_c*4, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_output = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_size*3, stride=1, padding=4, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = self.relu(self.conv_input(x))
        residual = x
        out = self.residual(x)
        out = self.bn_mid(self.conv_mid(out))
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.conv_output(out)
        return out

# in:3 + file out:3 + file
class SRModel(nn.Module):
    def __init__(self, in_c=3, mid_c=64, out_c=3, kernel_size=3, pre_RN=5):
        super(SRModel, self).__init__()
        self.head = nn.Conv2d(in_c, mid_c, 9, padding=9//2)
        self.GF = GFeature(RN_num=pre_RN, channel=mid_c, kernel_size=kernel_size)
        self.SR = _NetG(in_c=mid_c, out_c=out_c, kernel_size=kernel_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, feat_ID):
        start = x
        out = self.relu(self.head(x))
        out = self.GF(out, feat_ID)
        out = self.SR(out)
        out = torch.add(out, F.interpolate(start, scale_factor=4, mode='bilinear', align_corners=False))
        return out


