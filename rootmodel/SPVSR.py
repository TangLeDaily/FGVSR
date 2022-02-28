
import os

import numpy
import torch.nn.functional as F

from modules.modulated_deform_conv import _ModulatedDeformConv
from rootmodel.common import *


# DCNv2Pack = ModulatedDeformConvPack
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


class PCDAlignment(nn.Module):
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

    def forward(self, nbr_feat_l, ref_feat_l):
        # Pyramids
        upsampled_offset, upsampled_feat = None, None
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
    def __init__(self, num_feat=64, num_frame=2):
        super(TSAFusion, self).__init__()
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
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, 0, :, :, :].clone())
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


class fusion(nn.Module):
    def __init__(self, num_feat=64, num_frame=2, deformable_groups=8):
        super(fusion, self).__init__()
        self.conv_l2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_l3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.conv_l3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # pcd and tsa module
        self.pcd_align = PCDAlignment(num_feat=num_feat, deformable_groups=deformable_groups)
        self.fusion = TSAFusion(num_feat=num_feat, num_frame=num_frame)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, ref, pre):
        b, c, h, w = ref.size()
        t = 2
        # L1
        x = torch.stack((ref, pre), dim=1)
        feat_l1 = x.view(-1, c, h, w)
        # L2
        feat_l2 = self.lrelu(self.conv_l2_1(feat_l1))
        feat_l2 = self.lrelu(self.conv_l2_2(feat_l2))
        # L3
        feat_l3 = self.lrelu(self.conv_l3_1(feat_l2))
        feat_l3 = self.lrelu(self.conv_l3_2(feat_l3))

        feat_l1 = feat_l1.view(b, t, -1, h, w)
        feat_l2 = feat_l2.view(b, t, -1, h // 2, w // 2)
        feat_l3 = feat_l3.view(b, t, -1, h // 4, w // 4)

        # PCD alignment
        ref_feat_l = [  # reference feature list
            feat_l1[:, 0, :, :, :].clone(),
            feat_l2[:, 0, :, :, :].clone(),
            feat_l3[:, 0, :, :, :].clone()
        ]
        aligned_feat = []
        nbr_feat_l = [  # neighboring feature list
            feat_l1[:, 1, :, :, :].clone(),
            feat_l2[:, 1, :, :, :].clone(),
            feat_l3[:, 1, :, :, :].clone()
        ]
        aligned_feat.append(self.pcd_align(ref_feat_l, ref_feat_l))
        aligned_feat.append(self.pcd_align(nbr_feat_l, ref_feat_l))

        aligned_feat = torch.stack(aligned_feat, dim=1)  # (b, t, c, h, w)

        feat = self.fusion(aligned_feat)
        return feat


# in:64 + file  out:64 + new_file
class GFeature(nn.Module):
    def __init__(self, RN_num=5, channel=64, kernel_size=3):
        super(GFeature, self).__init__()
        m_body = [
            RB_NoBN(n_feats=channel, kernel_size=kernel_size, res_scale=1) for _ in range(RN_num)
        ]
        self.body = nn.Sequential(*m_body)
        self.attn = fusion(num_feat=channel)

    def get_feature(self, feat_ID):
        if not os.path.exists("../data/feature/"):
            os.makedirs("../data/feature/")
        if os.path.exists("../data/feature/" + feat_ID + ".npy"):
            np_feat = numpy.load("../data/feature/" + feat_ID + ".npy")
            tensor_feat = torch.from_numpy(np_feat)
            return True, tensor_feat
        else:
            return False, None

    def save_feature(self, feat_ID, feat):
        if not os.path.exists("../data/feature/"):
            os.makedirs("../data/feature/")
        np_feat = feat.cpu().detach().numpy()
        numpy.save("../data/feature/" + feat_ID + ".npy", np_feat)

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
class SRModle(nn.Module):
    def __init__(self, in_c=64, out_c=3, kernel_size=3, num_reconstruct_block=10):
        super(SRModle, self).__init__()
        # reconstruction2
        m_body = [
            RB_NoBN(n_feats=in_c, kernel_size=kernel_size, res_scale=1) for _ in range(num_reconstruct_block)
        ]
        self.reconstruction = nn.Sequential(*m_body)
        # upsample
        self.upconv1 = nn.Conv2d(in_c, in_c * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(in_c, in_c * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(in_c, in_c, 3, 1, 1)
        self.conv_last = nn.Conv2d(in_c, out_c, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.reconstruction(x)
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out


# in:3 + file out:3 + file
class SPVSR(nn.Module):
    def __init__(self, in_c=3, mid_c=64, out_c=3, kernel_size=3, pre_RN=5):
        super(SPVSR, self).__init__()
        self.head = nn.Conv2d(in_c, mid_c, 9, padding=9 // 2)
        self.GF = GFeature(RN_num=pre_RN, channel=mid_c, kernel_size=kernel_size)
        self.SR = SRModle(in_c=mid_c, out_c=out_c, kernel_size=kernel_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, feat_ID):
        start = x
        out = self.relu(self.head(x))
        out = self.GF(out, feat_ID)
        out = self.SR(out)
        out += F.interpolate(start, scale_factor=4, mode='bilinear', align_corners=False)
        return out
