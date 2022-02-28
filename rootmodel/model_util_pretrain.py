from torch import nn
import torch
from dcn_v2 import dcn_v2_conv
from dcn_v2 import DCNv2
class DCNv2Pack(DCNv2):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """
    def __init__(self,
                 in_channels=64,
                 out_channels=64,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 deformable_groups=1
                 ):
        super(DCNv2Pack, self).__init__(in_channels, out_channels, kernel_size, stride, padding,deformable_groups)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        channels_ = self.deformable_groups * 3 * self.kernel_size * self.kernel_size
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            channels_,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        return dcn_v2_conv(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.deformable_groups,
        )

