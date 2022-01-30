import torch
import torch.nn as nn
from model.dcn import (ModulatedDeformConvPack, modulated_deform_conv)
from torch.autograd import Variable


import torch.nn.functional as F


class DCN_l1(nn.Module):
    def __init__(self, num_feat=32, deformable_groups=1, device='cuda'):
        super(DCN_l1, self).__init__()
        self.deformable_groups = deformable_groups
        self.offset_conv1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dcn_pack = DCNv2Pack_l1(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.device = device

    def forward(self, nbr_feat_l, ref_feat_l):

        offset = torch.cat([nbr_feat_l, ref_feat_l], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        offset = self.lrelu(self.offset_conv3(offset))

        feat, offset, mask = self.dcn_pack(nbr_feat_l, offset)
        feat = self.lrelu(feat)

        return feat, offset, mask

class DCN(nn.Module):
    def __init__(self, num_feat=32, deformable_groups=1, device='cuda'):
        super(DCN, self).__init__()
        self.deformable_groups = deformable_groups
        self.offset_conv1 = nn.Conv2d(num_feat * 2 , num_feat, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.dcn_pack = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.dcn_pack_zero = DCNv2Zero(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.device = device


    def forward(self, nbr_feat_l, ref_feat_l, flow, mask):

        flow = F.interpolate(flow, mode='bicubic', scale_factor=2, align_corners=True)*2
        mask = F.interpolate(mask, mode='bicubic', scale_factor=2, align_corners=True)
        warp = self.dcn_pack_zero(nbr_feat_l, flow, mask)
        offset = torch.cat([warp, ref_feat_l], dim=1)
        offset = self.lrelu(self.offset_conv1(offset))
        offset = self.lrelu(self.offset_conv2(offset))
        offset = self.lrelu(self.offset_conv3(offset))
        feat, offset, mask = self.dcn_pack(nbr_feat_l, offset, flow)

        feat = self.lrelu(feat)

        return feat, offset, mask


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat, offsetdown):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        #mean = torch.mean(torch.abs(5 * torch.tanh(torch.cat((o1, o2), dim=1))))
        offset = 8 * torch.tanh(torch.cat((o1, o2), dim=1)) + offsetdown

        mask = torch.sigmoid(mask)

        '''
        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')
        '''
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups), offset, mask

class DCNv2Zero(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, offset, mask):

        '''
        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')
        '''
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class DCNv2Pack_l1(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = 15 * torch.tanh(torch.cat((o1, o2), dim=1))
        #offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        '''
        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')
        '''
        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups), offset, mask


