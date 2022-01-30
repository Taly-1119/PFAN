import torch
import time
import torch.nn as nn
import model.blocks as blocks
from model.CGDA import AlignFeature





class PFAN(nn.Module):

    def __init__(self, in_channels=3, n_sequence=3, out_channels=3, n_resblock=3, n_feat=32,
                 device='cuda'):
        super(PFAN, self).__init__()
        self.n_sequence = n_sequence
        self.device = device
        self.CGDA = AlignFeature(2, 2, 32, self.device)

        self.encoder = blocks.PFE(num_res=n_resblock, in_channel=in_channels, base_channel=n_feat)
        self.decoder = blocks.PR(num_res=n_resblock, out_channel=out_channels, base_channel=n_feat)

        self.TSA0 = blocks.TSAFusion(num_feat=n_feat, num_frame=3, center_frame_idx=1)
        self.TSA1 = blocks.TSAFusion(num_feat=n_feat*2, num_frame=3, center_frame_idx=1)
        self.TSA2 = blocks.TSAFusion(num_feat=n_feat*4, num_frame=3, center_frame_idx=1)

        assert n_sequence == 3, "Only support args.n_sequence=3; but get args.n_sequence={}".format(n_sequence)

    def forward(self, x):
        feat_list = []
        for i in range(self.n_sequence):
            feat = self.encoder(x[:, i, :, :, :])
            feat_list.append(feat)
        feat_list[0] = self.CGDA(feat_list[0], feat_list[1])
        feat_list[2] = self.CGDA(feat_list[2], feat_list[1])

        l_1 = torch.stack([feat_list[0][0], feat_list[1][0], feat_list[2][0]], dim=1)
        l_1 = self.TSA0(l_1)
        l_2 = torch.stack([feat_list[0][1], feat_list[1][1], feat_list[2][1]], dim=1)
        l_2 = self.TSA1(l_2)
        l_3 = torch.stack([feat_list[0][2], feat_list[1][2], feat_list[2][2]], dim=1)
        l_3 = self.TSA2(l_3)
        output = self.decoder(l_1, l_2, l_3)
        mid_loss = None

        return output, mid_loss





