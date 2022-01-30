import torch
import torch.nn as nn

from torch.autograd import Variable

from model.deconv import DCN, DCN_l1

class AlignFeature(nn.Module):
    def __init__(self, ksize=2, stride=2, feat_num=32, device='cuda'):
        super(AlignFeature, self).__init__()
        self.device = device
        self.ksize = ksize
        self.stride = stride
        self.feat_num = feat_num
        self.DCN1 = DCN_l1(self.feat_num*4)
        self.DCN2 = DCN(self.feat_num*2)
        self.DCN3 = DCN(self.feat_num)



    def forward(self, query ,key):

        l_2, offset_2, mask_2 = self.DCN1(query[2].clone(), key[2].clone())

        l_1, offset_1, mask_1 = self.DCN2(query[1].clone(), key[1].clone(), offset_2.clone(), mask_2.clone())

        l_0, offset_0, mask_0 = self.DCN3(query[0].clone(), key[0].clone(), offset_1.clone(), mask_1.clone())





        return [l_0, l_1, l_2], offset_0, offset_1, offset_2

