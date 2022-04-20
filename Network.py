import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.autograd import Variable
from torch.autograd import Variable
from skimage.segmentation import slic, mark_boundaries
import random

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.DownConv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 32),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=True),
        )
        self.DownSample1 = nn.MaxPool3d((1, 2, 2))

        self.DownConv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=True),
        )
        self.DownSample2 = nn.MaxPool3d((1, 2, 2))

        self.DownConv3 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=True),
        )
        self.DownSample3 = nn.MaxPool3d((1, 2, 2))

        self.botton = nn.Sequential(
            nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 512),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, inputs):
        b, c, d, w, h = inputs.shape

        DownConv1 = self.DownConv1(inputs)
        DownSample1 = self.DownSample1(DownConv1)

        DownConv2 = self.DownConv2(DownSample1)
        DownSample2 = self.DownSample2(DownConv2)

        DownConv3 = self.DownConv3(DownSample2)
        DownSample3 = self.DownSample3(DownConv3)

        botton = self.botton(DownSample3)

        return DownConv1, DownConv2, DownConv3, botton


class mainNetwork(nn.Module):
    def __init__(self,out):
        super(mainNetwork, self).__init__()

        self.globalnet = Encoder()

        self.localnet = Encoder()

        ###self-atten
        self.satten_k = nn.Conv3d(512, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.satten_q = nn.Conv3d(512, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.satten_v = nn.Conv3d(512, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.satten_o = nn.Conv3d(256, 512, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.gn = nn.GroupNorm(4, 512)
        self.satten_c = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        ###cross-atten
        self.catten_k = nn.Conv3d(512, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.catten_q = nn.Conv3d(512, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.catten_v = nn.Conv3d(512, 256, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.catten_o = nn.Conv3d(256, 512, kernel_size=(1, 1, 1), padding=(0, 0, 0))
        self.catten_c = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.UpSample1 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.UpConv1 = nn.Sequential(
            nn.Conv3d(768, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.UpSample2 = nn.ConvTranspose3d(256, 256, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.UpConv2 = nn.Sequential(
            nn.Conv3d(384, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(inplace=True),
        )

        self.UpSample3 = nn.ConvTranspose3d(128, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.UpConv3 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(inplace=True),
        )

        self.out = nn.Conv3d(64, out, kernel_size=(1, 1, 1), padding=(0, 0, 0))

    def forward(self, x):

        b, c, d, w, h = x.shape

        glo_DownConv1, glo_DownConv2, glo_DownConv3, glo_botton = self.globalnet(x)

        loc_DownConv1, loc_DownConv2, loc_DownConv3, loc_botton = self.localnet(x)

        bb, cb, db, wb, hb = loc_botton.shape
        ##self-atten
        self_loc_k = self.satten_k(loc_botton)
        self_loc_k = self_loc_k.reshape(-1, 256)
        self_loc_q = self.satten_q(loc_botton)
        self_loc_q = self_loc_q.reshape(256, -1)
        self_loc_v = self.satten_v(loc_botton)
        self_loc_v = self_loc_v.reshape(-1, 256)

        self_con = torch.mm(self_loc_k,self_loc_q)
        self_con = torch.softmax(self_con,dim=-1)
        self_atten = torch.mm(self_con,self_loc_v)
        self_atten = self_atten.view(1,256,db, wb, hb)
        self_atten = self.satten_o(self_atten)
        self_atten = self_atten+loc_botton
        self_atten = self.gn(self_atten)
        self_atten1 = self.satten_c(self_atten)
        self_atten1 = self_atten+self_atten1
        self_atten1 = self.gn(self_atten1)

        ##cross-atten
        cross_glo_q = self.catten_q(self_atten1)
        cross_glo_q = cross_glo_q.reshape(256, -1)
        # cat_glo_loc = torch.cat((self_atten1,glo_botton),1)
        cross_glo_k = self.catten_k(glo_botton)
        cross_glo_k = cross_glo_k.reshape(-1, 256)
        cross_glo_v = self.catten_v(self_atten1)
        cross_glo_v = cross_glo_v.reshape(-1, 256)

        cross_con = torch.mm(cross_glo_k,cross_glo_q)
        cross_con = torch.softmax(cross_con, dim=-1)
        cross_atten = torch.mm(cross_con,cross_glo_v)
        cross_atten = cross_atten.view(1, 256, db, wb, hb)
        cross_atten = self.catten_o(cross_atten)
        cross_atten = cross_atten+self_atten1
        cross_atten = self.gn(cross_atten)
        cross_atten1 = self.catten_c(cross_atten)
        cross_atten1 = cross_atten1+cross_atten
        cross_atten1 = self.gn(cross_atten1)

        cross_atten1 = torch.cat((self_atten1,cross_atten1),1)
        #######################

        Upsample1 = self.UpSample1(cross_atten1)

        a = Upsample1
        c = (w // 4 - a.size()[3])
        c1 = (h // 4 - a.size()[4])
        cc = (d - a.size()[2])
        bypass = F.pad(a, (0, c1, 0, c, 0, cc))
        cat1 = torch.cat((loc_DownConv3, bypass), 1)

        UpConv1 = self.UpConv1(cat1)

        Upsample2 = self.UpSample2(UpConv1)

        a = Upsample2
        c = (w // 2 - a.size()[3])
        c1 = (h // 2 - a.size()[4])
        cc = (d - a.size()[2])
        bypass = F.pad(a, (0, c1, 0, c, 0, cc))
        cat2 = torch.cat((loc_DownConv2, bypass), 1)

        UpConv2 = self.UpConv2(cat2)

        Upsample3 = self.UpSample3(UpConv2)

        a = Upsample3
        c = (w - a.size()[3])
        c1 = (h - a.size()[4])
        cc = (d - a.size()[2])
        bypass = F.pad(a, (0, c1, 0, c, 0, cc))
        cat3 = torch.cat((loc_DownConv1, bypass), 1)

        UpConv3 = self.UpConv3(cat3)

        out = self.out(UpConv3)
        segresult = F.softmax(out, dim=1)

        return segresult, glo_botton, loc_botton