import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch.nn.modules.activation import ReLU
#from grid_sample3d import grid_sample_3d
#from utils import SoftPlusPlus, Sine, sine_init, first_layer_sine_init
from torch.autograd import Variable

#from RemezNet.remez_net import rational_net
#from RemezNet.utils import *

# 1D conv usage:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution over only all features of one point sample


# 3D Single View Reconsturction (for 256**3 input voxelization) --------------------------------------
# ----------------------------------------------------------------------------------------------------

VERSION = 'new'

class NDF(nn.Module):


    def __init__(self, hidden_dim=256):
        super(NDF, self).__init__()

        if VERSION == 'new':
            self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
            self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 128
            self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
            self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')  # out: 64
            self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
            self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')  # out: 32
            self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
            self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16
            self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
            self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
            self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
        
        elif VERSION == 'old':
            self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')  # out: 256 ->m.p. 128
            self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')  # out: 128
            self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')  # out: 128 ->m.p. 64
            self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')  # out: 64
            self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')  # out: 64 -> mp 32
            self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')  # out: 32
            self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 32 -> mp 16
            self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16
            self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16 -> mp 8
            self.conv_4 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8
            self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 8
            

        feature_size = (1 +  16 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)

        # classification head
        self.fc_0_cls = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1_cls = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2_cls = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out_cls = nn.Conv1d(hidden_dim, 1, 1)

        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)

        # add remez_net
        #self.m = 7
        #self.n = 7
        #self.max_mn = self.m if self.m >= self.n else self.n
        #self.rat = rational_net(self.max_mn, self.max_mn, feature_size=feature_size)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

    def encoder(self,x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        
        # feature extraction
        if VERSION == 'new':
            feature_0 = F.grid_sample(f_0, p, padding_mode='border', align_corners=True)
            feature_1 = F.grid_sample(f_1, p, padding_mode='border', align_corners=True)
            feature_2 = F.grid_sample(f_2, p, padding_mode='border', align_corners=True)
            feature_3 = F.grid_sample(f_3, p, padding_mode='border', align_corners=True)
            feature_4 = F.grid_sample(f_4, p, padding_mode='border', align_corners=True)
            feature_5 = F.grid_sample(f_5, p, padding_mode='border', align_corners=True)
            feature_6 = F.grid_sample(f_6, p, padding_mode='border', align_corners=True)
        
        
        elif VERSION == 'old':
            # feature extraction
            feature_0 = F.grid_sample(f_0, p, padding_mode='border')
            feature_1 = F.grid_sample(f_1, p, padding_mode='border')
            feature_2 = F.grid_sample(f_2, p, padding_mode='border')
            feature_3 = F.grid_sample(f_3, p, padding_mode='border')
            feature_4 = F.grid_sample(f_4, p, padding_mode='border')
            feature_5 = F.grid_sample(f_5, p, padding_mode='border')
            feature_6 = F.grid_sample(f_6, p, padding_mode='border')
        '''
        # feature extraction
        feature_0 = grid_sample_3d(f_0, p)
        feature_1 = grid_sample_3d(f_1, p)
        feature_2 = grid_sample_3d(f_2, p)
        feature_3 = grid_sample_3d(f_3, p)
        feature_4 = grid_sample_3d(f_4, p)
        feature_5 = grid_sample_3d(f_5, p)
        feature_6 = grid_sample_3d(f_6, p)
        '''

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        p_r = None
        
        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        #net = self.fc_out(net)
        out = net.squeeze(1)

        # classification head
        net_cls = self.actvn(self.fc_0_cls(features))
        net_cls = self.actvn(self.fc_1_cls(net_cls))
        net_cls = self.actvn(self.fc_2_cls(net_cls))

        # classification task with no actvn
        out_cls = self.fc_out_cls(net_cls).squeeze(1)  # (B, 1, samples_num) -> (B, samples_num)

        # return occupancy probabilities for the sampled points
        #print(out_cls)
        #p_r = dist.Bernoulli(logits=out_cls)
        

        '''
        #print(features.get_device())
        shape = features.shape
        support = poly_recur(features.reshape((shape[0]*shape[2], shape[1])), orders=self.max_mn).reshape((-1, shape[0]*shape[2]))
        #label = torch.FloatTensor(ynodes).cuda()
        out = self.rat(support).reshape((shape[0], shape[2]))
        '''

        return  out, out_cls

    def forward(self, p, x):
        out, p_r = self.decoder(p, *self.encoder(x))
        return out, p_r

# for input scale 128
class NDF_128(nn.Module):


    def __init__(self, hidden_dim=256, generate_mode=False):
        super(NDF_128, self).__init__()

        self.generate_mode = generate_mode

        if VERSION == 'new':
            self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
            self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 128
            self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
            self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='replicate')  # out: 64
            self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
            self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')  # out: 32
            self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
            self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16
            self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
            
        elif VERSION == 'old':
            self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode='border')  # out: 256 ->m.p. 128
            self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode='border')  # out: 128
            self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode='border')  # out: 128 ->m.p. 64
            self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode='border')  # out: 64
            self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode='border')  # out: 64 -> mp 32
            self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='border')  # out: 32
            self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 32 -> mp 16
            self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16
            self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode='border')  # out: 16 -> mp 8


        feature_size = (1 +  16 + 32 + 64 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)

        # classification head
        self.fc_0_cls = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1_cls = nn.Conv1d(hidden_dim *2, hidden_dim, 1)
        self.fc_2_cls = nn.Conv1d(hidden_dim , hidden_dim, 1)
        self.fc_out_cls = nn.Conv1d(hidden_dim, 1, 1)

        self.actvn = nn.Tanh()

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)


        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = torch.Tensor(displacments).cuda()

        # siren init
        #self.apply(sine_init)
        #self.fc_0.apply(first_layer_sine_init)
        #self.fc_0_cls.apply(first_layer_sine_init)
        #self.conv_in.apply(first_layer_sine_init)

    def encoder(self,x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net

        return f_0, f_1, f_2, f_3, f_4, f_5

    def decoder(self, p, f_0, f_1, f_2, f_3, f_4, f_5):

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        if self.generate_mode:
            # feature extraction
            if VERSION == 'new':
                feature_0 = F.grid_sample(f_0, p, padding_mode='border', align_corners=True)
                feature_1 = F.grid_sample(f_1, p, padding_mode='border', align_corners=True)
                feature_2 = F.grid_sample(f_2, p, padding_mode='border', align_corners=True)
                feature_3 = F.grid_sample(f_3, p, padding_mode='border', align_corners=True)
                feature_4 = F.grid_sample(f_4, p, padding_mode='border', align_corners=True)
                feature_5 = F.grid_sample(f_5, p, padding_mode='border', align_corners=True)
                #feature_6 = F.grid_sample(f_6, p, padding_mode='border', align_corners=True)
            
            
            elif VERSION == 'old':
                # feature extraction
                feature_0 = F.grid_sample(f_0, p, padding_mode='border')
                feature_1 = F.grid_sample(f_1, p, padding_mode='border')
                feature_2 = F.grid_sample(f_2, p, padding_mode='border')
                feature_3 = F.grid_sample(f_3, p, padding_mode='border')
                feature_4 = F.grid_sample(f_4, p, padding_mode='border')
                feature_5 = F.grid_sample(f_5, p, padding_mode='border')
                #feature_6 = F.grid_sample(f_6, p, padding_mode='border')
            
        else:
        
            # feature extraction
            feature_0 = grid_sample_3d(f_0, p)
            feature_1 = grid_sample_3d(f_1, p)
            feature_2 = grid_sample_3d(f_2, p)
            feature_3 = grid_sample_3d(f_3, p)
            feature_4 = grid_sample_3d(f_4, p)
            feature_5 = grid_sample_3d(f_5, p)
            #feature_6 = grid_sample_3d(f_6, p)
        

        # here every channel corresponds to one feature.

        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        #print('feature size: {}'.format(features.shape))

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_out(net))
        #net = self.fc_out(net)
        out = net.squeeze(1)

        #print(out)

        #print(out)

        # classification head
        net_cls = self.actvn(self.fc_0_cls(features))
        net_cls = self.actvn(self.fc_1_cls(net_cls))
        net_cls = self.actvn(self.fc_2_cls(net_cls))

        # classification task with no actvn
        out_cls = self.fc_out_cls(net_cls).squeeze(1)  # (B, 1, samples_num) -> (B, samples_num)

        #print(out_cls)

        # return occupancy probabilities for the sampled points
        #print(out_cls)
        p_r = dist.Bernoulli(logits=out_cls)

        return  out, p_r

    def forward(self, p, x):
        out, p_r = self.decoder(p, *self.encoder(x))
        return out, p_r