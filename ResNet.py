import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

check_layer=True

class BasicBlock(nn.Module):
    def __init__(self,in_plane,block_parm,downsample):
        super(BasicBlock, self).__init__()
        self.main_line=nn.ModuleList([])
        in_channel=in_plane
        for i,layer_parm in enumerate(block_parm):
            ksize = layer_parm[0]
            pad = ksize // 2
            out_channel = layer_parm[1]

            if i==0:
                if downsample:
                    stride=2
                    self.main_line.append(module=nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=ksize, stride=2,padding=pad))
                else:
                    stride=1
                    self.main_line.append(module=nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=ksize, stride=1,padding=pad))
            else:
                self.main_line.append(module=nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=ksize, stride=1,padding=pad))
            self.main_line.append(module=nn.BatchNorm2d(out_channel))
            in_channel=out_channel
        self.out_plane=out_channel
        self.in_plane=in_plane
        self.stride=stride

        self.short_cut=nn.Sequential(nn.Conv2d(self.in_plane, self.out_plane, kernel_size=1, stride=self.stride, bias=False),
                                nn.BatchNorm2d(self.out_plane))

    def forward(self,x):
        feat=x
        for i in range(len(self.main_line)):
            feat=self.main_line[i](feat)
            # print('          after layer_'+str(i)+':',feat.shape)

        if self.out_plane!=self.in_plane or self.stride!=1:
            feat+=self.short_cut(x)
        else:
            feat+=x

        return feat

class Bottleneck(nn.Module):
    def __init__(self,parameter,downsample):
        super(Bottleneck, self).__init__()
        in_plane=parameter['in_plane']
        out_plane=parameter['out_plane']
        self.bottleneck=nn.ModuleList([])
        for i in range(parameter['repeat']):
            if i==0 and downsample:
                block=BasicBlock(in_plane=in_plane,block_parm=parameter['block_parm'],downsample=True)
            else:
                block=BasicBlock(in_plane=in_plane,block_parm=parameter['block_parm'],downsample=False)
            self.bottleneck.append(block)
            in_plane=out_plane

    def forward(self,x):
        feat=x
        for i in range(len(self.bottleneck)):
            feat=self.bottleneck[i](feat)
            # print('      after block_ '+str(i)+':',feat.shape)

        return feat

class Resnet(nn.Module):
    def __init__(self,parameter):
        super(Resnet, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3,bias=False),
                                 nn.BatchNorm2d(64),                            #n*64*112*112
                                 nn.MaxPool2d(kernel_size=3,stride=2))          #n*64*55*55
        self.conv2=Bottleneck(parameter['conv2_x'],downsample=False)            #n*256*55*55
        self.conv3=Bottleneck(parameter['conv3_x'],downsample=True)             #n*512*28*28
        self.conv4=Bottleneck(parameter['conv4_x'],downsample=True)             #n*1024*14*14
        self.conv5=Bottleneck(parameter['conv5_x'],downsample=True)             #n*2048*7*7

        self.fc_layer=nn.Sequential(nn.AvgPool2d(kernel_size=7),                #n*2048*1*1
                                    nn.Flatten(),                               #n*2048
                                    nn.Linear(parameter['out_plane'],1000),     #n*1000
                                    nn.Softmax(dim=1))                          #n*1000
        # self.fc_1=nn.AvgPool2d(kernel_size=7)
        # self.fc_2=nn.Flatten()
        # self.fc_3=nn.Linear(parameter['out_plane'],1000)
        # self.fc_4=nn.Softmax()

    def forward(self,x):
        feat=self.conv1(x)
        # print('after conv1:', feat.shape)
        feat=self.conv2(feat)
        # print('after conv2:', feat.shape)
        feat=self.conv3(feat)
        # print('after conv3:', feat.shape)
        feat=self.conv4(feat)
        # print('after conv4:',feat.shape)
        feat=self.conv5(feat)
        # print('after conv5:',feat.shape)
        feat=self.fc_layer(feat)
        # print('after fc:', feat.shape)
        return feat

def resnet_18():
    parameter={
        'conv2_x':{'in_plane': 64,'block_parm':[[3, 64],[3, 64]],'out_plane': 64, 'repeat':2},
        'conv3_x':{'in_plane': 64,'block_parm':[[3,128],[3,128]],'out_plane':128, 'repeat':2},
        'conv4_x':{'in_plane':128,'block_parm':[[3,256],[3,256]],'out_plane':256, 'repeat':2},
        'conv5_x':{'in_plane':256,'block_parm':[[3,512],[3,512]],'out_plane':512, 'repeat':2},
        'out_plane':512}
    net=Resnet(parameter)
    return net

def resnet_34():
    parameter={
        'conv2_x':{'in_plane': 64,'block_parm':[[3, 64],[3, 64]],'out_plane': 64, 'repeat':3},
        'conv3_x':{'in_plane': 64,'block_parm':[[3,128],[3,128]],'out_plane':128, 'repeat':4},
        'conv4_x':{'in_plane':128,'block_parm':[[3,256],[3,256]],'out_plane':256, 'repeat':6},
        'conv5_x':{'in_plane':256,'block_parm':[[3,512],[3,512]],'out_plane':512, 'repeat':3},
        'out_plane':512}
    net=Resnet(parameter)
    return net

def resnet_50():
    parameter={
        'conv2_x':{'in_plane':  64,'block_parm':[[1, 64],[3, 64],[1, 256]],'out_plane': 256, 'repeat':3},
        'conv3_x':{'in_plane': 256,'block_parm':[[1,128],[3,128],[1, 512]],'out_plane': 512, 'repeat':4},
        'conv4_x':{'in_plane': 512,'block_parm':[[1,256],[3,256],[1,1024]],'out_plane':1024, 'repeat':6},
        'conv5_x':{'in_plane':1024,'block_parm':[[1,512],[3,512],[1,2048]],'out_plane':2048, 'repeat':3},
        'out_plane':2048}
    net=Resnet(parameter)
    return net

def resnet_101():
    parameter={
        'conv2_x':{'in_plane':  64,'block_parm':[[1, 64],[3, 64],[1, 256]],'out_plane': 256, 'repeat': 3},
        'conv3_x':{'in_plane': 256,'block_parm':[[1,128],[3,128],[1, 512]],'out_plane': 512, 'repeat': 4},
        'conv4_x':{'in_plane': 512,'block_parm':[[1,256],[3,256],[1,1024]],'out_plane':1024, 'repeat':23},
        'conv5_x':{'in_plane':1024,'block_parm':[[1,512],[3,512],[1,2048]],'out_plane':2048, 'repeat': 3},
        'out_plane':2048}
    net=Resnet(parameter)
    return net

def resnet_152():
    parameter={
        'conv2_x':{'in_plane':  64,'block_parm':[[1, 64],[3, 64],[1, 256]],'out_plane': 256, 'repeat': 3},
        'conv3_x':{'in_plane': 256,'block_parm':[[1,128],[3,128],[1, 512]],'out_plane': 512, 'repeat': 8},
        'conv4_x':{'in_plane': 512,'block_parm':[[1,256],[3,256],[1,1024]],'out_plane':1024, 'repeat':36},
        'conv5_x':{'in_plane':1024,'block_parm':[[1,512],[3,512],[1,2048]],'out_plane':2048, 'repeat': 3},
        'out_plane':2048}
    net=Resnet(parameter)
    return net

def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == '__main__':
    size = 224
    net = resnet_18()
    x = torch.autograd.Variable(torch.FloatTensor(4, 3, size, size).uniform_(-1, 1))

    out = net(x)