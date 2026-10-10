import torch
import torch.nn as nn
import torch.nn.functional as F
from wtconv2d import WTConv2d
import torch.utils.model_zoo as model_zoo
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(BasicBlock, self).__init__()

        self.wtconv = WTConv2d(in_channel, in_channel, kernel_size=3, stride=stride, wt_levels=1, wt_type='db1')
        self.conv01 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.bn01 = nn.BatchNorm2d(out_channel)
        self.relu0 = nn.ReLU()
        
    def forward(self, x):
        #print("xinput.shape_line19==",x.shape)
        out = self.wtconv(x)
        #print("xwtconv.shape_line21==",x.shape)
        out = self.conv01(out)
        #print("x01.shape_line23 ==",x.shape)
        
        out = self.bn01(out)
        out = self.relu0(out)

        return out


class BodyBlock(nn.Module):
    def __init__(self, in_channels, out_channels, copy_cnt, specical_stride=1):
        super(BodyBlock, self).__init__()
        self.copy_cnt = copy_cnt
        mid_channels = out_channels // 4
        #print("our_cha",in_channels,out_channels,mid_channels)
        self.conv1 = nn.Sequential(
            BasicBlock(in_channels, mid_channels, 1, 1, 0),
            BasicBlock(mid_channels, mid_channels, 3, specical_stride, 1),
            BasicBlock(mid_channels, out_channels, 1, 1, 0)
        )
        
        self.conv2 = BasicBlock(in_channels, out_channels, 1, specical_stride, 0)

        self.conv3 = nn.Sequential(
            BasicBlock(out_channels, mid_channels, 1, 1, 0),
            BasicBlock(mid_channels, mid_channels, 3, 1, 1),
            BasicBlock(mid_channels, out_channels, 1, 1, 0)
        )
    
    def forward(self, x):

        x = self.conv1(x) + self.conv2(x)
        
        for _ in range(self.copy_cnt):
            identity = x
            x = self.conv3(x) + identity
        return x



class WTConvNeXt(nn.Module):
    def __init__(self, num_classes=1000):
        super(WTConvNeXt, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        self.layer1 = BodyBlock(in_channels=64, out_channels=144, copy_cnt=3, specical_stride=1)
        self.layer2 = BodyBlock(in_channels=144, out_channels=288, copy_cnt=4, specical_stride=2)
        self.layer3 = BodyBlock(in_channels=288, out_channels=576, copy_cnt=6, specical_stride=2)
        self.layer4 = BodyBlock(in_channels=576, out_channels=1152, copy_cnt=3, specical_stride=2)

    def forward(self, x):
        
        #print(x.shape)
        x =  self.layer0(x)
        print("##########################x0.shape = ", x.shape)
        x =  self.layer1(x)
        print("############################x1.shape = ", x.shape)
        x =  self.layer2(x)
        print("##########################x2.shape = ", x.shape)
        x =  self.layer3(x)
        print("###############################x3.shape = ", x.shape)
        x =  self.layer4(x)
        print("################################x4.shape = ", x.shape)


        return x

def main():
    X = torch.randn(4, 1, 1024, 1024).cuda(0)
    
    net = WTConvNeXt().cuda(0)
    X = net(X)
    
    ##print(X.shape)
    
if __name__ == '__main__':
    main()


