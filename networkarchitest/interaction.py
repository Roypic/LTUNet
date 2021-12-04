import torch
from torch import nn, einsum
from einops import rearrange
from networkarchitest.former import Former
from torchsummaryX import summary
from collections import *


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class conv_attention_neck(nn.Module):
    def __init__(self, ch_in, mid,patches):
        super(conv_attention_neck, self).__init__()
        self.relu=nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, mid, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True))

        self.former = Former(dim=192)
        self.Conv2Fromer = Conv2Fromer(dim=192, heads=2, c=mid, in_dim=mid, dropout=0,patches=patches)

        self.conv2=nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True)
        )
        self.Former2Conv = Former2Conv(dim=192, heads=2, c=mid)
        self.conv3=nn.Sequential(
            nn.Conv2d(mid, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        x, z = inputs
        x_mid = self.conv1(x)
        z_hid = self.Conv2Fromer(x_mid, z)
        z_out = self.former(z_hid)
        x_hid = self.conv2(x_mid)
        x_out = self.Former2Conv(x_hid, z_out)
        x_out = self.conv3(x_out)+x
        x_out = self.relu(x_out)
        return [x_out, z_out]



class ConvFormerblock(nn.Module):
    def __init__(self, inp, exp, out,patches):
        super(ConvFormerblock, self).__init__()
        self.inconv=conv_block(inp,out)
        self.mixconv=conv_attention_neck(out,exp,patches)

    def forward(self, inputs):
        x, z = inputs
        x_mid=self.inconv(x)
        x_out,z_out=self.mixconv([x_mid,z])
        return [x_out, z_out]

class Conv2Fromer(nn.Module):
    def __init__(self, dim, heads, c, in_dim, dropout=0.,patches=[2,3]):
        super(Conv2Fromer, self).__init__()
        inner_dim = heads * c
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim)
        self.adjust_dim_pool=nn.Conv2d(in_dim, 192, kernel_size=1, stride=1, padding=0, bias=True)
        self.attend = nn.Softmax(dim=-1)
        self.scale = c ** -0.5
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.patchfeature = nn.AdaptiveAvgPool2d((patches[0], patches[1]))
    def forward(self, x, z):
        b, m, d = z.shape
        #print(b, m, d)
        b, c, h, w = x.shape

        x_pool= self.adjust_dim_pool(x)
        x_pool=self.patchfeature(x_pool)


        x_pool = x_pool.contiguous().view(b, d, m)
        x_pool = x_pool.permute(0, 2, 1)
        z=z+x_pool

        #x_pool+
        #x_pool=x_pool.contiguous().view(b, m, d)
        # b l c -> b l h*c -> b h l c
        x = x.contiguous().view(b, h * w, c).repeat(1, 1, self.heads)
        x = x.contiguous().view(b, self.heads, h * w, c)
        k, v = x, x
        # b m d -> b m h*c -> b h m c
        q = self.to_q(z).view(b, self.heads, m, c)
        #print(b, self.heads, m, c)
        dots = einsum('b h m c, b h l c -> b h m l', q, k) * self.scale
        # b h m l
        attn = self.attend(dots)
        out = einsum('b h m l, b h l c -> b h m c', attn, v)
        out = rearrange(out, 'b h m c -> b m (h c)')
        return z + self.to_out(out)


# inputs: x(b c h w) z(b m d)
# output: x(b c h w)
class Former2Conv(nn.Module):
    def __init__(self, dim, heads, c, dropout=0.):
        super(Former2Conv, self).__init__()
        inner_dim = heads * c
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = c ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, c),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        x_ = x.contiguous().view(b, h * w, c).repeat(1, 1, self.heads)
        x_ = x_.contiguous().view(b, self.heads, h * w, c)
        q = x_
        # b h m c
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        # b h l m
        dots = einsum('b h l c, b h m c -> b h l m', q, k) * self.scale
        # b h l m
        attn = self.attend(dots)
        out = einsum('b h l m, b h m c -> b h l c', attn, v)
        out = rearrange(out, 'b h l c -> b l (h c)')
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out


class LTUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1 ,inner_dim=192,patches=[3,3]):
        super(LTUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.token = nn.Parameter(nn.Parameter(torch.randn(1, patches[0]*patches[1], 192)))
        self.block1= ConvFormerblock(img_ch,64,64,patches)
        self.block2=self._make_layer(64,2,inner_dim,patches)
        self.block3=self._make_layer(128,2,inner_dim,patches)
        self.block4=self._make_layer(256,2, inner_dim,patches)
        self.block5=self._make_layer(512,2, inner_dim,patches)

        self.neck= nn.Sequential(
                    nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True))



        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, inplanes,expansion,innderdim,patches):
        layers = []
        #layers.append(ConvFormerblock(img_ch,64,64))
        # inplanes expand for next block
        self.inplanes=inplanes
        self.outplanes = inplanes * expansion
        if self.outplanes>=innderdim:
            layers.append(ConvFormerblock(self.inplanes, innderdim,self.outplanes,patches))
        else:
            layers.append(ConvFormerblock(self.inplanes, self.outplanes, self.outplanes,patches))
        return nn.Sequential(*layers)
    def forward(self, x):
        b, _, _, _ = x.shape
        z = self.token.repeat(b, 1, 1)
        x1,z=self.block1([x,z])
        x2=self.Maxpool(x1)
        x2, z = self.block2([x2, z])
        x3 = self.Maxpool(x2)
        x3, z = self.block3([x3, z])
        x4 = self.Maxpool(x3)
        x4, z = self.block4([x4, z])
        x5 = self.Maxpool(x4)
        x5, z = self.block5([x5, z])



        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


if __name__ == "__main__":
    model = LTUNet()
    #print(summary(model, torch.zeros(2,3, 224, 224)))
    inputs = torch.randn((3, 3,224,224))
    print(inputs.shape)
    print("Total number of parameters in networks is {} M".format(sum(x.numel() for x in model.parameters()) / 1e6))
    output = model(inputs)
    print(output[0].shape)