import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 discriminator=False,
                 use_act=True,
                 use_bn=True,
                 **kwargs
                 ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels,**kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)

        )
    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c * scale_factor ** 2, 3, 1, 1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(in_channels, in_channels,kernel_size=3,stride=1, padding=1)
        self.block2 = ConvBlock(in_channels,in_channels,kernel_size=3,stride=1,padding=1,use_act=False)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x   # considering skip connection

class Generator(nn.Module):
    def __init__(self, in_channels=3,num_channels=64, num_block=16):
        super().__init__()
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)                             #tukka
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_block)])
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3,stride=1, padding=1, use_act=False)
        self.upsample = nn.Sequential(UpsampleBlock(num_channels, 2), UpsampleBlock(num_channels, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.convblock(x) + initial           # skip connection
        x = self.upsample(x)
        return torch.tanh(self.final(x))                        #doubtfull not mentioned about the activation fxn used


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, featurs= [64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for index, featurs in enumerate(featurs):
            blocks.append(
                ConvBlock(
                    in_channels,
                    featurs,
                    kernel_size=3,
                    stride= 1 + index % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn= False if index ==0 else True,

                )
            )
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024,1 ),
        )


    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)



def test():
    low_resolution = 24
    device = torch.device('cpu')
    with torch.cpu.amp.autocast():
        x = torch.randn((5,3,low_resolution, low_resolution))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

if __name__ == "__main__":
    test()