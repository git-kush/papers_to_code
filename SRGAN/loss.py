import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
import config


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # phi_5,4 5th conv layer before maxpooling but after activation on printing comes out to be at 36
        self.vgg = vgg19(weights=VGG19_Weights).features[:36].eval().to(config.DEVICE)
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)