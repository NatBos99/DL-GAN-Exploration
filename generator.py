import torch.nn as nn
import numpy as np

class GeneratorCNN(nn.Module):
    """
    G(z|theta)
    """
    def __init__(self,
                 image_shape, #channels x width x height
                 latent_dim,
                 starting_layer_dim: int = 128
                 ):
        super().__init__()
        self.latent_dim = latent_dim
        self.starting_layer_dim = starting_layer_dim
        self.init_width = image_shape[1]
        self.init_height = image_shape[2]


        self.linear_layer = nn.Sequential(nn.Linear(latent_dim,
                                                    starting_layer_dim * self.init_width * self.init_height))


        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, image_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )


        self.image_shape = image_shape

    def forward(self, z):
        """
        z - latent representation
        :return:
        """
        out = self.linear_layer(z)
        out = out.view(out.shape[0], self.starting_layer_dim, self.init_width, self.init_height)
        img = self.conv_blocks(out)
        return img

if __name__ == "__main__":
    a = GeneratorCNN((1, 28, 28), 100)
    print()
