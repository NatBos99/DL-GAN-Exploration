import torch.nn as nn
import numpy as np


class Discriminator_CNN(nn.Module):
    """
    D(x | theta)
    """
    def __init__(self, image_shape):
        super().__init__()

        image_size = np.prod(image_shape)

        self.model = nn.Sequential(
            nn.Linear(image_size, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )


    def forward(self, img):
        """
        img  - raw img
        :return:
        """
        img_flattened = img.view(img.size(0), -1)
        img = self.model(img_flattened)
        return img
