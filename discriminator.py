import torch.nn as nn
import numpy as np


class DiscriminatorCNN(nn.Module):
    """
    D(x | theta)
    """
    def __init__(self, input_shape, dims):
        super().__init__()

        self.input_shape = input_shape

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

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
