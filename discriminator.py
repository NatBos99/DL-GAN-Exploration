import torch.nn as nn
import numpy as np


class DiscriminatorCNN(nn.Module):
    """
    D(x | theta)
    """
    def __init__(self, input_shape, first_hidden_dim=16):
        super().__init__()

        self.input_shape = input_shape

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(self.input_shape[0], first_hidden_dim, bn=False),
            *discriminator_block(first_hidden_dim, first_hidden_dim * 2),
            *discriminator_block(first_hidden_dim * 2, first_hidden_dim * 4),
            *discriminator_block(first_hidden_dim * 4, first_hidden_dim * 8),
        )

        # The height and width of downsampled image
        ds_size = self.input_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(first_hidden_dim * 8 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
