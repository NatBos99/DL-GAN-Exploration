import torch.nn as nn
import numpy as np

class GeneratorCNN(nn.Module):
    """
    G(z|theta)
    """
    def __init__(self, image_shape, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        cell = lambda in_shape, out_shape: [nn.Linear(in_shape, out_shape),
                                            nn.BatchNorm1d(out_shape),
                                            nn.LeakyReLU(negative_slope=0.2)
                                            ]
        self.image_shape = image_shape
        image_size = np.prod(image_shape)

        self.model = nn.Sequential(
            *cell(self.latent_dim, 128),
            *cell(128, 256),
            *cell(256, 512),
            *cell(512, 1024),
            nn.Linear(1024, image_size),
        )

    # @property
    # def latent_dim(self):
    #     """Gets number of latent dimensions """
    #     return self._latent_dim
    #
    # @latent_dim.setter
    # def latent_dim(self, value):
    #     self._latent_dim = value

    def forward(self, z):
        """
        z - latent representation
        :return:
        """
        flat_img = self.model(z)
        img = flat_img.view(flat_img.size(0), *self.image_shape)
        return img

if __name__ == "__main__":
    a = GeneratorCNN((1, 28, 28), 100)
    print()
