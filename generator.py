from typing import List

import torch
import torch.nn as nn
import math
import warnings


class GeneratorCNN(nn.Module):
    """
    G(z|theta)
    """

    def __init__(self,
                 image_shape,  # channels x width x height
                 latent_dim,
                 starting_layer_dim: int = 128
                 ):
        super(GeneratorCNN, self).__init__()
        self.latent_dim = latent_dim
        self.starting_layer_dim = starting_layer_dim
        self.init_width = image_shape[1] // 4
        self.init_height = image_shape[2] // 4

        self.linear_layer = nn.Linear(latent_dim, starting_layer_dim * self.init_width * self.init_height)

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

    def __name__(self):
        return "GeneratorCNN"


    def forward(self, z):
        """
        z - latent representation
        :return:
        """
        out = self.linear_layer(z)
        out = out.view(out.shape[0], self.starting_layer_dim, self.init_width, self.init_height)
        img = self.conv_blocks(out)
        return img


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def pixel_upsample(x, H, W):
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class GeneratorTransformer(nn.Module):
    """
    G(z|theta)
    """

    def __init__(self,
                 image_shape,  # channels x width x height
                 latent_dim,
                 starting_layer_dim: int = 128,
                 encoder_stack_dims: List[int] = None
                 ):
        super().__init__()
        if encoder_stack_dims is None:
            encoder_stack_dims = [5, 2, 2]
        self.latent_dim = latent_dim
        self.starting_layer_dim = starting_layer_dim
        self.init_width = image_shape[1] // 4
        self.init_height = image_shape[2] // 4

        self.linear_layer = nn.Linear(latent_dim, starting_layer_dim * self.init_width * self.init_height)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, self.init_width ** 2, starting_layer_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, (2 * self.init_width) ** 2, starting_layer_dim // 4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, (4 * self.init_width) ** 2, starting_layer_dim // 16))

        self.pos_embed = [self.pos_embed_1, self.pos_embed_2, self.pos_embed_3]
        for i in range(len(self.pos_embed)):
            trunc_normal_(self.pos_embed[i], std=.02)

        self.block1 = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=starting_layer_dim, nhead=4, dim_feedforward=starting_layer_dim * 4)
             for i in range(encoder_stack_dims[0])])
        self.block2 = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=starting_layer_dim // 4, nhead=4, dim_feedforward=starting_layer_dim)
             for i in range(encoder_stack_dims[1])])
        self.block3 = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=starting_layer_dim // 16, nhead=4, dim_feedforward=starting_layer_dim // 4)
             for i in range(encoder_stack_dims[2])])

        self.deconv = nn.Sequential(
            nn.Conv2d(self.starting_layer_dim // 16, image_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.image_shape = image_shape

    def __name__(self):
        return "GeneratorTransformer"

    def forward(self, z):
        """
        z - latent representation
        :return:
        """
        out = self.linear_layer(z).view(-1, self.init_width * self.init_height, self.starting_layer_dim)
        out = out + self.pos_embed[0].to(out.get_device())
        B = out.size()
        H, W = self.init_width, self.init_height

        for index, blk in enumerate(self.block1):
            out = blk(out)

        out, H, W = pixel_upsample(out, H, W)
        out = out + self.pos_embed[1].to(out.get_device())
        for index, blk in enumerate(self.block2):
            out = blk(out)

        out, H, W = pixel_upsample(out, H, W)
        out = out + self.pos_embed[2].to(out.get_device())
        for index, blk in enumerate(self.block3):
            out = blk(out)

        out = self.deconv(out.permute(0, 2, 1).view(-1, self.starting_layer_dim // 16, H, W))
        return out


if __name__ == "__main__":
    a = GeneratorCNN((1, 28, 28), 100)
    print()
