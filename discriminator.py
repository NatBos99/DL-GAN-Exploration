import torch
import torch.nn as nn
import numpy as np
import math


class DiscriminatorCNN(nn.Module):
    """
    D(x | theta)
    """
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # self.model = nn.Sequential(
        #     *discriminator_block(self.input_shape[0], 16, bn=False),
        #     *discriminator_block(16, 32),
        #     *discriminator_block(32, 64),
        #     *discriminator_block(64, 128),
        # )

        self.model = nn.Sequential(
            *discriminator_block(self.input_shape[0], 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 256),
        )

        # The height and width of downsampled image
        ds_size = self.input_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.ReLU(), nn.Linear(256 * ds_size ** 2, 1))#, nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


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


class DiscriminatorTransformer(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.input_shape = input_shape
        self.patch_size = 3
        self.latent_dim = 128
        self.patch_embed = nn.Conv2d(self.input_shape[0], self.latent_dim, kernel_size=self.patch_size, stride=self.patch_size, padding=0)
        num_patches = (self.input_shape[1] // self.patch_size)**2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.latent_dim))
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=self.latent_dim, nhead=4, dim_feedforward=self.latent_dim*4) for i in range(4)])

        self.norm = nn.LayerNorm(self.latent_dim)
        
        self.head = nn.Linear(self.latent_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img):
        img = self.patch_embed(img)
        img = img.flatten(2)
        img = img.permute(0,2,1)
        B = img.shape[0]

        cls_tokens = self.cls_token.expand(B, 1, -1)
        img = torch.cat((cls_tokens, img), dim=1)
        img += self.pos_embed

        for blk in self.blocks:
            img = blk(img)

        img = self.norm(img)[:,0]

        validity = self.head(img)
        return validity