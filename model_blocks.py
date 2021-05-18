from torch import nn
import torch.nn.functional as F


class ConvolutionalBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            up_mode,
            norm=None,
            short_cut=False,
            num_skip_in=0,
    ):
        super(ConvolutionalBlock, self).__init__()
        self.skip_in_ops = None
        self.c_sc = None
        self.up_mode = up_mode

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        if norm:
            self.n1 = nn.BatchNorm2d(in_channels)
            self.n2 = nn.BatchNorm2d(out_channels)
        else:
            self.n1 = nn.Identity()
            self.n2 = nn.Identity()

        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if num_skip_in:
            self.skip_in_ops = nn.ModuleList(
                [
                    nn.Conv2d(out_channels, out_channels, kernel_size=1)
                    for _ in range(num_skip_in)
                ]
            )

    def forward(self, x, skip_ft=None):
        residual = self.n1(x)
        h = nn.ReLU()(residual)
        h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h

        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h)
        h = nn.ReLU()(h)
        final_out = self.c2(h)

        # shortcut
        if self.c_sc:
            final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(
        self, d_spectral_norm, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()
    ):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(
        self,
        d_spectral_norm,
        in_channels,
        out_channels,
        hidden_channels=None,
        ksize=3,
        pad=1,
        activation=nn.ReLU(),
        downsample=False,
    ):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=ksize, padding=pad
        )
        self.c2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=ksize, padding=pad
        )
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)