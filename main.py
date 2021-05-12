import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

import pytorch_lightning as pl

from GAN import GAN
from generator import GeneratorCNN
from discriminator import DiscriminatorCNN


def set_up(args):
    if args.dataset == "CIFAR10":
        data = CIFAR10(root="Datasets/cifar-10-batches-py",
                       download=True,
                       transform=transforms.ToTensor())
    elif args.dataset == "MNIST":
        new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
        MNIST.resources = [
            ('/'.join([new_mirror, url.split('/')[-1]]), md5)
            for url, md5 in MNIST.resources
        ]
        data = MNIST(root="Datasets/MNIST",
                     download=True,
                     transform=transforms.Compose(
                         [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                     ),
                     )
    else:
        raise NotImplementedError

    img, _ = data[1]  # take second image
    img_shape = img.size()
    data_loader = DataLoader(data,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             )

    return data_loader, list(img_shape)


def train(args, generator, discriminator, data_loader):
    model = GAN(generator, discriminator, batch_size=args.batch_size)
    trainer = pl.Trainer(gpus=None, max_epochs=args.n_epoch,
                         progress_bar_refresh_rate=20)
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple GAN implementation"
    )
    parser.add_argument('--dataset', type=str, default="MNIST",
                        help='options are as follows: "CIFAR10", ')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--n_epoch', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='if we want to shuffle our dataset')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='if we want to shuffle our dataset')

    args = parser.parse_args()

    CIFAR10_data_loader, img_shape = set_up(args)

    gen = GeneratorCNN(img_shape, args.latent_dim)
    dis = DiscriminatorCNN(img_shape)

    train(args, gen, dis, CIFAR10_data_loader)
