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
from datatsets import get_dataset


def training(args, generator, discriminator, train_loader, valid_loader):
    model = GAN(generator, discriminator, batch_size=args.batch_size)
    gpus = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(gpus=gpus, max_epochs=args.n_epoch,
                         progress_bar_refresh_rate=20)
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple GAN implementation"
    )
    parser.add_argument('--dataset', type=str, default="MNIST",
                        help='options are as follows: "CIFAR10", MNIST, Fahsion-MNIST, CelebA')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--n_epoch', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='if we want to shuffle our dataset')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Latent dimension of the generator')

    parser.add_argument('--image_size', type=int, default=32,
                        help='')

    parser.add_argument('--train_valid_split', type=float, default=0.9,
                        help='Training validation split in favor of training set')

    args = parser.parse_args()

    train, valid, test, img_shape = get_dataset(args)

    gen = GeneratorCNN(img_shape, args.latent_dim)
    dis = DiscriminatorCNN(img_shape)

    training(args, gen, dis, train, valid)
