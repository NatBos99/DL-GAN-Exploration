import argparse

import torch
import torchvision
from torchvision.transforms import ToTensor
import pytorch_lightning as pl

from GAN import GAN
from generator import Generator_CNN
from discriminator import Discriminator_CNN


def set_up(args):
    if args.dataset == "CIFAR10":
        data = torchvision.datasets.CIFAR10(root="Datasets/cifar-10-batches-py",
                                            download=True,
                                            transform=ToTensor())
    elif args.dataset == "MNIST":
        data = torchvision.datasets.MNIST(root="Datasets/MNIST",
                                            download=True,
                                            transform=ToTensor())
    else:
        raise NotImplementedError

    data_loader = torch.utils.data.DataLoader(data,
                                            batch_size=args.batch_size,
                                            shuffle=args.shuffle,
                                            )

    # get the correct image dimensions
    batch_input, _ = next(iter(data_loader))
    img_shape = batch_input.shape[1:] # since this is a batch we do not need the first element
    return data_loader, img_shape

def train(args, gen, dis, data_loader):
    model = GAN(gen, dis, batch_size=args.batch_size)
    trainer = pl.Trainer(gpus=1, max_epochs=args.n_epoch, progress_bar_refresh_rate=20)
    trainer.fit(model, data_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple GAN implementation"
    )
    parser.add_argument('--dataset', type=str, default="MNIST",
                        help='options are as follows: "CIFAR10", ')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')

    parser.add_argument('--n_epoch', type=int, default=2, help='number of epochs')
    parser.add_argument('--shuffle', type=bool, default=True, help='if we want to shuffle our dataset')
    parser.add_argument('--latent_dim', type=int, default=100, help='if we want to shuffle our dataset')

    args = parser.parse_args()

    CIFAR10_data_loader, img_shape = set_up(args)

    gen = Generator_CNN(img_shape, args.latent_dim)
    dis = Discriminator_CNN(img_shape)

    train(args, gen, dis, CIFAR10_data_loader)
