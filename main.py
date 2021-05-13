import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

import pytorch_lightning as pl

from GAN import GAN
from generator import GeneratorCNN, GeneratorTransformer
from discriminator import DiscriminatorCNN
from datatsets import get_dataset
from utils import get_args


def training(args, generator, discriminator, train_loader, valid_loader, checkpoint_callback):
    model = GAN(generator, discriminator, batch_size=args.batch_size)
    gpus = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(gpus=gpus, max_epochs=args.n_epoch,
                         progress_bar_refresh_rate=20) # callbacks=[checkpoint_callback]
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    args = get_args()

    # training
    train, valid, test, img_shape = get_dataset(args)

    gen = GeneratorTransformer(img_shape, args.latent_dim)
    dis = DiscriminatorCNN(img_shape)

    # checkpoint_callback = pl.ModelCheckpoint(
    #                     monitor='FID',
    #                     dirpath='Checkpoints',
    #                     filename=f'{gen.__name__}-{args.dataset}'+'{epoch:02d}-{FID:.2f}',
    #                     save_top_k=5,
    #                     mode='min',
    #                 )
    checkpoint_callback = 1
    training(args, gen, dis, train, valid, checkpoint_callback)
