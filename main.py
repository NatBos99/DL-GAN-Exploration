import argparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from GAN import GAN
from generator import GeneratorCNN, GeneratorTransformer
from discriminator import DiscriminatorCNN, DiscriminatorTransformer
from datatsets import get_dataset
from utils import get_args


def training(args, generator, discriminator, train_loader, valid_loader, checkpoint_callback):
    model = GAN(generator, discriminator, lr_gen=args.lr_gen, lr_dis=args.lr_dis, batch_size=args.batch_size,
                no_validation_images=args.no_validation_images, dataset=args.dataset, FID_step=args.FID_step,
                FID_dim=args.FID_dim, fid_max_data=args.fid_max_data)
    gpus = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(gpus=gpus, max_epochs=args.n_epoch,
                         progress_bar_refresh_rate=20) # callbacks=[checkpoint_callback]
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    args = get_args(dataset="MNIST_128", n_epoch=20, no_validation_images=100, fid_max_data=100,
                    FID_dim=2048, FID_step=1)

    # training
    train, valid, test, img_shape = get_dataset(args)

    # gen = GeneratorTransformer(img_shape, args.latent_dim)
    gen = GeneratorCNN(img_shape, args.latent_dim)
    dis = DiscriminatorCNN(img_shape, args.dis_hidden)
    checkpoint_callback = ModelCheckpoint(
                        monitor='FID',
                        dirpath='Checkpoints',
                        filename=f'{gen.__name__}-{args.dataset}'+'{epoch:02d}-{FID:.2f}',
                        save_top_k=5,
                        mode='min',
                    )
    training(args, gen, dis, train, valid, checkpoint_callback)
