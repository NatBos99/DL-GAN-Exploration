import argparse
import os.path
from shutil import rmtree


import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from GAN import GAN
from generator import GeneratorCNN, GeneratorTransformer, GeneratorAutoGAN, PretrainedGeneratorTransformer
from discriminator import DiscriminatorCNN, DiscriminatorTransformer, DiscriminatorAutoGAN
from datatsets import get_dataset
from utils import get_args


def training(args, generator, discriminator, train_loader, valid_loader, checkpoint_callback=None):
    model = GAN(generator, discriminator, lr_gen=args.lr_gen, lr_dis=args.lr_dis, batch_size=args.batch_size,
                no_validation_images=args.no_validation_images, dataset=args.dataset, FID_step=args.FID_step,
                FID_dim=args.FID_dim, fid_max_data=args.fid_max_data)
    gpus = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(gpus=gpus, max_epochs=args.n_epoch,
                         progress_bar_refresh_rate=20, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    args = get_args(dataset="CIFAR10", n_epoch=10, no_validation_images=1000, fid_max_data=1000,
                    FID_dim=2048, FID_step=1, latent_dim=128, train_valid_split=0.9)

    #  delete the previously created images
    dest = 'Validation-Gen-Images'
    if os.path.exists(dest):
        rmtree(dest)
    # training
    train, valid, test, img_shape = get_dataset(args)

    # gen = GeneratorAutoGAN(channels=64, bottom_width=4, latent_dim=128, out_channels=3)
    dis = DiscriminatorAutoGAN(channels=80, in_channels=3)

    print(args.batch_size)
    gen = PretrainedGeneratorTransformer(img_shape, 128, batch_size=args.batch_size, freeze_net=True)
    # gen = GeneratorCNN(img_shape, 96)
    # dis = DiscriminatorCNN(img_shape, args.dis_hidden)
    # dis = DiscriminatorTransformer(img_shape)
    checkpoint_callback = ModelCheckpoint(
                        monitor='FID',
                        dirpath='Checkpoints',
                        filename=f"{gen.name}{dis.name}-{args.dataset}_"+'{epoch:02d}-{FID:.2f}',
                        save_top_k=10,
                        mode='min',
                        every_n_val_epochs=args.FID_step,
                    )
    training(args, gen, dis, train, valid, checkpoint_callback)
