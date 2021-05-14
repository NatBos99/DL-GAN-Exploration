import argparse
import os
import time

import torch
from pytorch_fid import fid_score
from torchvision.utils import save_image
from datatsets import get_dataset


def get_args(
            dataset="MNIST",
            batch_size=32,
            n_epoch=2,
            shuffle=True,
            latent_dim=100,
            image_size=32,
            train_valid_split=0.9
            ):
    # cli arguments
    parser = argparse.ArgumentParser(
        description="Simple GAN implementation"
    )
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='options are as follows: "CIFAR10", MNIST, FashionMNIST, CelebA')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')

    parser.add_argument('--n_epoch', type=int, default=n_epoch,
                        help='number of epochs')
    parser.add_argument('--shuffle', type=bool, default=shuffle,
                        help='if we want to shuffle our dataset')
    parser.add_argument('--latent_dim', type=int, default=latent_dim,
                        help='Latent dimension of the generator')

    parser.add_argument('--image_size', type=int, default=image_size,
                        help='')

    parser.add_argument('--train_valid_split', type=float, default=train_valid_split,
                        help='Training validation split in favor of training set')

    parser.add_argument('--custom_mnist_download', dest="custom_mnist_download",
                        action="store_true", help="Use in case the base approach does not work")

    args = parser.parse_args()

    return args

def create_dir_from_tensors(Tensors, dir_name="Validation-Gen-Images"):
    """

    :param Tensors: img_tensor to be saved
    :param dir_name: directory to save the images to
    :return: path to directory
    """
    # Create Dir if it does not exist
    os.makedirs(dir_name, exist_ok=True)
    # unpack sensor?
    for img in Tensors:
        save_image(img, f'{dir_name}/gen-img{time.time():.20f}.png')

    return dir_name

def compute_FID(imgs, args, device, dims, valid_loader):
    fake_path = create_dir_from_tensors(imgs)

    real_img_path = f"Datasets/{args.dataset}/{args.dataset}/processed"

    paths = [fake_path, real_img_path]
    fid_score.calculate_fid_given_paths(paths, args.batch_size, device, dims, valid_loader)


if __name__ == "__main__":

    args = get_args(batch_size=16)

    train_loader, valid_loader, test_loader, img_shape = get_dataset(args)
    imgs = torch.randn(32, 1, 32, 32).type(torch.float32)

    compute_FID(imgs, args, 'cpu', 192, valid_loader)
    # {64: 0, 192: 1, 768: 2, 2048: 3}