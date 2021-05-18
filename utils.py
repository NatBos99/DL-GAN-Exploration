import argparse
import os
import time
import struct
import shutil
# import _pickle as cPickle
import pickle
from array import array


import numpy as np

import torch
from pytorch_fid import fid_score
from torchvision.utils import save_image
from datatsets import get_dataset

MY_PATH = os.path.abspath(os.path.dirname(__file__))

DATASET_DIR = {"MNIST": os.path.join(MY_PATH, "Datasets/MNIST/MNIST/raw/t10k-images-idx3-ubyte"),
               "FashionMNIST": os.path.join(MY_PATH, "Datasets/FashionMNIST/FashionMNIST/raw/t10k-images-idx3-ubyte"),
                "MNIST_128":  os.path.join(MY_PATH, "Datasets/MNIST/MNIST/raw/t10k-images-idx3-ubyte"),
                "CIFAR10":  os.path.join(MY_PATH, "Datasets/cifar-10-batches-py/data_batch_1")
                }
UBYTE_DATASETS = ["MNIST", "FashionMNIST", "MNIST_128"]



def get_args(
            dataset: str = "MNIST",
            batch_size: int = 32,
            n_epoch: int = 2,
            shuffle: bool = True,
            latent_dim: int = 100,
            image_size: int = 32,
            train_valid_split: float = 0.9,
            fid_max_data: int = 10000,
            no_validation_images: int = 16
            ):
    # cli arguments
    parser = argparse.ArgumentParser(
        description="Simple GAN implementation"
    )
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='options are as follows: "CIFAR10", MNIST, MNIST_128, FashionMNIST, CelebA')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')

    parser.add_argument('--n_epoch', type=int, default=n_epoch,
                        help='number of epochs')
    parser.add_argument('--shuffle', type=bool, default=shuffle,
                        help='if we want to shuffle our dataset')
    parser.add_argument('--latent_dim', type=int, default=latent_dim,
                        help='Latent dimension of the generator')

    parser.add_argument('--image_size', type=int, default=image_size,
                        help='')
    parser.add_argument('--lr_gen', type=float, default=1E-3,
                        help='Learning rate for the generator')
    parser.add_argument('--lr_dis', type=float, default=1E-3,
                        help='Learning rate for the discriminator')
    parser.add_argument('--train_valid_split', type=float, default=train_valid_split,
                        help='Training validation split in favor of training set')

    parser.add_argument('--custom_mnist_download', dest="custom_mnist_download",
                        action="store_true", help="Use in case the base approach does not work")
    parser.add_argument('--num_workers', type=int, default=4,
                         help="How many CPU workers should dataloaders use.")

    parser.add_argument('--dis_hidden', type=int, default=16,
                        help="Number of channels in first block of convolutional discriminator, doubles with each block.")

    parser.add_argument('--no_validation_images', type=int, default=no_validation_images,
                        help="Number of validation images to create")


    args = parser.parse_args()

    return args


def create_images_from_ubyte(src, dest, dataset):
    """

        :param src:
        :param dest:
        :return:
        """
    try:
        import cv2
    except ImportError:
        return

    with open(src, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        image_data = array("B", file.read())

    images = np.zeros((size, rows, cols))

    for i in range(size):
        images[i, :, :] = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(rows, cols)
        cv2.imwrite(f'{dest}/{dataset}-{i}.jpg', images[i, :])

def create_images_from_pickle_py(src, dest, dataset):
    try:
        import cv2
    except ImportError:
        return
    with open(src, 'rb') as file:
        dict = pickle.load(file, encoding='latin1')
        images = dict['data'].reshape(-1, 3, 32, 32)

    for i in range(images.shape[0]):
        # reshape for cv write so that the channel is the last dim
        img = images[i].transpose(1, 2, 0)

        # array is RGB. cv2 needs BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'{dest}/{dataset}-{i}.jpg', img)


def create_dir_from_tensors(Tensors, dir_name="Validation-Gen-Images"):
    """

    :param Tensors: img_tensor to be saved
    :param dir_name: directory to save the images to
    :return: path to directory
    """
    # Create Dir if it does not exist
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)

    # unpack sensor?
    for img in Tensors:
        save_image(img, f'{dir_name}/gen-img{time.time():.20f}.png')

    return dir_name


def compute_FID(imgs, dataset, batch_size, device, dims):
    """
    TODO: this function only currently can work with the MNIST datasets
    :param imgs:
    :param batch_size:
    :param dataset:
    :param device:
    :param dims:
    :return:
    """
    fake_path = create_dir_from_tensors(imgs)

    dataset_src = DATASET_DIR[dataset]
    dataset_dest = os.path.join(MY_PATH, f"FID_TESTING/{dataset}")

    if not os.path.exists(dataset_dest):
        os.makedirs(dataset_dest)

        if dataset in UBYTE_DATASETS:
            create_images_from_ubyte(dataset_src, dataset_dest, dataset)
        elif dataset == "CIFAR10":
            create_images_from_pickle_py(dataset_src, dataset_dest, dataset)
        else:
            raise NotImplementedError('Unknown dataset')


    paths = [fake_path, dataset_dest]
    fid = fid_score.calculate_fid_given_paths(paths, batch_size, device, dims)

    return fid
if __name__ == "__main__":
    dataset = "CIFAR10"
    args = get_args(batch_size=32, dataset=dataset)
    train_loader, valid_loader, test_loader, img_shape = get_dataset(args)

    imgs = torch.randn(16, 1, 32, 32).type(torch.float32)

    fid = compute_FID(imgs, args, 'cuda', 64)
    print(fid)
    # args = get_args(dataset="CIFAR10", n_epoch=20, no_validation_images=100)
    #
    # # training
    # train, valid, test, img_shape = get_dataset(args)
    # dest = "FID_TESTING/TEST_CIFAR10"
    # create_images_from_pickle_py("Datasets/cifar-10-batches-py/data_batch_1", dest, "CIFAR10")