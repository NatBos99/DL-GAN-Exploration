import logging

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, CelebA


def data_augmentation():
    trans = [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip()]
    return trans


def get_dataset(args):
    logging.info('Loading {} dataset'.format(args.dataset))

    data_aug_trans = data_augmentation() if args.data_augmentation else []

    if args.dataset == "CIFAR10":
        train = CIFAR10(root="Datasets/",
                        download=True,
                        transform=transforms.Compose(
                            [transforms.Resize(args.image_size), *data_aug_trans, *data_aug_trans,
                             transforms.ToTensor()]))
        test = CIFAR10(root="Datasets/",
                       download=True,
                       train=False,
                       transform=transforms.Compose(
                           [transforms.Resize(args.image_size), *data_aug_trans, *data_aug_trans,
                            transforms.ToTensor()]))

    elif args.dataset == "MNIST":
        if args.custom_mnist_download:
            new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
            MNIST.resources = [
                ('/'.join([new_mirror, url.split('/')[-1]]), md5)
                for url, md5 in MNIST.resources
            ]
        train = MNIST(root="Datasets/MNIST",
                      download=True,
                      transform=transforms.Compose(
                          [transforms.Resize(args.image_size), *data_aug_trans, *data_aug_trans, transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5])]),
                      )
        test = MNIST(root="Datasets/",
                     download=True,
                     train=False,
                     transform=transforms.Compose(
                         [transforms.Resize(args.image_size), *data_aug_trans, *data_aug_trans, transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])]),
                     )

    elif args.dataset == "MNIST_128":
        if args.custom_mnist_download:
            new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
            MNIST.resources = [
                ('/'.join([new_mirror, url.split('/')[-1]]), md5)
                for url, md5 in MNIST.resources
            ]
        train = MNIST(root="Datasets/",
                      download=True,
                      transform=transforms.Compose(
                          [transforms.Resize(args.image_size), *data_aug_trans, *data_aug_trans, transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5])])
                      )
        train.data = train.data[:128]
        train.targets = train.targets[:128]
        test = MNIST(root="Datasets/MNIST",
                     download=True,
                     train=False,
                     transform=transforms.Compose(
                         [transforms.Resize(args.image_size), *data_aug_trans, *data_aug_trans, transforms.Grayscale(3),
                          transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])])
                     )

    elif args.dataset == "FashionMNIST":
        train = FashionMNIST(root="Datasets/",
                             download=True,
                             transform=transforms.Compose(
                                 [transforms.Resize(args.image_size), *data_aug_trans, transforms.ToTensor()]))
        test = FashionMNIST(root="Datasets/",
                            download=True,
                            train=False,
                            transform=transforms.Compose(
                                [transforms.Resize(args.image_size), *data_aug_trans, transforms.ToTensor()]))

    elif args.dataset == "CelebA":
        train = CelebA(root="Datasets/",
                       download=True,
                       transform=transforms.Compose(
                           [transforms.Resize(args.image_size), *data_aug_trans, transforms.ToTensor()]))
        test = CelebA(root="Datasets/",
                      download=True,
                      train=False,
                      transform=transforms.Compose(
                          [transforms.Resize(args.image_size), *data_aug_trans, transforms.ToTensor()]))

    else:
        raise NotImplementedError('Unknown dataset')

    img, _ = train[1]  # take second image
    img_shape = img.size()
    data_size = len(train)
    train_size = int(args.train_valid_split * data_size)
    train, validation = random_split(train, [train_size, data_size - train_size],
                                     generator=torch.Generator().manual_seed(41))

    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              num_workers=args.num_workers
                              )
    valid_loader = DataLoader(validation,
                              batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              num_workers=args.num_workers
                              )

    test_loader = DataLoader(test,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             num_workers=args.num_workers
                             )

    return train_loader, valid_loader, test_loader, tuple(img_shape)
