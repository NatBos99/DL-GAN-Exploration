import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, CelebA
from torchvision import transforms
import logging


def get_dataset(args):
    logging.info('Loading {} dataset'.format(args.dataset))
    if args.dataset == "CIFAR10":
        train = CIFAR10(root="Datasets/cifar-10-batches-py",
                        download=True,
                        transform=transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()]))
        test = CIFAR10(root="Datasets/cifar-10-batches-py",
                       download=True,
                       train=False,
                       transform=transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()]))

    elif args.dataset == "MNIST":
        new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
        MNIST.resources = [
            ('/'.join([new_mirror, url.split('/')[-1]]), md5)
            for url, md5 in MNIST.resources
        ]
        train = MNIST(root="Datasets/MNIST",
                      download=True,
                      transform=transforms.Compose(
                          [transforms.Resize(args.image_size), transforms.ToTensor(),
                           transforms.Normalize([0.5], [0.5])]),
                      )
        test = MNIST(root="Datasets/MNIST",
                     download=True,
                     train=False,
                     transform=transforms.Compose(
                         [transforms.Resize(args.image_size), transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])]),
                     )
    elif args.dataset == "FashionMNIST":
        train = FashionMNIST(root="Datasets/FashionMNIST",
                        download=True,
                        transform=transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()]))
        test = FashionMNIST(root="Datasets/FashionMNIST",
                       download=True,
                       train=False,
                       transform=transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()]))

    elif args.dataset == "CelebA":
        train = CelebA(root="Datasets/CelebA",
                        download=True,
                        transform=transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()]))
        test = CelebA(root="Datasets/CelebA",
                       download=True,
                       train=False,
                       transform=transforms.Compose([transforms.Resize(args.image_size), transforms.ToTensor()]))

    else:
        raise NotImplementedError('Unknown dataset')

    img, _ = train[1]  # take second image
    img_shape = img.size()
    data_size = len(train)
    train_size = int(args.train_valid_split * data_size)
    train, validation = random_split(train, [train_size, data_size - train_size], generator=torch.Generator().manual_seed(41))

    train_loader = DataLoader(train,
                              batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              )
    valid_loader = DataLoader(validation,
                              batch_size=args.batch_size,
                              shuffle=args.shuffle,
                              )
    test_loader = DataLoader(test,
                             batch_size=args.batch_size,
                             shuffle=args.shuffle,
                             )

    return train_loader, valid_loader, test_loader, tuple(img_shape)
