from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image

"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    hostname = socket.gethostname()
    if hostname.startswith('visiongpu'):
        data_folder = '/data/vision/phillipi/rep-learn/datasets'
        raise NotImplementedError
    elif hostname.startswith('yonglong-home'):
        data_folder = '/home/yonglong/Data/data'
        raise NotImplementedError
    else:
        data_folder = '/data3/rschaef/datasets/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class CIFAR100Instance(datasets.CIFAR100):
    """CIFAR100Instance Dataset.
    """

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
            # img, target = self.train_data[index], self.train_labels[index]
        else:
            # img, target = self.test_data[index], self.test_labels[index]
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def get_cifar10_dataloaders(batch_size: int = 128,
                            num_workers=8,
                            is_instance=False,
                            train_transform=None,
                            eval_transform=None
                            ):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if eval_transform is None:
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    train_set = datasets.CIFAR10(root=data_folder,
                                 download=True,
                                 train=True,
                                 transform=train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR10(root=data_folder,
                                download=True,
                                train=False,
                                transform=eval_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader, n_data


def get_cifar100_dataloaders(batch_size: int = 128,
                             num_workers=8,
                             is_instance=False,
                             train_transform=None,
                             eval_transform=None
                             ):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if eval_transform is None:
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if is_instance:
        train_set = CIFAR100Instance(root=data_folder,
                                     download=True,
                                     train=True,
                                     transform=train_transform)
    else:
        train_set = datasets.CIFAR100(root=data_folder,
                                      download=True,
                                      train=True,
                                      transform=train_transform)
    n_data = len(train_set)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=eval_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader, n_data


class CIFAR100InstanceSample(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 k=4096,
                 mode='exact',
                 negative_sampling='different_class',
                 is_sample=True,
                 percent=1.0):

        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.is_sample = is_sample

        num_classes = 100
        if self.train:
            # num_samples = len(self.train_data)
            num_samples = len(self.data)
            # label = self.train_labels
            label = self.targets
        else:
            # num_samples = len(self.test_data)
            num_samples = len(self.data)
            # label = self.test_labels
            label = self.targets

        assert negative_sampling in {'different_class', 'random'}
        self.negative_sampling = negative_sampling

        # Will become 2D array of shape = (number of classes, num of data in class)
        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        # Will become 2D array of shape = (number of classes, num of data not in class)
        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
        self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
        self.possible_sample_indices = list(range(num_samples))

        if 0 < percent < 1:
            n = int(len(self.cls_negative[0]) * percent)
            self.cls_negative = [np.random.permutation(self.cls_negative[i])[0:n]
                                 for i in range(num_classes)]

    def __getitem__(self, index):
        if self.train:
            # img, target = self.train_data[index], self.train_labels[index]
            img, target = self.data[index], self.targets[index]
        else:
            # img, target = self.test_data[index], self.test_labels[index]
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.is_sample:
            # directly return
            return img, target, index
        else:
            # sample contrastive examples
            if self.mode == 'exact':
                pos_idx = index
            elif self.mode == 'relax':
                pos_idx = np.random.choice(self.cls_positive[target], 1)
                pos_idx = pos_idx[0]
            else:
                raise NotImplementedError(self.mode)
            replace = True if self.k > len(self.cls_negative[target]) else False
            if self.negative_sampling == 'different_class':
                neg_idx = np.random.choice(
                    self.cls_negative[target],
                    self.k,
                    replace=replace)
            elif self.negative_sampling == 'random':
                neg_idx = np.random.choice(
                    self.possible_sample_indices[:index] + self.possible_sample_indices[index + 1:],
                    self.k,
                    replace=replace)
            else:
                raise ValueError(f'Impermissible negative sampling: {self.negative_sampling}')
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx


def get_cifar100_dataloaders_sample(batch_size=128,
                                    num_workers=8,
                                    k=4096,
                                    mode='exact',
                                    negative_sampling='different_class',
                                    is_sample=True,
                                    percent=1.0,
                                    train_transform=None,
                                    eval_transform=None):
    """
    cifar 100
    """
    data_folder = get_data_folder()

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_set = CIFAR100InstanceSample(root=data_folder,
                                       download=True,
                                       train=True,
                                       transform=train_transform,
                                       k=k,
                                       mode=mode,
                                       negative_sampling=negative_sampling,
                                       is_sample=is_sample,
                                       percent=percent)
    n_data = len(train_set)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=int(batch_size / 2),
                             shuffle=False,
                             num_workers=int(num_workers / 2))

    return train_loader, test_loader, n_data
