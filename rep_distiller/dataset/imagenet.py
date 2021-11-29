"""
get data loaders
"""
from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms
from typing import Tuple


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    # hostname = socket.gethostname()
    # if hostname.startswith('visiongpu'):
    #     data_folder = '/data/vision/phillipi/rep-learn/datasets/imagenet'
    # elif hostname.startswith('yonglong-home'):
    #     data_folder = '/home/yonglong/Data/data/imagenet'
    # else:
    data_folder = '/data5/chengxuz/imagenet_raw/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


class ImageFolderInstance(datasets.ImageFolder):
    """: Folder datasets which returns the index of the image as well::
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class ImageFolderSample(datasets.ImageFolder):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """

    def __init__(self, root, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(root=root, transform=transform, target_transform=target_transform)

        self.k = k
        self.is_sample = is_sample

        print('stage1 finished!')

        if self.is_sample:
            num_classes = len(self.classes)
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                path, target = self.imgs[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]

        print('dataset initialized!')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index


# def get_test_loader(dataset='imagenet', batch_size=128, num_workers=8):
#     """get the test data loader"""
#
#     if dataset == 'imagenet':
#         data_folder = get_data_folder()
#     else:
#         raise NotImplementedError('dataset not supported: {}'.format(dataset))
#
#     normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                  std=[0.229, 0.224, 0.225])
#     test_transform = torchvision.transforms.Compose([
#         torchvision.transforms.Resize(256),
#         torchvision.transforms.CenterCrop(224),
#         torchvision.transforms.ToTensor(),
#         normalize,
#     ])
#
#     test_folder = os.path.join(data_folder, 'val')
#     test_set = datasets.ImageFolder(test_folder, transform=test_transform)
#     test_loader = DataLoader(test_set,
#                              batch_size=batch_size,
#                              shuffle=False,
#                              num_workers=num_workers,
#                              pin_memory=True)
#
#     return test_loader


def get_imagenet_dataloaders_sample(dataset='imagenet', batch_size=128, num_workers=8, is_sample=False, k=4096):
    # TODO: deduplicate with get_imagenet_dataloaders
    """Data Loader for ImageNet"""

    if dataset == 'imagenet':
        data_folder = get_data_folder()
    else:
        raise NotImplementedError('dataset not supported: {}'.format(dataset))

    # add data transform
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ])
    train_folder = os.path.join(data_folder, 'train')
    test_folder = os.path.join(data_folder, 'val')

    train_set = ImageFolderSample(train_folder, transform=train_transform, is_sample=is_sample, k=k)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    print('num_samples', len(train_set.samples))
    print('num_class', len(train_set.classes))

    return train_loader, test_loader, len(train_set), len(train_set.classes)


def get_imagenet_dataloaders(dataset='imagenet',
                             batch_size=128,
                             num_workers=16,
                             is_instance=False,
                             train_transform=None,
                             eval_transform=None,
                             ) -> [DataLoader, DataLoader, int]:
    """
    Data Loader for imagenet
    """
    assert dataset == 'imagenet'
    data_folder = get_data_folder()

    normalize = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    if train_transform is None:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    if eval_transform is None:
        eval_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

    train_folder = os.path.join(data_folder, 'train')
    # if is_instance:
    #     train_set = ImageFolderInstance(train_folder, transform=train_transform)
    #     n_data = len(train_set)
    # else:
    train_set = datasets.ImageFolder(train_folder, transform=train_transform)

    # Raises error: RuntimeError: The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted. You need to download it externally and place it in /data5/chengxuz/imagenet_raw/.
    # train_set = datasets.ImageNet(root=data_folder,
    #                               train=True,
    #                               transform=train_transform)

    eval_folder = os.path.join(data_folder, 'validation')
    eval_set = datasets.ImageFolder(eval_folder, transform=eval_transform)

    # Raises RuntimeError: The archive ILSVRC2012_devkit_t12.tar.gz is not present in the root directory or is corrupted. You need to download it externally and place it in /data5/chengxuz/imagenet_raw/.
    # test_set = datasets.ImageNet(root=data_folder,
    #                              train=False,
    #                              transform=eval_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              # pin_memory=True,
                              )

    eval_loader = DataLoader(eval_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers // 2,
                             pin_memory=True)

    if is_instance:
        return train_loader, eval_loader, len(train_set)
    else:
        return train_loader, eval_loader, None
