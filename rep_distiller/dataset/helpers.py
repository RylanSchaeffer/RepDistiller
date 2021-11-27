import argparse
from torch.utils.data.dataloader import DataLoader
from typing import Tuple

from rep_distiller.dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from rep_distiller.dataset.imagenet import get_imagenet_dataloaders


def load_dataloaders(dataset: str,
                     opt: argparse.Namespace,
                     train_transform=None,
                     eval_transform=None,
                     ) -> Tuple[DataLoader, DataLoader, int]:

    if dataset == 'cifar100':
        if opt.distill in {'crd'}:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                k=opt.nce_k,
                mode=opt.mode,
                negative_sampling=opt.crd_negative_sampling,
                train_transform=train_transform,
                eval_transform=eval_transform)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                is_instance=False,
                train_transform=train_transform,
                eval_transform=eval_transform)
    elif dataset == 'imagenet':
        if opt.distill in {'crd'}:
            # train_loader, val_loader = get_imagenet_dataloaders_sample
            raise NotImplementedError
        else:
            train_loader, val_loader, _ = get_imagenet_dataloaders(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                is_instance=False,
                train_transform=train_transform,
                eval_transform=eval_transform)
    elif dataset == 'coco':
        raise NotImplementedError
    else:
        raise NotImplementedError(opt.dataset)

    num_classes = len(set(train_loader.dataset.classes).union(
        set(val_loader.dataset.classes)))

    return train_loader, val_loader, num_classes
