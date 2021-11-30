import os

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# Control GPU Access
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import seaborn as sns
import torch
import torchvision

from rep_distiller.dataset.helpers import get_cifar100_dataloaders
from rep_distiller.models.helpers import load_selfsupervised_pretrained_model


torch.set_default_dtype(torch.float32)

pretrained_model, train_transform, eval_transform = load_selfsupervised_pretrained_model(
    model_name='simclr',
    pretrain_dataset='imagenet')

train_dataloader, eval_dataloader, _ = get_cifar100_dataloaders(
    batch_size=100,
    train_transform=train_transform,
    eval_transform=eval_transform)


output_tensors = []
target_tensors = []
for batch_idx, (input_tensor, target_tensor) in enumerate(eval_dataloader):

    # TODO: Figure out what to do about SwAV-like transforms
    # For different pretrained models (e.g. SwAV), the transforms map
    # each sample into the batch into a list of many (e.g. 7) tensors.
    # Here, we stack them and treat them as one big batch.
    if isinstance(input_tensor, list):
        # For now, use hack of taking just the first
        input_tensor = input_tensor[0]
    output_tensor = pretrained_model(input_tensor)
    output_tensors.append(output_tensor)
    target_tensors.append(target_tensor)
    print(f'Batch idx: {batch_idx}')
    if batch_idx > 120:
        break

output_tensors = torch.cat(output_tensors, dim=0)
target_tensors = torch.cat(target_tensors, dim=0)
reorder_indices = torch.argsort(target_tensors)
output_tensors = output_tensors[reorder_indices]
target_tensors = target_tensors[reorder_indices]


num_data, data_dim = output_tensors.shape
cs = np.logspace(-3, -1, 3)
num_cols = len(cs)
subset_sizes = [10000, 5000, 2000, 1000, 250]
# subset_sizes = [100, 500]
num_rows = len(subset_sizes)
cutoff = 1e-8
possible_indices = np.arange(len(output_tensors))

fig, axes = plt.subplots(
    nrows=num_rows,
    ncols=num_cols,
    figsize=(5 * num_cols, 5 * num_rows))
for r_idx, subset_size in enumerate(subset_sizes):

    subset_indices = np.random.choice(
        possible_indices,
        size=subset_size,
        replace=False)

    reorder_indices = torch.argsort(target_tensors[subset_indices])
    F = output_tensors[subset_indices][reorder_indices]

    for c_idx, c in enumerate(cs):

        H = F @ torch.linalg.inv(F.T @ F + c * torch.eye(data_dim)) @ F.T
        H = H.numpy()
        H[H < cutoff] = np.nan
        # np.percentile(H.reshape(-1), [10., 20., 30., ])
        ax = axes[r_idx, c_idx]
        im = ax.imshow(
            H,
            cmap='jet',
            norm=LogNorm(),
            vmin=cutoff,
            vmax=2.,
            aspect='equal',
            interpolation='none')
        # plt.show()
        print(f'Row: {r_idx}\tCol: {c_idx}')
        # sns.heatmap(
        #     H,
        #     mask=np.isnan(H),
        #     # ax=ax,
        #     square=True,
        #     vmax=1.,
        #     vmin=cutoff,
        #     norm=LogNorm(),
        #     cmap="jet",
        #     xticklabels=False,
        #     yticklabels=False)

        ax.set_title(f'c={c}')
        if c_idx == 0:
            ax.set_ylabel(f'subset size={subset_size}')

# plt.colorbar()
# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.savefig(f'00_pretrained_hat_matrix.png', dpi=300)
plt.show()

print(10)
