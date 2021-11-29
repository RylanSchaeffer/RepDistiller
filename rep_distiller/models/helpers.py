import argparse
import copy
import numpy as np
import torch
from typing import Dict, Tuple

from . import architecture_dict
from rep_distiller.models.readout import LinearReadout, MLPReadout


def create_finetune_model(model: torch.nn.Module,
                          dim_out: int,
                          linear_or_nonlinear_readout: str = 'linear',
                          train_only_readout: bool = True,
                          ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:

    assert linear_or_nonlinear_readout in {'linear', 'mlp'}

    # https://discuss.pytorch.org/t/can-i-deepcopy-a-model/52192
    model_copy = copy.deepcopy(model)

    # First create model to fine-tune
    if linear_or_nonlinear_readout == 'linear':
        finetune_model = LinearReadout(
            dim_in=model_copy.feat_dim,
            dim_out=dim_out,
            encoder=model_copy,
            train_only_readout=train_only_readout,
        )
    elif linear_or_nonlinear_readout == 'mlp':
        # finetune_model = MLPReadout(
        #     encoder=model,
        #     only_readout=only_readout,
        # )
        raise NotImplementedError
    else:
        raise ValueError

    if torch.cuda.is_available():
        finetune_model = finetune_model.cuda()

    if train_only_readout:
        params = finetune_model.readout.parameters()
    else:
        params = finetune_model.parameters()

    finetune_optimizer = torch.optim.SGD(
        params,
        lr=1e-3)

    return finetune_model, finetune_optimizer


def create_model_from_architecture_str(architecture_str: str,
                                       output_dim: int,
                                       num_samples: int,
                                       batch_size: int,
                                       dataset_str: str,
                                       gpus: int = 4,
                                       ) -> torch.nn.Module:

    if architecture_str == 'swav':

        from pl_bolts.models.self_supervised import SwAV

        model = SwAV(
            num_samples=num_samples,
            batch_size=batch_size,
            dataset=dataset_str,
            gpus=gpus,
            feat_dim=output_dim)

    elif architecture_str == 'simclr':

        from pl_bolts.models.self_supervised import SimCLR

        model = SimCLR(
            num_samples=num_samples,
            batch_size=batch_size,
            dataset=dataset_str,
            gpus=gpus,
            feat_dim=output_dim)
    else:
        model = architecture_dict[architecture_str](num_classes=output_dim)
        # Ensure model has a self.feat_dim property

        model.feat_dim = output_dim
    return model


def load_supervised_teacher(teacher_model_path: str,
                            num_classes: int,
                            ) -> torch.nn.Module:
    print('==> loading teacher model')
    teacher_name = get_teacher_name(teacher_model_path)
    model = architecture_dict[teacher_name](num_classes=num_classes)
    model.load_state_dict(torch.load(teacher_model_path)['model'])
    print('==> done')
    return model


def load_selfsupervised_pretrained_model(model_name: str,
                                         pretrain_dataset: str,
                                         ) -> torch.nn.Module:
    print('==> loading teacher model')

    if model_name == 'swav':
        # https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#swav
        from pl_bolts.models.self_supervised import SwAV
        from pl_bolts.models.self_supervised.swav.transforms import (SwAVTrainDataTransform, SwAVEvalDataTransform)
        if pretrain_dataset == 'imagenet':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
            train_transform = SwAVTrainDataTransform()
            eval_transform = SwAVEvalDataTransform()
            # eval_transform = SwAVTrainDataTransform()
        elif pretrain_dataset == 'stl10':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/checkpoints/swav_stl10.pth.tar'
            from pl_bolts.transforms.dataset_normalizations import stl10_normalization
            train_transform = SwAVTrainDataTransform(
                normalize=stl10_normalization())
            eval_transform = SwAVEvalDataTransform(
                normalize=stl10_normalization())
        else:
            raise NotImplementedError
        model = SwAV.load_from_checkpoint(weight_path, strict=True)
        # add feat_dim property for later readout
        model.feat_dim = 2048
    elif model_name == 'simclr':
        # https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr
        from pl_bolts.models.self_supervised import SimCLR
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        model = SimCLR.load_from_checkpoint(weight_path, strict=False)
        model.feat_dim = 2048
        from pl_bolts.models.self_supervised.simclr.transforms import (
            SimCLREvalDataTransform, SimCLRTrainDataTransform)
        # TODO: Why are the input heights 32s? 36 seems to be the norm for SwAV
        # train_transform = SimCLRTrainDataTransform(32)
        # eval_transform = SimCLREvalDataTransform(32)
        train_transform = SimCLRTrainDataTransform(input_height=36)
        eval_transform = SimCLREvalDataTransform(input_height=36)
    elif model_name == 'cpc_v2':
        from pl_bolts.models.self_supervised import CPC_v2
        if pretrain_dataset == 'cifar10':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-cifar10-v4-exp3/epoch%3D474.ckpt'
            from pl_bolts.models.self_supervised.cpc import (
                CPCTrainTransformsCIFAR10, CPCEvalTransformsCIFAR10)
            train_transform = CPCTrainTransformsCIFAR10()
            eval_transform = CPCEvalTransformsCIFAR10()
        elif pretrain_dataset == 'imagenet':
            # TODO: figure out CPC weight path
            weight_path = ''
            from pl_bolts.models.self_supervised.cpc import (
                CPCTrainTransformsImageNet128, CPCEvalTransformsImageNet128)
            train_transform = CPCTrainTransformsImageNet128()
            eval_transform = CPCEvalTransformsImageNet128()
            raise NotImplementedError
        elif pretrain_dataset == 'stl10':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/cpc/cpc-stl10-v0-exp3/epoch%3D624.ckpt'
            from pl_bolts.models.self_supervised.cpc import (
                CPCTrainTransformsSTL10, CPCEvalTransformsSTL10)
            train_transform = CPCTrainTransformsSTL10()
            eval_transform = CPCEvalTransformsSTL10()
        else:
            raise NotImplementedError
        model = CPC_v2.load_from_checkpoint(weight_path, strict=False)
    elif model_name == 'byol':
        raise NotImplementedError
    elif model_name == 'clip':
        import clip

        # Check available models
        # clip.available_models()
        # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_image_and_text_model, preprocess = clip.load("ViT-B/32", device=device)

        import pytorch_lightning as pl

        class CLIPImageEncoder(pl.LightningModule):

            def __init__(self,
                         clip_image_and_text_model):
                super().__init__()
                self.clip_image_and_text_model = clip_image_and_text_model
                self.feat_dim = clip_image_and_text_model.visual.output_dim

            def forward(self,
                        x: torch.Tensor,
                        ) -> torch.Tensor:
                x = self.clip_image_and_text_model.encode_image(x)
                return x

        model = CLIPImageEncoder(clip_image_and_text_model=clip_image_and_text_model)

        # image_features = model.encode_image(image)

        train_transform = preprocess
        eval_transform = preprocess
    else:
        raise NotImplementedError

    # PyTorch Lightning Modules have freeze
    if hasattr(model, 'freeze'):
        model.freeze()

    print('==> done')
    return model, train_transform, eval_transform


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]

