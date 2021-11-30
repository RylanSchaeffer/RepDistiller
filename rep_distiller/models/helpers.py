from __future__ import print_function


import argparse
import copy
import math
import numpy as np
import torch
import torch.nn as nn

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


class Paraphraser(nn.Module):
    """Paraphrasing Complex Network: Network Compression via Factor Transfer"""
    def __init__(self, t_shape, k=0.5, use_bn=False):
        super(Paraphraser, self).__init__()
        in_channel = t_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(out_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s, is_factor=False):
        factor = self.encoder(f_s)
        if is_factor:
            return factor
        rec = self.decoder(factor)
        return factor, rec


class Translator(nn.Module):
    def __init__(self, s_shape, t_shape, k=0.5, use_bn=True):
        super(Translator, self).__init__()
        in_channel = s_shape[1]
        out_channel = int(t_shape[1] * k)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.BatchNorm2d(in_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1),
            nn.BatchNorm2d(out_channel) if use_bn else nn.Sequential(),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, f_s):
        return self.encoder(f_s)


class Connector(nn.Module):
    """Connect for Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons"""
    def __init__(self, s_shapes, t_shapes):
        super(Connector, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    @staticmethod
    def _make_conenctors(s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        connectors = []
        for s, t in zip(s_shapes, t_shapes):
            if s[1] == t[1] and s[2] == t[2]:
                connectors.append(nn.Sequential())
            else:
                connectors.append(ConvReg(s, t, use_relu=False))
        return connectors

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConnectorV2(nn.Module):
    """A Comprehensive Overhaul of Feature Distillation (ICCV 2019)"""
    def __init__(self, s_shapes, t_shapes):
        super(ConnectorV2, self).__init__()
        self.s_shapes = s_shapes
        self.t_shapes = t_shapes

        self.connectors = nn.ModuleList(self._make_conenctors(s_shapes, t_shapes))

    def _make_conenctors(self, s_shapes, t_shapes):
        assert len(s_shapes) == len(t_shapes), 'unequal length of feat list'
        t_channels = [t[1] for t in t_shapes]
        s_channels = [s[1] for s in s_shapes]
        connectors = nn.ModuleList([self._build_feature_connector(t, s)
                                    for t, s in zip(t_channels, s_channels)])
        return connectors

    @staticmethod
    def _build_feature_connector(t_channel, s_channel):
        C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
             nn.BatchNorm2d(t_channel)]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class ConvReg(nn.Module):
    """Convolutional regression for FitNet"""
    def __init__(self, s_shape, t_shape, use_relu=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1+s_H-t_H, 1+s_W-t_W))
        else:
            raise NotImplemented('student size {}, teacher size {}'.format(s_H, t_H))
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)


class Regress(nn.Module):
    """Simple Linear Regression for hints"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Regress, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class LinearEmbed(nn.Module):
    """Linear Embedding"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(LinearEmbed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, 2 * dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.l2norm(self.linear2(x))
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Flatten(nn.Module):
    """flatten module"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)


class PoolEmbed(nn.Module):
    """pool and embed"""
    def __init__(self, layer=0, dim_out=128, pool_type='avg'):
        super().__init__()
        if layer == 0:
            pool_size = 8
            nChannels = 16
        elif layer == 1:
            pool_size = 8
            nChannels = 16
        elif layer == 2:
            pool_size = 6
            nChannels = 32
        elif layer == 3:
            pool_size = 4
            nChannels = 64
        elif layer == 4:
            pool_size = 1
            nChannels = 64
        else:
            raise NotImplementedError('layer not supported: {}'.format(layer))

        self.embed = nn.Sequential()
        if layer <= 3:
            if pool_type == 'max':
                self.embed.add_module('MaxPool', nn.AdaptiveMaxPool2d((pool_size, pool_size)))
            elif pool_type == 'avg':
                self.embed.add_module('AvgPool', nn.AdaptiveAvgPool2d((pool_size, pool_size)))

        self.embed.add_module('Flatten', Flatten())
        self.embed.add_module('Linear', nn.Linear(nChannels*pool_size*pool_size, dim_out))
        self.embed.add_module('Normalize', Normalize(2))

    def forward(self, x):
        return self.embed(x)


if __name__ == '__main__':
    import torch

    g_s = [
        torch.randn(2, 16, 16, 16),
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 64, 4, 4),
    ]
    g_t = [
        torch.randn(2, 32, 16, 16),
        torch.randn(2, 64, 8, 8),
        torch.randn(2, 128, 4, 4),
    ]
    s_shapes = [s.shape for s in g_s]
    t_shapes = [t.shape for t in g_t]

    net = ConnectorV2(s_shapes, t_shapes)
    out = net(g_s)
    for f in out:
        print(f.shape)
