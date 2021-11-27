"""
the general training framework
"""

from __future__ import print_function

import os

# Ensure system GPU indices match PyTorch CUDA GPU indices
import rep_distiller.run.loops
import rep_distiller.run.helpers

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# Control GPU Access
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse
import socket

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import wandb

import rep_distiller.dataset.helpers
# import rep_distiller.run.hooks
import rep_distiller.losses
import rep_distiller.models.helpers


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--pretrain_epochs', type=int, default=240,
                        help='Number of epochs for pretraining')
    parser.add_argument('--num_pretrain_epochs_per_finetune', type=int, default=10,
                        help='Number of pretraining epochs per finetune run')
    parser.add_argument('--finetune_epochs', type=int, default=240,
                        help='Number of epochs for fine tuning')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--pretrain_dataset', type=str, default='imagenet', choices=['imagenet'], help='dataset')
    parser.add_argument('--finetune_dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--student_architecture', type=str, default='resnet8x4',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--teacher_name', type=str, default=None, help='Name of teacher model',
                        choices=['swav', 'simclr', 'byol', 'cpc_v2', 'clip'])

    # distillation
    parser.add_argument('--distill', type=str, default='prd', choices=['hint', 'attention', 'similarity',
                                                                       'correlation', 'vid', 'crd',
                                                                       'prd', 'kdsvd', 'fsp',
                                                                       'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('--custom_weight', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=512, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # CRD flags
    parser.add_argument('--crd_negative_sampling', default='different_class',
                        type=str, choices=['random', 'different_class'],
                        help='How to sample negative')
    parser.add_argument('--crd_normalize', dest='crd_normalize', default=True,
                        action='store_true', help='Whether to normalize CRD representations')
    parser.add_argument('--crd_dont_normalize', dest='crd_normalize', default=False,
                        action='store_false')

    # pretrained representation distillation
    parser.add_argument('--prd_primal_or_dual', default='primal', type=str, help='Whether to use Primal or Dual')
    parser.add_argument('--prd_c', default=1e-1, type=float, help='Ridge regression weight')
    parser.add_argument('--prd_normalize', dest='prd_normalize', default=False,
                        action='store_true', help='Whether to normalize KRD representations')
    parser.add_argument('--prd_dont_normalize', dest='prd_normalize', default=True,
                        action='store_false', help='Whether to normalize KRD representations')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.student_architecture in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.exp_path = '/data3/rschaef/pretrained_representation_distillation/save/'
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = os.path.join(opt.exp_path, 'selfsupervised_student_model')
        opt.tb_path = os.path.join(opt.exp_path, 'selfsupervised_student_tensorboards')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'S:{}_T:{}_{}_{}_{}_b:{}_{}'.format(
        opt.student_architecture, opt.teacher_name, opt.pretrain_dataset, opt.finetune_dataset, opt.distill,
        opt.custom_weight, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():

    opt = parse_option()

    # wandb.init(project='pretrained_representation_distillation',
    #            config=opt)

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # model
    model_t, train_transform, eval_transform = rep_distiller.models.helpers.load_selfsupervised_teacher(
        teacher_name=opt.teacher_name,
        pretrain_dataset=opt.pretrain_dataset)
    model_s = rep_distiller.models.helpers.create_model_from_architecture_str(
        architecture_str=opt.student_architecture,
        output_dim=opt.feat_dim)

    # dataloader
    pretrain_train_loader, pretrain_eval_loader, pretrain_n_cls = \
        rep_distiller.dataset.helpers.load_dataloaders(
            dataset=opt.pretrain_dataset,
            opt=opt,
            train_transform=train_transform,
            eval_transform=eval_transform)

    finetune_train_loader, finetune_eval_loader, finetune_n_cls = \
        rep_distiller.dataset.helpers.load_dataloaders(
            dataset=opt.finetune_dataset,
            opt=opt)

    models_dict = nn.ModuleDict({
        'student': model_s,
        'teacher': model_t})
    trainable_dict = nn.ModuleDict({
        'student': model_s,
    })

    criteria_dict = rep_distiller.losses.create_criteria_dict(
        opt=opt)

    # optimizer
    optimizer = optim.SGD(trainable_dict.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    if torch.cuda.is_available():
        models_dict.cuda()
        criteria_dict.cuda()
        cudnn.benchmark = True

    # fn_hook_dict = rep_distiller.run.hooks.create_hook_fns_train(
    #     start_grad_step=0,
    #     num_grad_steps=opt.
    # )
    #
    # rep_distiller.run.loops.run_hooks(
    #     fn_hook_dict=fn_hook_dict,
    #     models_dict=models_dict,
    #     pretrain_train_loader=pretrain_train_loader,
    #     pretrain_eval_loader=pretrain_eval_loader,
    #     pretrain_epochs=opt.pretrain_epochs,
    #     finetune_train_loader=finetune_train_loader,
    #     finetune_eval_loader=finetune_eval_loader,
    #     finetune_epochs=opt.finetune_epochs,
    #     criteria_dict=criteria_dict,
    #     optimizer=optimizer,
    #     opt=opt,
    # )

    rep_distiller.run.loops.pretrain_and_finetune(
        models_dict=models_dict,
        pretrain_train_loader=pretrain_train_loader,
        pretrain_eval_loader=pretrain_eval_loader,
        pretrain_epochs=opt.pretrain_epochs,
        finetune_train_loader=finetune_train_loader,
        finetune_eval_loader=finetune_eval_loader,
        finetune_epochs=opt.finetune_epochs,
        losses_callables_dict=criteria_dict,
        optimizer=optimizer,
        opt=opt,
        logger=logger)


if __name__ == '__main__':
    main()
