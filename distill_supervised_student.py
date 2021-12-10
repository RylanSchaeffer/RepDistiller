"""
the general training framework
"""

from __future__ import print_function

import os

# Ensure system GPU indices match PyTorch CUDA GPU indices
import rep_distiller.models.helpers

os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# Control GPU Access
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import argparse
import socket
import time
import wandb

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from rep_distiller.models import architecture_dict
from rep_distiller.models.helpers import ConvReg, LinearEmbed
from rep_distiller.models.helpers import Connector, Translator, Paraphraser

from rep_distiller.dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample

from rep_distiller.run.helpers import adjust_learning_rate

from rep_distiller.distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, \
    PretrainedRepresentationDistillation, VIDLoss, RKDLoss
from rep_distiller.distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss
from rep_distiller.crd.criterion import CRDLoss

from rep_distiller.run.loops_old import train_epoch_distill, validate, init


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=100, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity',
                                                                      'correlation', 'vid', 'crd',
                                                                      'prd', 'kdsvd', 'fsp',
                                                                      'rkd', 'pkt', 'abound', 'factor', 'nst'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('--classification_weight', type=float, default=1., help='weight for classification')
    parser.add_argument('--kl_div_weight', type=float, default=None, help='weight for KL Divergence')
    parser.add_argument('--custom_weight', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
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
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.exp_path = '/data3/rschaef/pretrained_representation_distillation/save/'
    if hostname.startswith('visiongpu'):
        opt.model_path = '/path/to/my/student_model'
        opt.tb_path = '/path/to/my/student_tensorboards'
    else:
        opt.model_path = os.path.join(opt.exp_path, 'student_model')
        opt.tb_path = os.path.join(opt.exp_path, 'student_tensorboards')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = rep_distiller.models.helpers.get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(
        opt.model_s, opt.model_t, opt.dataset, opt.distill,
        opt.classification_weight, opt.kl_div_weight, opt.custom_weight, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():
    best_acc = 0

    opt = parse_option()

    wandb.init(project='pretrained_representation_distillation',
               config=opt)

    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                k=opt.nce_k,
                mode=opt.mode,
                negative_sampling=opt.crd_negative_sampling)
        else:
            train_loader, val_loader, n_data = get_cifar100_dataloaders(
                batch_size=opt.batch_size,
                num_workers=opt.num_workers,
                is_instance=True)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = rep_distiller.models.helpers.load_supervised_teacher(opt.path_t, n_cls)
    model_s = architecture_dict[opt.model_s](num_classes=n_cls)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    models_dict = nn.ModuleDict({'student': model_s})
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        models_dict.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = n_data
        criterion_kd = CRDLoss(
            student_dim=opt.s_dim,
            teacher_dim=opt.t_dim,
            normalize=opt.crd_normalize,
            projection_dim=opt.feat_dim,
            num_data=opt.n_data,
            num_neg_examples_per_pos_example=opt.nce_k,
            softmax_temp=opt.nce_t,
            momentum=opt.nce_m)
        models_dict.append(criterion_kd.embed_s)
        models_dict.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'prd':
        criterion_kd = PretrainedRepresentationDistillation(
            primal_or_dual=opt.prd_primal_or_dual,
            ridge_prefactor=opt.prd_c,
            normalize=opt.prd_normalize)
        init_trainable_modules = nn.ModuleList([model_s])
        init(model_s=model_s,
             model_t=model_t,
             trainable_modules=init_trainable_modules,
             criterion=criterion_kd,
             loader=train_loader,
             logger=logger,
             lr=1e-3,
             opt=opt)
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        models_dict.append(embed_s)
        models_dict.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_modules = nn.ModuleList([])
        init_trainable_modules.append(connector)
        init_trainable_modules.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_modules, criterion_kd, train_loader, logger, opt)
        # classification
        models_dict.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_modules = nn.ModuleList([])
        init_trainable_modules.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_modules, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        models_dict.append(translator)
        models_dict.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_modules = nn.ModuleList([])
        init_trainable_modules.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_modules, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)  # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    models_dict['teacher'] = model_t

    if torch.cuda.is_available():
        models_dict.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc1, teacher_acc5, _ = validate(
        val_loader=val_loader,
        model=model_t,
        model_name='teacher',
        criterion=criterion_cls,
        opt=opt)
    print(f'Teacher:\tTop1: {teacher_acc1}\tTop5: {teacher_acc5}')

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, train_classification_loss, train_kd_loss, train_custom_loss = \
            train_epoch_distill(
                epoch=epoch,
                train_loader=train_loader,
                models_dict=models_dict,
                criterion_list=criterion_list,
                optimizer=optimizer,
                opt=opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(
            val_loader=val_loader,
            model=model_s,
            model_name='student',
            criterion=criterion_cls,
            opt=opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)

        wandb.log({
            'train_acc': train_acc,
            'train_loss': train_loss,
            'train_classification_loss': train_classification_loss,
            'train_kl_div_loss': train_kd_loss,
            'train_custom_loss': train_custom_loss,
            'test_acc': test_acc,
            'test_acc_top5': test_acc_top5,
            'test_loss': test_loss,
        },
            step=epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
