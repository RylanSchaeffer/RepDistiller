from __future__ import print_function, division

import argparse
import copy
import numpy as np
import os
import sys
import time
import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Dict, List, Tuple
import wandb

import rep_distiller.models.readout
import rep_distiller.run.util


# def eval_epoch(models_dict: torch.nn.ModuleDict,
#                eval_loader: DataLoader,
#                criterion: torch.nn.ModuleList,
#                opt: argparse.Namespace):
#     """Eval one epoch."""
#     batch_time_by_model = {model_name: rep_distiller.run.util.AverageMeter()
#                            for model_name in models_dict}
#     total_losses_by_model = {model_name: rep_distiller.run.util.AverageMeter()
#                              for model_name in models_dict}
#     top1_acc_by_model = {model_name: rep_distiller.run.util.AverageMeter()
#                          for model_name in models_dict}
#     top5_acc_by_model = {model_name: rep_distiller.run.util.AverageMeter()
#                          for model_name in models_dict}
#
#     # switch to evaluate mode
#     models_dict.eval()
#
#     with torch.no_grad():
#         end = time.time()
#         for batch_idx, (input, target) in enumerate(eval_loader):
#
#             input = input.float()
#             if torch.cuda.is_available():
#                 input = input.cuda()
#                 target = target.cuda()
#
#             for model_name, model in models_dict:
#                 # compute output
#                 output = model(input)
#                 loss = criterion(output, target)
#
#                 # measure accuracy and record loss
#                 acc1, acc5 = rep_distiller.run.util.accuracy(
#                     output,
#                     target,
#                     topk=(1, 5))
#
#                 total_losses_by_model[model_name].update(loss.item(), input.size(0))
#                 top1_acc_by_model[model_name].update(acc1[0], input.size(0))
#                 top5_acc_by_model[model_name].update(acc5[0], input.size(0))
#
#             # measure elapsed time
#             batch_time_by_model[model_name].update(time.time() - end)
#             end = time.time()
#
#             if batch_idx % opt.print_freq == 0:
#                 print('Test: [{0}/{1}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     batch_idx, len(eval_loader), batch_time=batch_time_by_model, loss=total_losses_by_model,
#                     top1=top1_acc_by_model, top5=top5_acc_by_model))
#
#         print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
#               .format(top1=top1_acc_by_model, top5=top5_acc_by_model))
#
#     avg_top1_acc_by_model = {model_name: top1_acc_by_model[model_name].avg
#                              for model_name in models_dict}
#     avg_top5_acc_by_model = {model_name: top5_acc_by_model[model_name].avg
#                              for model_name in models_dict}
#     avg_total_loss_by_model = {model_name: total_losses_by_model[model_name].avg
#                                for model_name in models_dict}
#
#     return avg_top1_acc_by_model, avg_top5_acc_by_model, avg_total_loss_by_model
#

def finetune(models_dict: torch.nn.ModuleDict,
             num_epochs: int,
             finetune_train_loader: DataLoader,
             finetune_eval_loader: DataLoader,
             finetune_criteria_list: torch.nn.ModuleList,
             opt: argparse.Namespace,
             finetune_linear_or_nonlinear: str = 'linear',
             only_readout: bool = True,
             **kwargs,
             ):
    assert finetune_linear_or_nonlinear in {'linear', 'mlp'}

    finetune_models_dicts = dict()
    finetune_optimizers = dict()
    for model_name, model in models_dict.items():

        # https://discuss.pytorch.org/t/can-i-deepcopy-a-model/52192
        model_copy = copy.deepcopy(model)

        # First create model to fine-tune
        if finetune_linear_or_nonlinear == 'linear':
            finetune_model = rep_distiller.models.readout.LinearReadout(
                dim_in=model_copy.feat_dim,
                dim_out=len(np.unique(finetune_train_loader.dataset.targets)),
                encoder=model,
                only_readout=only_readout,
            )
        elif finetune_linear_or_nonlinear == 'mlp':
            # finetune_model = rep_distiller.models.classifier.MLPReadout(
            #     num_classes=len(finetune_train_loader.target.unique()),
            #     encoder=model,
            # )
            raise NotImplementedError
        else:
            raise ValueError

        if only_readout:
            params = finetune_model.readout.parameters()
        else:
            params = finetune_model.parameters()

        finetune_optimizers = torch.optim.SGD(
            params,
            lr=1e-3)

        run_epoch(split='train',
                  models_dict=torch.nn.ModuleDict({f'finetune_{model_name}': finetune_model}),
                  criterion=10,
                  loader=finetune_train_loader,
                  optimizer=finetune_optimizers)

        # Then call train
        train(models_dict=torch.nn.ModuleDict({f'finetune_{model_name}': finetune_model}),
              num_epochs=num_epochs,
              optimizer=finetune_optimizers,
              opt=opt,
              train_loader=finetune_train_loader,
              eval_loader=finetune_eval_loader,
              criteria_list=finetune_criteria_list)

    return 10


def pretrain_and_finetune(models_dict: torch.nn.ModuleDict,
                          pretrain_train_loader: DataLoader,
                          pretrain_eval_loader: DataLoader,
                          pretrain_epochs: int,
                          finetune_train_loader: DataLoader,
                          finetune_eval_loader: DataLoader,
                          finetune_epochs: int,
                          criteria_dict: torch.nn.ModuleDict,
                          optimizer: torch.optim.Optimizer,
                          opt: argparse.Namespace,
                          logger,
                          **kwargs):
    """
    Loop to alternate between pretraining and fine-tuning.
    """

    def compute_pretrain_loss(model_outputs: Dict[str, torch.Tensor],
                              target: torch.Tensor,
                              **kwargs
                              ) -> Dict[str, torch.Tensor]:
        del target
        distillation_loss = criteria_dict['custom'](
            model_outputs['student'],
            model_outputs['teacher'])
        total_loss = distillation_loss
        losses = dict(distillation_loss=distillation_loss,
                      total_loss=total_loss)
        return losses

    def compute_finetune_loss(model_outputs: Dict[str, torch.Tensor],
                              target: torch.Tensor,
                              criteria_dict: torch.nn.ModuleDict,
                              **kwargs
                              ) -> Dict[str, torch.Tensor]:

        classification_loss = criteria_dict['cross_entropy'](
            input=model_outputs['student_output'],
            target=target, )
        distillation_loss = criteria_dict['custom'](
            input=model_outputs['student_output'],
            target=model_outputs['teacher_output'])

        losses = dict(classification_loss=classification_loss,
                      distillation_loss=distillation_loss)
        return losses

    # # fine tune student and teacher
    # finetune(models_dict=models_dict,
    #          num_epochs=finetune_epochs,
    #          finetune_train_loader=finetune_train_loader,
    #          finetune_eval_loader=finetune_eval_loader,
    #          finetune_criteria_list=finetune_criteria_list,
    #          opt=opt)

    best_total_loss = np.inf

    for epoch_idx in range(1, pretrain_epochs + 1):

        rep_distiller.run.util.adjust_learning_rate(epoch_idx, opt, optimizer)
        print("==> pretraining...")
        for split in ['pretrain_train', 'pretrain_eval']:
            time1 = time.time()
            epoch_avg_stats_by_model = run_epoch(
                split=split,
                models_dict=models_dict,
                loader=pretrain_train_loader if split == 'pretrain_train' else pretrain_eval_loader,
                optimizer=optimizer,
                loss_fn=compute_pretrain_loss)
            time2 = time.time()
            print('epoch {}, split {}, loss: {}, total time {:.2f}'.format(
                epoch_idx,
                split,
                epoch_avg_stats_by_model['total_loss'],
                time2 - time1))

            wandb.log({f'{split}_{k}': v for k, v in epoch_avg_stats_by_model.items()},
                      step=epoch_idx)

        # save the best model
        if epoch_avg_stats_by_model['total_loss'] > best_total_loss:
            best_total_loss = epoch_avg_stats_by_model['total_loss']
            state = {
                'epoch': epoch_idx,
                'model': models_dict['student'].state_dict(),
                'best_total_loss': best_total_loss,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(
                opt.student_architecture))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch_idx % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch_idx,
                'model': models_dict['student'].state_dict(),
                'total_loss': epoch_avg_stats_by_model['total_loss'],
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(
                epoch=epoch_idx))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best total loss:', best_total_loss)


def pretrain(model_s, model_t, init_modules, criterion, train_loader, logger, opt):
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01
    else:
        lr = opt.learning_rate
    optimizer = torch.optim.SGD(init_modules.parameters(),
                                lr=lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)

    batch_time = rep_distiller.run.util.AverageMeter()
    data_time = rep_distiller.run.util.AverageMeter()
    losses = rep_distiller.run.util.AverageMeter()
    for epoch in range(1, opt.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):
            if opt.distill in ['crd']:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                # target = target.cuda()
                # index = index.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()

            # ============= forward ==============
            preact = (opt.distill == 'abound')
            feat_s, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if opt.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplemented('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # end of epoch
        logger.log_value('init_train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
            epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()


def run_epoch(split: str,
              models_dict: torch.nn.ModuleDict,
              loss_fn: Callable,
              loader: DataLoader,
              optimizer: torch.optim.Optimizer,
              ) -> Dict[str, float]:

    assert split in {'pretrain_train', 'pretrain_eval', 'train', 'eval', 'test'}
    training = split in {'pretrain_train', 'train'}
    torch.set_grad_enabled(training)

    if training:
        models_dict.train()
    else:
        models_dict.eval()

    stats_by_model = {model_name: rep_distiller.run.util.Statistics()
                      for model_name in models_dict}

    for batch_idx, (input_tensors, target_tensors) in enumerate(loader):

        # TODO: Figure out what to do about SwAV-like transforms
        # For different pretrained models (e.g. SwAV), the transforms map
        # each sample into the batch into a list of many (e.g. 7) tensors.
        # Here, we stack them and treat them as one big batch.
        if isinstance(input_tensors, list):
            # For now, use hack of taking all with shape (3, 36, 36)
            input_tensors = torch.cat([input_tensor for input_tensor in input_tensors
                                       if input_tensor.shape[1:] == (3, 36, 36)],
                                      dim=0)

        input_tensors = input_tensors.float()
        if torch.cuda.is_available():
            input_tensors = input_tensors.cuda()
            target_tensors = target_tensors.cuda()

        model_outputs = dict()
        for model_name, model in models_dict.items():
            output = model(input_tensors)
            model_outputs[model_name] = output

            # if split in {'train', 'eval', 'test'}:
            #     # measure accuracy and record loss
            #     acc1, acc5 = rep_distiller.run.util.accuracy(
            #         output,
            #         target_tensors,
            #         topk=(1, 5))
            #
            #     stats_by_model[model_name].update(
            #         loss=loss.item(),
            #         top1_acc_by_model=acc1,
            #         top5_acc_by_model=acc5,
            #         batch_size=input_tensors.shape[0],
            #     )

        losses = loss_fn(model_outputs, target_tensors)

        print('split {}, Batch {}, loss {}'.format(
            split,
            batch_idx,
            losses['total_loss']))

        if training:
            optimizer.zero_grad()
            losses['total_loss'].backward()
            optimizer.step()

            stats_by_model['student'].update(
                distillation_loss=losses['distillation_loss'].item(),
                total_loss=losses['total_loss'].item(),
                batch_size=input_tensors.shape[0])

        # if batch_idx % opt.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #         batch_idx, len(eval_loader), batch_time=batch_time_by_model, loss=total_losses_by_model,
        #         top1=top1_acc_by_model, top5=top5_acc_by_model))

    # avg_top1_acc_by_model = {model_name: top1_acc_by_model[model_name].avg
    #                          for model_name in models_dict}
    # avg_top5_acc_by_model = {model_name: top5_acc_by_model[model_name].avg
    #                          for model_name in models_dict}
    # avg_total_loss_by_model = {model_name: total_losses_by_model[model_name].avg
    #                            for model_name in models_dict}

    avg_stats_by_model = {stats_by_model[model_name].averages()
                          for model_name in stats_by_model}
    # log_epoch_summary(epoch, split, avg_stats_by_model)
    return avg_stats_by_model


def train(models_dict,
          num_epochs: int,
          optimizer: torch.optim.Optimizer,
          opt: argparse.Namespace,
          train_loader: DataLoader,
          eval_loader: DataLoader,
          criteria_list: torch.nn.ModuleList,
          ):
    # routine
    for epoch in range(1, num_epochs + 1):

        # Initial eval
        finetune_acc, _, _ = eval_epoch(
            eval_loader=eval_loader,
            model=finetune_model,
            criterion=finetune_criteria_list,
            opt=opt)

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss, train_classification_loss, train_kd_loss, train_custom_loss = train(
            epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(
            val_loader, model_s, criterion_cls, opt)

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


def train_epoch_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = rep_distiller.run.util.AverageMeter()
    data_time = rep_distiller.run.util.AverageMeter()
    losses = rep_distiller.run.util.AverageMeter()
    top1 = rep_distiller.run.util.AverageMeter()
    top5 = rep_distiller.run.util.AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_epoch_distill(epoch: int,
                        train_loader,
                        models_dict: torch.nn.ModuleDict,
                        criterion_list,
                        optimizer: torch.optim.Optimizer,
                        opt):
    """Train one epoch (distillation)"""
    # set modules as train()
    for module in models_dict:
        module.train()
    # set teacher as eval()
    models_dict[-1].eval()

    if opt.distill == 'abound':
        models_dict[1].eval()
    elif opt.distill == 'factor':
        models_dict[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = models_dict[0]
    model_t = models_dict[-1]

    batch_time = rep_distiller.run.util.AverageMeter()
    data_time = rep_distiller.run.util.AverageMeter()
    total_losses = rep_distiller.run.util.AverageMeter()
    classification_losses = rep_distiller.run.util.AverageMeter()
    kl_div_losses = rep_distiller.run.util.AverageMeter()
    custom_losses = rep_distiller.run.util.AverageMeter()
    top1 = rep_distiller.run.util.AverageMeter()
    top5 = rep_distiller.run.util.AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = models_dict[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'krd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = models_dict[1](feat_s[-1])
            f_t = models_dict[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = models_dict[1](feat_s[-2])
            factor_t = models_dict[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        classification_loss = opt.classification_weight * loss_cls
        kl_div_loss = opt.kl_div_weight * loss_div
        custom_loss = opt.custom_weight * loss_kd
        total_loss = classification_loss + kl_div_loss + custom_loss

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        classification_losses.update(classification_loss.item(), input.size(0))
        kl_div_losses.update(kl_div_loss.item(), input.size(0))
        custom_losses.update(custom_loss.item(), input.size(0))
        total_losses.update(total_loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=total_losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, total_losses.avg, classification_losses.avg, kl_div_losses.avg, custom_losses.avg


def init(model_s, model_t, init_modules, criterion, train_loader, logger, opt):
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if opt.model_s in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01
    else:
        lr = opt.learning_rate
    optimizer = optim.SGD(init_modules.parameters(),
                          lr=lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    batch_time = rep_distiller.run.util.AverageMeter()
    data_time = rep_distiller.run.util.AverageMeter()
    losses = rep_distiller.run.util.AverageMeter()
    for epoch in range(1, opt.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):
            if opt.distill in ['crd']:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                # target = target.cuda()
                # index = index.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()

            # ============= forward ==============
            preact = (opt.distill == 'abound')
            feat_s, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if opt.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            else:
                raise NotImplemented('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

        # end of epoch
        logger.log_value('init_train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
            epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()


if __name__ == '__main__':
    pass
