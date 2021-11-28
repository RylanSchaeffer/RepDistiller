from __future__ import print_function, division

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import seaborn as sns
import time
import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Dict, List, Tuple
import wandb

import rep_distiller.globals
import rep_distiller.losses
import rep_distiller.models.helpers
import rep_distiller.models.readout
import rep_distiller.run.helpers


def finetune(models_dict: torch.nn.ModuleDict,
             num_epochs: int,
             finetune_train_loader: DataLoader,
             finetune_eval_loader: DataLoader,
             criteria_dict: torch.nn.ModuleDict,
             opt: argparse.Namespace,
             finetune_linear_or_nonlinear: str = 'linear',
             only_readout: bool = True,
             **kwargs):

    assert finetune_linear_or_nonlinear in {'linear', 'mlp'}
    finetune_models_dict = dict()
    optimizers_dict = dict()
    for model_name, model in models_dict.items():
        finetune_model, finetune_optimizer = rep_distiller.models.helpers.create_finetune_model(
            model=model,
            dim_out=len(np.unique(finetune_train_loader.dataset.targets)),
            train_only_readout=only_readout)
        finetune_models_dict[f'finetune_{model_name}'] = finetune_model
        optimizers_dict[f'finetune_{model_name}'] = finetune_optimizer
    finetune_models_dict = torch.nn.ModuleDict(finetune_models_dict)

    finetune_results_dict = {'finetune_epoch_idx': []}
    for finetune_epoch_idx in range(num_epochs):
        finetune_results_dict['finetune_epoch_idx'].append(finetune_epoch_idx)
        for split in ['eval', 'train']:
            split_epoch_avg_stats_by_model = run_epoch_finetune(
                split=split,
                models_dict=finetune_models_dict,
                losses_callables_dict=criteria_dict,
                loader=finetune_train_loader if split == 'train' else finetune_eval_loader,
                optimizers_dict=optimizers_dict)
            for model_name, model_split_epoch_avg_stats in split_epoch_avg_stats_by_model.items():
                for key, value in model_split_epoch_avg_stats.items():
                    new_key = f'{model_name}_{split}_{key}'
                    if new_key not in finetune_results_dict:
                        finetune_results_dict[new_key] = [value]
                    else:
                        finetune_results_dict[new_key].append(value)

        print(f'grad_steps: {rep_distiller.globals.num_gradient_steps} Finetune Epoch: {finetune_epoch_idx}\n'
              f'WandB Dict: {finetune_results_dict}')

    finetune_results_df = pd.DataFrame.from_dict(finetune_results_dict)
    finetune_results_df['finetune_student_minus_teacher_eval_top_1_acc'] = \
        finetune_results_df['finetune_student_eval_top_1_acc'] - finetune_results_df['finetune_teacher_eval_top_1_acc']
    finetune_results_df['finetune_student_minus_teacher_eval_top_5_acc'] = \
        finetune_results_df['finetune_student_eval_top_5_acc'] - finetune_results_df['finetune_teacher_eval_top_5_acc']

    sns.lineplot(data=finetune_results_df,
                 x='finetune_epoch_idx',
                 y='finetune_student_minus_teacher_eval_top_1_acc')
    plt.xlabel('Finetuning Epoch Index')
    plt.ylabel('Student - Teacher Finetune Eval Top 1 Acc')
    # Convert to PIL to be able to save
    # See https://stackoverflow.com/a/61756899/4570472 and comment
    fig = plt.gcf()
    fig.canvas.get_renderer()
    pil_img = Image.frombytes('RGB',
                              fig.canvas.get_width_height(),
                              fig.canvas.tostring_rgb())
    wandb.log({'finetune_table': wandb.Table(dataframe=finetune_results_df),
               'finetune_learning_curve': wandb.Image(data_or_path=pil_img)},
              step=rep_distiller.globals.num_gradient_steps)


def pretrain_and_finetune(models_dict: torch.nn.ModuleDict,
                          pretrain_train_loader: DataLoader,
                          pretrain_eval_loader: DataLoader,
                          pretrain_epochs: int,
                          finetune_train_loader: DataLoader,
                          finetune_eval_loader: DataLoader,
                          finetune_epochs: int,
                          losses_callables_dict: torch.nn.ModuleDict,
                          optimizer: torch.optim.Optimizer,
                          opt: argparse.Namespace,
                          **kwargs):
    """
    Loop to alternate between pretraining and fine-tuning.
    """

    best_total_loss = np.inf

    # for pretrain_epoch_idx in range(1, 1 + pretrain_epochs):
    for pretrain_epoch_idx in range(pretrain_epochs):

        rep_distiller.run.helpers.adjust_learning_rate(pretrain_epoch_idx, opt, optimizer)

        if pretrain_epoch_idx % opt.num_pretrain_epochs_per_finetune == 0:
            finetune(models_dict=models_dict,
                     num_epochs=finetune_epochs,
                     finetune_train_loader=finetune_train_loader,
                     finetune_eval_loader=finetune_eval_loader,
                     criteria_dict=losses_callables_dict,
                     opt=opt)

        wandb_dict = {}
        for split in ['pretrain_eval', 'pretrain_train']:
            start_time = time.time()
            split_epoch_avg_stats_by_model = run_epoch_pretrain(
                split=split,
                models_dict=models_dict,
                loader=pretrain_train_loader if split == 'pretrain_train' else pretrain_eval_loader,
                optimizers_dict={'student': optimizer},
                losses_callables_dict=losses_callables_dict)
            end_time = time.time()
            print('num grad steps {}, split {}, student loss: {}, total time {:.2f}'.format(
                rep_distiller.globals.num_gradient_steps,
                split,
                split_epoch_avg_stats_by_model['student']['total_loss'],
                end_time - start_time))

            for model_name, model_split_epoch_avg_stats in split_epoch_avg_stats_by_model.items():
                wandb_dict.update(**{
                    f'{model_name}_{split}_{k}': v for k, v in
                    model_split_epoch_avg_stats.items()})

        wandb.log(wandb_dict,
                  step=rep_distiller.globals.num_gradient_steps)

        # save the best model
        student_eval_loss = split_epoch_avg_stats_by_model['student']['total_loss']
        if student_eval_loss > best_total_loss:
            best_total_loss = student_eval_loss
            state = {
                'epoch': pretrain_epoch_idx,
                'model': models_dict['student'].state_dict(),
                'best_total_loss': best_total_loss,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(
                opt.student_architecture))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if pretrain_epoch_idx % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': pretrain_epoch_idx,
                'model': models_dict['student'].state_dict(),
                'total_loss': student_eval_loss,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(
                epoch=pretrain_epoch_idx))
            torch.save(state, save_file)

    print('best total loss:', best_total_loss)


def run_epoch_finetune(split: str,
                       models_dict: torch.nn.ModuleDict,
                       loader: DataLoader,
                       optimizers_dict: Dict[str, torch.optim.Optimizer],
                       losses_callables_dict: torch.nn.ModuleDict,
                       ) -> Dict[str, Dict[str, float]]:
    assert split in {'train', 'eval', 'test'}

    # Are we training?
    training = split in {'train'}
    torch.set_grad_enabled(training)

    for model_name, model in models_dict.items():
        if training:
            model.train()
        else:
            models_dict.eval()

    stats_by_model = {model_name: rep_distiller.run.helpers.Statistics()
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

        output_tensors_by_model = dict()
        for model_name, model in models_dict.items():
            start_time = time.time()
            model_output_tensors = model(input_tensors)
            end_time = time.time()
            output_tensors_by_model[model_name] = model_output_tensors
            stats_by_model[model_name].update(
                batch_size=input_tensors.shape[0])

        # measure accuracy and record loss
        topk_acc_by_model = rep_distiller.run.helpers.compute_accuracy(
            output_tensors_by_model=output_tensors_by_model,
            target=target_tensors,
            topk=(1, 5))

        for model_name, model_top_k_acc in topk_acc_by_model.items():
            stats_by_model[model_name].update(
                **model_top_k_acc)

        losses_by_model = rep_distiller.losses.compute_classification_loss(
            output_tensors_by_model=output_tensors_by_model,
            target_tensors=target_tensors,
            losses_callables_dict=losses_callables_dict)

        for model_name, model_losses in losses_by_model.items():
            stats_by_model[model_name].update(**{
                k: v.item() for k, v in model_losses.items()})

        # if batch_idx % opt.print_freq == 0:
        #             # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        # print('Test: [{0}/{1}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #     batch_idx, len(loader), batch_time=batch_time,  # loss=losses['total_loss'],
        #     top1=acc1, top5=acc5))

        if training:
            for model_name in losses_by_model:
                model_optimizer = optimizers_dict[model_name]
                model_optimizer.zero_grad()
                losses_by_model[model_name]['total_loss'].backward()
                model_optimizer.step()

        break

    avg_stats_by_model = {model_name: model_stats.averages()
                          for model_name, model_stats in stats_by_model.items()}
    return avg_stats_by_model


def run_epoch_pretrain(split: str,
                       models_dict: torch.nn.ModuleDict,
                       loader: DataLoader,
                       optimizers_dict: Dict[str, torch.optim.Optimizer],
                       losses_callables_dict: torch.nn.ModuleDict,
                       ) -> Dict[str, Dict[str, float]]:

    assert split in {'pretrain_train', 'pretrain_eval', 'pretrain_test'}

    # Are we training?
    training = split in {'pretrain_train'}
    torch.set_grad_enabled(training)
    for model_name, model in models_dict.items():
        if training:
            model.train()
        else:
            models_dict.eval()

    stats_by_model = {model_name: rep_distiller.run.helpers.Statistics()
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

        output_tensors_by_model = dict()
        for model_name, model in models_dict.items():

            model_output_tensors = model(input_tensors)
            output_tensors_by_model[model_name] = model_output_tensors
            stats_by_model[model_name].update(
                batch_size=input_tensors.shape[0])

        losses_by_model = rep_distiller.losses.compute_pretrain_loss(
            output_tensors_by_model=output_tensors_by_model,
            target_tensors=target_tensors,
            losses_callables_dict=losses_callables_dict)

        for model_name, model_losses in losses_by_model.items():
            stats_by_model[model_name].update(**{
                k: v.item() for k, v in model_losses.items()})


        # if batch_idx % opt.print_freq == 0:
        #             # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        # print('Test: [{0}/{1}]\t'
        #       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #       'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #       'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #     batch_idx, len(loader), batch_time=batch_time,  # loss=losses['total_loss'],
        #     top1=acc1, top5=acc5))

        if training:
            for model_name in losses_by_model:
                model_optimizer = optimizers_dict[model_name]
                model_optimizer.zero_grad()
                losses_by_model[model_name]['total_loss'].backward()
                model_optimizer.step()
                rep_distiller.globals.num_gradient_steps += 1

        break

    avg_stats_by_model = {model_name: model_stats.averages()
                          for model_name, model_stats in stats_by_model.items()}
    return avg_stats_by_model


def train_distill():
    raise NotImplementedError
