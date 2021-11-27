import torch
import torch.nn as nn
from typing import Callable, Dict

from rep_distiller.distiller_zoo import DistillKL, PretrainedRepresentationDistillation


def create_criteria_dict(opt,
                         ) -> torch.nn.ModuleDict:
    # data = torch.randn(2, 3, 32, 32)
    # model_t.eval()
    # model_s.eval()
    # feat_t, _ = model_t(data, is_feat=True)
    # feat_s, _ = model_s(data, is_feat=True)

    criteria_dict = nn.ModuleDict({
        'classification_loss': nn.CrossEntropyLoss(),
        'knowledge_distillation_loss': DistillKL(opt.kd_T)
    })
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
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
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
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
        module_list.append(embed_s)
        module_list.append(embed_t)
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
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    # other distillation loss
    criteria_dict['custom_loss'] = criterion_kd

    return criteria_dict


def compute_all_losses(output_tensors_by_model: Dict[str, torch.Tensor],
                       target_tensors: torch.Tensor,
                       losses_callables_dict: torch.nn.ModuleDict,
                       losses_prefactors_dict: Dict[str, float],
                       **kwargs) -> Dict[str, torch.Tensor]:
    losses_dict = {}
    total_loss = torch.tensor(0., requires_grad=True)
    for loss_str, loss_prefactor in losses_prefactors_dict.items():
        if loss_str == 'classification':
            assert target_tensors is not None
            classification_loss = losses_callables_dict['classification'](
                model_outputs,
                target=target_tensors, )
            total_loss = total_loss + loss_prefactor * classification_loss
        elif loss_str == 'knowledge_distillation':
            raise NotImplementedError
        else:
            raise NotImplementedError

    losses_dict['total_loss'] = total_loss
    return losses_dict


def compute_classification_loss(output_tensors_by_model: Dict[str, torch.Tensor],
                                target_tensors: torch.Tensor,
                                losses_callables_dict: torch.nn.ModuleDict,
                                **kwargs,
                                ) -> Dict[str, torch.Tensor]:
    losses_by_model = dict()
    for model_name, model_outputs in output_tensors_by_model.items():
        total_loss = torch.tensor(0., requires_grad=True, device='cuda')
        classification_loss = losses_callables_dict['classification_loss'](
            input=output_tensors_by_model[model_name],
            target=target_tensors)
        total_loss = total_loss + classification_loss
        losses_by_model[model_name] = dict(classification_loss=classification_loss,
                                           total_loss=total_loss)
    return losses_by_model


def compute_pretrain_loss(output_tensors_by_model: Dict[str, torch.Tensor],
                          losses_callables_dict: torch.nn.ModuleDict,
                          **kwargs
                          ) -> Dict[str, torch.Tensor]:
    losses_by_model = dict()
    total_loss = torch.tensor(0., requires_grad=True, device='cuda')
    pretrained_distillation_loss = losses_callables_dict['custom_loss'](
        output_tensors_by_model['student'],
        output_tensors_by_model['teacher'])
    total_loss = total_loss + pretrained_distillation_loss
    losses_by_model['student'] = dict(
        distillation_loss=pretrained_distillation_loss,
        total_loss=total_loss)
    return losses_by_model
