import torch

from . import architecture_dict


def create_model_from_architecture_str(architecture_str: str,
                                       output_dim: int,
                                       ) -> torch.nn.Module:

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


def load_selfsupervised_teacher(teacher_name: str,
                                pretrain_dataset: str,
                                ) -> torch.nn.Module:
    print('==> loading teacher model')

    if teacher_name == 'swav':
        # https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#swav
        from pl_bolts.models.self_supervised import SwAV
        from pl_bolts.models.self_supervised.swav.transforms import (SwAVTrainDataTransform, SwAVEvalDataTransform)
        if pretrain_dataset == 'imagenet':
            weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
            train_transform = SwAVTrainDataTransform()
            eval_transform = SwAVEvalDataTransform()
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
    elif teacher_name == 'simclr':
        # https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_models.html#simclr
        from pl_bolts.models.self_supervised import SimCLR
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        model = SimCLR.load_from_checkpoint(weight_path, strict=False)
        from pl_bolts.models.self_supervised.simclr.transforms import (
            SimCLREvalDataTransform, SimCLRTrainDataTransform)
        # TODO: What are these 32s
        train_transform = SimCLRTrainDataTransform(32)
        eval_transform = SimCLREvalDataTransform(32)
    elif teacher_name == 'cpc_v2':
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
    elif teacher_name == 'byol':
        raise NotImplementedError
    elif teacher_name == 'clip':
        import clip

        # Check available models
        # clip.available_models()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)

        # image_features = model.encode_image(image)

        train_transform = preprocess
        eval_transform = preprocess
    else:
        raise NotImplementedError

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

