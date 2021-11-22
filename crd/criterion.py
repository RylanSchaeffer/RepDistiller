import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7


class CRDLoss(nn.Module):
    """CRD Loss function
    includes two symmetric parts:
    (a) using teacher as anchor, choose positive and negatives over the student side
    (b) using student as anchor, choose positive and negatives over the teacher side

    Args:
        opt.s_dim: the dimension of student's feature
        opt.t_dim: the dimension of teacher's feature
        opt.feat_dim: the dimension of the projection space
        opt.nce_k: number of negatives paired with each positive
        opt.nce_t: the temperature
        opt.nce_m: the momentum for updating the memory buffer
        opt.n_data: the number of samples in the training set, therefore the memory buffer is: opt.n_data x opt.feat_dim
    """
    def __init__(self,
                 student_dim: int,
                 teacher_dim: int,
                 projection_dim: int,
                 normalize: bool,
                 num_data: int,
                 num_neg_examples_per_pos_example: int,
                 softmax_temp: float,
                 momentum: float,
                 ):

        assert softmax_temp > 0.
        assert momentum > 0.

        super(CRDLoss, self).__init__()
        self.embed_s = Embed(dim_in=student_dim,
                             dim_out=projection_dim,
                             normalize_bool=normalize)
        self.embed_t = Embed(dim_in=teacher_dim,
                             dim_out=projection_dim,
                             normalize_bool=normalize)
        self.contrast = ContrastMemory(
            projection_dim,
            num_data,
            num_neg_examples_per_pos_example,
            softmax_temp=softmax_temp,
            momentum=momentum)
        self.criterion_t = ContrastLoss(num_data)
        self.criterion_s = ContrastLoss(num_data)

    def forward(self, f_s, f_t, idx, contrast_idx=None):
        """
        Args:
            f_s: the feature of student network, size [batch_size, s_dim]
            f_t: the feature of teacher network, size [batch_size, t_dim]
            idx: the indices of these positive samples in the dataset, size [batch_size]
            contrast_idx: the indices of negative samples, size [batch_size, nce_k]

        Returns:
            The contrastive loss
        """
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        out_s, out_t = self.contrast(f_s, f_t, idx, contrast_idx)
        s_loss = self.criterion_s(out_s)
        t_loss = self.criterion_t(out_t)
        loss = s_loss + t_loss
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (18)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self,
                 dim_in: int = 1024,
                 dim_out: int = 128,
                 normalize_bool: bool = True):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)
        if normalize_bool:
            print('Using normalization')
        else:
            print('Not using normalization')
        self.normalize_bool = normalize_bool

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        if self.normalize_bool:
            x = self.l2norm(x)
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
