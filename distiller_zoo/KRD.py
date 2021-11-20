from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelRepresentationDistillation(nn.Module):
    """Kernel Representation Distillation."""

    def __init__(self,
                 primal_or_dual: str,
                 c: float):
        assert primal_or_dual in {'primal', 'dual'}
        assert c > 0
        self.primal_or_dual = primal_or_dual
        self.c = c
        super(KernelRepresentationDistillation, self).__init__()

    def forward(self,
                f_s,
                f_t):
        batch_size = f_s.shape[0]
        f_s = f_s.reshape(batch_size, -1)
        f_t = f_t.reshape(batch_size, -1)

        if self.primal_or_dual == 'primal':
            H_s = self.compute_primal_hat_matrix(f_s, c=self.c)
            H_t = self.compute_primal_hat_matrix(f_t, c=self.c)
        elif self.primal_or_dual == 'dual':
            H_s = self.compute_dual_hat_matrix(f_s, c=self.c)
            H_t = self.compute_dual_hat_matrix(f_t, c=self.c)
        else:
            raise ValueError

        # G_s = torch.nn.functional.normalize(G_s)
        # G_t = torch.nn.functional.normalize(G_t)

        loss = torch.mean(torch.square(H_s - H_t))
        return loss

    @staticmethod
    def compute_primal_hat_matrix(X: torch.Tensor,
                                  c: float = 1.0) -> torch.Tensor:
        """
        Computes the primal hat matrix X (X^T X + c I)^{-1} X^T

        Args:
            X: shape (batch size, feature dim)
            c: float, must be > 0
        """

        assert c > 0.
        batch_size, feature_dim = X.shape
        hat_matrix = torch.einsum(
            'ab,bc,cd->ad',
            X,
            torch.linalg.inv(X.T @ X + c * torch.eye(feature_dim).cuda()),
            X.T,
        )
        return hat_matrix

    @staticmethod
    def compute_dual_hat_matrix(X: torch.Tensor,
                                c: float = 1.0) -> torch.Tensor:
        """
        Computes the dual hat matrix X X^T (X X^T + c I)^{-1}

        Args:
            X: shape (batch size, feature dim)
            c: float, must be > 0
        """

        assert c > 0.
        batch_size, feature_dim = X.shape
        hat_matrix = torch.einsum(
            'ab,bc,cd->ad',
            X,
            X.T,
            torch.linalg.inv(X @ X.T + c * torch.eye(batch_size).cuda()),
        )
        return hat_matrix
