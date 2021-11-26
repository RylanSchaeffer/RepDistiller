from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class KernelRepresentationDistillation(nn.Module):
    """Kernel Representation Distillation."""

    def __init__(self,
                 primal_or_dual: str,
                 ridge_prefactor: float,
                 normalize: bool):
        assert primal_or_dual in {'primal', 'dual'}
        assert ridge_prefactor > 0
        super(KernelRepresentationDistillation, self).__init__()
        self.primal_or_dual = primal_or_dual
        self.ridge_prefactor = ridge_prefactor
        self.normalize = normalize

    def forward(self,
                f_s,
                f_t):
        batch_size = f_s.shape[0]
        f_s = f_s.reshape(batch_size, -1)
        f_t = f_t.reshape(batch_size, -1)

        if self.normalize:
            f_s = torch.nn.functional.normalize(f_s)
            f_t = torch.nn.functional.normalize(f_t)

        if self.primal_or_dual == 'primal':
            H_s = self.compute_primal_hat_matrix(f_s, c=self.ridge_prefactor)
            H_t = self.compute_primal_hat_matrix(f_t, c=self.ridge_prefactor)
        elif self.primal_or_dual == 'dual':
            H_s = self.compute_dual_hat_matrix(f_s, c=self.ridge_prefactor)
            H_t = self.compute_dual_hat_matrix(f_t, c=self.ridge_prefactor)
        else:
            raise ValueError

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