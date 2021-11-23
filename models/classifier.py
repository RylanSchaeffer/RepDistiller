from __future__ import print_function

import torch.nn as nn


#########################################
# ===== Classifiers ===== #
#########################################

class LinearClassifier(nn.Module):

    def __init__(self,
                 dim_in: int,
                 num_classes: int = 10):
        super(LinearClassifier, self).__init__()

        self.net = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.net(x)


class NonLinearClassifier(nn.Module):

    def __init__(self,
                 dim_in: int,
                 num_classes: int = 10,
                 dropout_prob: float = 0.1):
        super(NonLinearClassifier, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 200),
            nn.Dropout(p=dropout_prob),
            nn.BatchNorm1d(200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        return self.net(x)
