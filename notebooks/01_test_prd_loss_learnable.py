import os


os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
# Control GPU Access
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7,8,9"

import numpy as np
import torch
import wandb

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


from rep_distiller.distiller_zoo.PRD import PretrainedRepresentationDistillation
from rep_distiller.models.readout import MLPReadout

hyperparameter_defaults = {
    'num_steps': 10 ** 5,
    'primal_or_dual': 'primal',
    'learning_rate': 1e-1,
    'weight_decay': 1e-4,
    'sizes': [32, 32, 32, 13],
    'ridge_prefactor': 1e2,
    'bias_additive': 1.25,
    'bias_multiplicative': None,
    'weight_additive': None,
    'weight_multiplicative': 3.5,
    'normalize_teacher': True,
    'normalize_student': False,
    'seed': 0,
}
hyperparameter_defaults['batch_size'] = 2 * np.sum(
    np.square(hyperparameter_defaults['sizes']) + hyperparameter_defaults['sizes'])
wandb.init(project='test_prd_loss_learnable',
           config=hyperparameter_defaults)
config = wandb.config

np.random.seed(config.seed)

teacher = MLPReadout(sizes=config.sizes,
                     train_only_readout=False).double().cuda()
for layer in teacher.readout:
    if hasattr(layer, 'bias'):
        layer.bias.data[:] += config.bias_additive
    if hasattr(layer, 'weight'):
        layer.weight.data[:] *= config.weight_multiplicative
student = MLPReadout(sizes=config.sizes,
                     train_only_readout=False).double().cuda()

optimizer = torch.optim.SGD(
    student.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay)

loss_fn = PretrainedRepresentationDistillation(
    primal_or_dual=config.primal_or_dual,
    ridge_prefactor=config.ridge_prefactor,
    normalize=False,
)

losses_per_epoch = np.zeros(config.num_steps)
random_data = torch.from_numpy(
    np.random.normal(loc=3, scale=7, size=(config.batch_size, config.sizes[0]))).double()

random_data = random_data.cuda()

with torch.no_grad():
    f_t = teacher(random_data)
    if config.normalize_teacher:
        f_t = torch.divide(f_t, torch.norm(f_t, dim=1)[:, np.newaxis])

for step_idx in range(0, config.num_steps + 1):
    optimizer.zero_grad()
    f_s = student(random_data)
    if config.normalize_student:
        f_s = torch.divide(f_s, torch.norm(f_s, dim=1)[:, np.newaxis])
    loss = loss_fn(f_s, f_t)

    if step_idx == 0:
        init_loss = loss

    loss.backward()
    optimizer.step()
    # optimizer.zero_grad()

    print(f'Step: {step_idx}\tLoss: {loss}')

    if step_idx % 100 == 0:
        wandb.log({
            'loss': loss / init_loss,
        },
            step=step_idx)
