import numpy as np
from typing import Callable, Dict, List, Tuple

from rep_distiller.run.loops import run_epoch_finetune


def create_hook_fns_dict(hook_fns_frequencies: Dict[int, Callable],
                         start_grad_step: int,
                         num_grad_steps,
                         ) -> Dict[int, List[Callable]]:

    # hook_fns_frequencies: list of (how many gradient steps per function call, function to call).
    # function must accept single input argument, hook_input.
    # Two unique values:
    #   0: run at start
    #   -1: run at end
    hooks_fn_dict = {}
    for freq, hook_fn in hook_fns_frequencies:

        # decide which step(s) to call hook at
        if freq == 0:
            hook_call_at_grad_steps = [start_grad_step]
        elif freq == -1:
            hook_call_at_grad_steps = [start_grad_step + num_grad_steps - 1]
        else:
            hook_call_at_grad_steps = np.arange(
                start=start_grad_step,
                stop=start_grad_step + num_grad_steps,
                step=freq,
                dtype=np.int)

        # add hook object reference to hooks_fn_dict at appropriate steps
        for grad_step in hook_call_at_grad_steps:
            if grad_step not in hooks_fn_dict:
                hooks_fn_dict[grad_step] = []
            hooks_fn_dict[grad_step].append(hook_fn)

    return hooks_fn_dict


def create_hook_fns_train(start_grad_step: int,
                          num_grad_steps: int,
                          ):

    plot_freq = 250

    hook_fns_frequencies = [
        # (0, hook_log_params),
        # (plot_freq, hook_print_model_progress),
        # (100, hook_write_scalars),
        # (0, utils.plot.hook_plot_task_block_side_trial_side_by_trial_number),
        # (5000, hook_save_model),
        (1, hook_pretrain_train_epoch),
    ]

    train_hooks = create_hook_fns_dict(
        hook_fns_frequencies=hook_fns_frequencies,
        start_grad_step=start_grad_step,
        num_grad_steps=num_grad_steps)

    return train_hooks


def hook_pretrain_train_epoch(hook_inputs: Dict):
    raise NotImplementedError
