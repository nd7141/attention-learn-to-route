import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            output = model(move_to(bat, opts.device), return_pi=False)
            cost = output[0]
            # print('Dataset on valid', bat)
            # print('Cost on valid', cost)
            if len(output) > 3:
                print()
                print(output[0])
                print(output[-1])
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler,
                epoch, val_dataset, problem, tb_logger, opts, extra):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    start_time = time.time()
    lr_scheduler.step(epoch)

    # Generate training data
    dataset = problem.make_dataset(
        filename=opts.train_dataset, num_samples=opts.epoch_size,
        size=opts.graph_size, distribution=opts.data_distribution,
        degree=opts.degree, steps=opts.awe_steps, awe_samples=opts.awe_samples
    )

    training_dataset = baseline.wrap_dataset(dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1)

    if len(training_dataset) <= opts.batch_size:
        step = epoch
    else:
        step = epoch * (len(training_dataset) // opts.batch_size + 1)

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], epoch)

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts,
            extra
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)
    if avg_reward < extra["avg_reward"]:
        extra["avg_reward"] = avg_reward
        extra["best_epoch"] = epoch

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_reward, epoch)

    updated = baseline.epoch_callback(model, epoch)
    # update to baseline
    if updated is not None and updated == True:
        extra["updates"] += 1
    if not opts.no_tensorboard:
        tb_logger.log_value('update_baseline', extra["updates"], step)




def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts,
        extra
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # Evaluate model, get costs and log probabilities
    cost, log_likelihood, entropy = model(x)
    # print('Cost on train', cost)

    # Evaluate baseline, get baseline loss if any (only for critic)
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
    if bl_loss == True and opts.baseline == 'constant':
        extra["updates"] += 1

    # Calculate loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    entropy_loss = entropy.mean()
    loss = reinforce_loss + bl_loss - opts.entropy * entropy_loss

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    # Logging
    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
            log_likelihood, reinforce_loss, bl_loss, entropy_loss,  tb_logger, opts, extra)
