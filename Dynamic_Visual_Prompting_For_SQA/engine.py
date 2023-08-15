import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched
import torch.nn.functional as F
import numpy as np

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, data_loader_eval: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
  
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))



    prefix_img = torch.tensor(data_loader.dataset.tokenizer.encode("Image: ", bos=False, eos=False), dtype=torch.int64)
    prefix_nonimg = torch.tensor(data_loader.dataset.tokenizer.encode("Image: N/A", bos=False, eos=False), dtype=torch.int64)

    if args.kab_app:
        kab_val_iter = iter(data_loader_eval)
    else:
        kab_val_iter = None

    for data_iter_step, (examples, labels, example_mask, images, indicators, prompt_len) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        prefix_img=prefix_img.to(examples.device)
        prefix_nonimg=prefix_nonimg.to(examples.device)

        kab_state = False
        c_loss = model(examples, labels,images=images, prefix_img=prefix_img, prefix_nonimg=prefix_nonimg, img_indicators=indicators, prompt_len = prompt_len, kab_state=kab_state)
        loss = c_loss
        loss_value = loss.item()
        c_loss_value = c_loss.item()


        if torch.isnan(loss):
            print("NaN loss encountered. Skipping this batch.")
            continue

        loss = loss/accum_iter

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0,clip_grad=args.clip_grad)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

            if args.kab_app:
                try:
                    examples, labels, example_mask, images, indicators, prompt_len = next(kab_val_iter)

                except StopIteration:
                    kab_val_iter = iter(data_loader_eval)
                    examples, labels, example_mask, images, indicators, prompt_len = next(kab_val_iter)
                kab_state = True
                ave_ce_loss, ave_score, ave_arch_loss, max_reward = kab_update_step(model, examples, labels,
                                                                                    example_mask, images, indicators,
                                                                                    prompt_len, kab_state, prefix_img, prefix_nonimg)
                model.module.kab_print_info += 1
                if model.module.kab_print_info % 10 == 0 and model.module.kab_print_info != 0:
                    print('Arch Loss %.4f\t CE Loss %.4f\t Ave Score %.4f\t Max Reward %.4f\t' % (
                        ave_arch_loss, ave_ce_loss, ave_score, max_reward))
                    print('%s' % (model.module.get_layer()))
                    model.module.kab_print_info = 0

                model.module.active_index_buffer = []
                model.module.up_weight_buffer = []
                model.module.arch_loss_buffer = []
                model.module.loss_buffer = []
                model.module.score_buffer = []
                model.module.reward_buffer = []
            else:
                pass

        torch.cuda.synchronize()

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def kab_update_step(model, examples, labels, example_mask, images, indicators, prompt_len, kab_state, prefix_img, prefix_nonimg):
    for i in range(10):
        with torch.no_grad():
            loss = model(examples,
                         labels,
                         images=images,
                         prefix_img=prefix_img,
                         prefix_nonimg=prefix_nonimg,
                         img_indicators=indicators,
                         prompt_len = prompt_len,
                         kab_state=kab_state)
            model.module.loss_buffer.append(loss)
            score = math.exp(-loss) * 100
            model.module.score_buffer.append(score)
        arch_loss = 0
        if model.module.layer_alphas.grad is not None:
            model.module.layer_alphas.grad.data.zero_()
        for log_probs in model.module.log_probs_buffer:
            arch_loss = arch_loss + log_probs
        model.module.log_probs_buffer = []
        arch_loss = -arch_loss
        model.module.arch_loss_buffer.append(arch_loss)

    ave_loss = sum(model.module.loss_buffer) / 10
    ave_arch_loss = sum(model.module.arch_loss_buffer) / 10
    ave_score = sum(model.module.score_buffer) / 10

    for i in range(10):
        reward = model.module.score_buffer[i] - ave_score
        model.module.reward_buffer.append(reward)

    for j in range(10):
        sample = model.module.active_index_buffer[j]
        model.module.layer_alphas.data[sample] += model.module.reward_buffer[j] * model.module.up_weight_buffer[j]

    max_reward = max(model.module.reward_buffer)

    return ave_loss, ave_score, ave_arch_loss, max_reward



def val_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    accum_iter = args.accum_iter

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    for data_iter_step, (examples, labels, example_mask) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        with torch.no_grad():
             c_loss  = model(examples, labels)
        loss = c_loss
        loss_value = loss.item()

        c_loss_value = c_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(closs=c_loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        c_loss_value_reduce = misc.all_reduce_mean(c_loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('c_train_loss', c_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
