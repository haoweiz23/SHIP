import math
import sys
from typing import Iterable, Optional
from timm.utils.model import unwrap_model
import torch
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from lib import utils
import random
import time
import numpy as np

def sample_configs(choices, is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):

    config = {}
    depth = choices['depth']

    if is_visual_prompt_tuning == False and is_adapter == False and is_LoRA == False and is_prefix==False:
        visual_prompt_depth = random.choice(choices['visual_prompt_depth'])
        lora_depth = random.choice(choices['lora_depth'])
        adapter_depth = random.choice(choices['adapter_depth'])
        prefix_depth = random.choice(choices['prefix_depth'])
        config['visual_prompt_dim'] = [random.choice(choices['visual_prompt_dim']) for _ in range(visual_prompt_depth)] + [0] * (depth - visual_prompt_depth)
        config['lora_dim'] = [random.choice(choices['lora_dim']) for _ in range(lora_depth)] + [0] * (depth - lora_depth)
        config['adapter_dim'] = [random.choice(choices['adapter_dim']) for _ in range(adapter_depth)] + [0] * (depth - adapter_depth)
        config['prefix_dim'] = [random.choice(choices['prefix_dim']) for _ in range(prefix_depth)] + [0] * (depth - prefix_depth)

    else:
        if is_visual_prompt_tuning:
            config['visual_prompt_dim'] = [choices['super_prompt_tuning_dim']] * (depth)
        else:
            config['visual_prompt_dim'] = [0] * (depth)
        
        if is_adapter:
             config['adapter_dim'] = [choices['super_adapter_dim']] * (depth)
        else:
            config['adapter_dim'] = [0] * (depth)

        if is_LoRA:
            config['lora_dim'] = [choices['super_LoRA_dim']] * (depth)
        else:
            config['lora_dim'] = [0] * (depth)

        if is_prefix:
            config['prefix_dim'] = [choices['super_prefix_dim']] * (depth)
        else:
            config['prefix_dim'] = [0] * (depth)
        
    return config

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    amp: bool = True, teacher_model: torch.nn.Module = None, matching_loss = False,
                    teach_loss: torch.nn.Module = None, choices=None, mode='super', retrain_config=None,is_visual_prompt_tuning=False,is_adapter=False,is_LoRA=False,is_prefix=False):
    model.train()
    criterion.train()

    # set random seed
    random.seed(epoch)

    # warm_up_epochs = 100
    # num_steps = int(warm_up_epochs * len(data_loader))         # 100 epoch 
    # max_scale = 0.3

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    if mode == 'retrain':
        config = retrain_config
        model_module = unwrap_model(model)
        print(config)
        model_module.set_sample_config(config=config)
        print(model_module.get_sampled_params_numel(config))

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # if epoch < warm_up_epochs:
        #     current_step = epoch * len(data_loader) + i
        #     lamda = current_step / num_steps * max_scale
        #     model.lamda = lamda
        # else:
        #     model.lamda = max_scale

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # sample random config
        if mode == 'super':
            # sample
            config = sample_configs(choices=choices,is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter,is_LoRA=is_LoRA,is_prefix=is_prefix)
            # print("current iter config: {}".format(config))
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        elif mode == 'retrain':
            config = retrain_config
            model_module = unwrap_model(model)
            model_module.set_sample_config(config=config)
        
        outputs = model(samples, matching_loss)
        preds = outputs[0]
        prompt_matching_loss = outputs[2]
        loss = criterion(preds, targets) + 0.5 * prompt_matching_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # print("Prompt matching loss is : {}".format(prompt_matching_loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





@torch.no_grad()
def evaluate(data_loader, model, device, amp=True, choices=None, mode='super', retrain_config=None, save_feature=False,
             is_visual_prompt_tuning=False, is_adapter=False, is_LoRA=False, is_prefix=False, output_dir=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    if mode == 'super':
        config = sample_configs(choices=choices, is_visual_prompt_tuning=is_visual_prompt_tuning,is_adapter=is_adapter, is_LoRA=is_LoRA,is_prefix=False)
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)
    else:
        config = retrain_config
        model_module = unwrap_model(model)
        model_module.set_sample_config(config=config)

    print("sampled model config: {}".format(config))
    parameters = model_module.get_sampled_params_numel(config)
    print("sampled model parameters: {}".format(parameters))

    intermidiate_feats = []
    intermidiate_attns = []
    total_labels = []
    total_images = []

    # print("lambda: ", model.lamda)
    # print(model.prompt_scale)
    # print(torch.nn.functional.sigmoid(model.prompt_scale))

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        output, feats, attns = model(images)
        loss = criterion(output, target)

        '''
        feats = np.stack([x.detach().cpu().numpy() for x in feats])
        intermidiate_feats.append(feats)
        
        attns = np.stack([x.detach().cpu().numpy() for x in attns])
        intermidiate_attns.append(attns)
        total_labels.append(target.detach().cpu().numpy())
        total_images.append(images.detach().cpu().numpy())
        '''
        # np.save(f'intermidate_feats.npy', feats)
        # np.save(f'intermidate_image.npy', images.detach().cpu().numpy())
        # raise NotImplementedError

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # if save_feature:
    #     intermidiate_feats = np.concatenate(intermidiate_feats[:3], axis=1)
    #     intermidiate_attns = np.concatenate(intermidiate_attns[:3], axis=1)
    #     total_labels = np.concatenate(total_labels[:3])
    #     total_images = np.concatenate(total_images[:3])
    #     print("intermidiate feats shape :", intermidiate_feats.shape)
    #     np.savez(f'{output_dir}/feats_for_viz', total_images=total_images, intermidiate_attns=intermidiate_attns,
    #              total_labels=total_labels)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f} indices {indices}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, 
                  losses=metric_logger.loss, indices=model.visual_prompt_indices))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
