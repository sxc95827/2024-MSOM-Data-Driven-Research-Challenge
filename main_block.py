# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import copy
import os
import random
import sys
import time
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torch.distributed as dist
from torchvision import transforms, models, datasets

# from block_utils import chopped_resnet50, change_trainable_module, chop_network, count_num_trainable_params
from model import block_resnet50
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument('data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=20, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=0.2, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=3.9e-3, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--scale-loss', default=1 / 32, type=float,
                    metavar='S', help='scale the loss')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--filter-size', default=3, type=int, metavar='N',
                    help='filter size to be used for reduction')
parser.add_argument('--noise-type', default="none", choices=["none", "hw", "c", "all"],
                    help='noise type to be used for addition')
parser.add_argument('--noise-std', default=None, type=float, metavar='N',
                    help='noise std to be used in case noise is enabled')
parser.add_argument('--checkpoint-dir', default='./checkpoint_adaptive_pool/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--rge-step-size', default=0.1, type=float,
                    metavar='S', help='rge step size')
parser.add_argument('--rge-lr', default=0.1, type=float,
                    metavar='S', help='rge learning rate')
parser.add_argument('--rge-optimizer', type=str, choices=['sgd', 'sign'],default='sgd',
                    help = ' we can choose the optimizer for rge ZO')
parser.add_argument('--rge-block', type=int, choices=[0,1,2,3],default=0,
                    help = 'we can choose which block to adopt rge')
parser.add_argument('--seed', type=int, default=324823217)

def set_seed(seed):
    """
    多卡应该设置相同的random seed，形成相同的扰动。这样不同卡之间的base loss与perturb loss
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_blockwise_params(model: nn.Module):
    """
    返回一个 dict: 0，1，2，3对应resnet block 0,1,2,3; p对应的是pool_conv + projector
    用于后续选择哪个block或者projector应用零阶
    """
    blockwise_dict = {0: [], 1: [], 2: [], 3: [], 'p0':[],'p1':[],'p2':[],'p3':[] }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # backbone layers
        if "layer1" in name:
            blockwise_dict[0].append((name, param))
        elif "layer2" in name:
            blockwise_dict[1].append((name, param))
        elif "layer3" in name:
            blockwise_dict[2].append((name, param))
        elif "layer4" in name:
            blockwise_dict[3].append((name, param))
        # projector_list
        elif "projector_list.0" in name:
            blockwise_dict['p0'].append((name, param))
        elif "projector_list.1" in name:
            blockwise_dict['p1'].append((name, param))
        elif "projector_list.2" in name:
            blockwise_dict['p2'].append((name, param))
        elif "projector_list.3" in name:
            blockwise_dict['p3'].append((name, param))
        # pool_conv_block
        elif "block_1" in name:
            blockwise_dict['p0'].append((name, param))
        elif "block_2" in name:
            blockwise_dict['p1'].append((name, param))
        elif "block_3" in name:
            blockwise_dict['p2'].append((name, param))
        elif "block_4" in name:
            blockwise_dict['p3'].append((name, param))
        else:
            blockwise_dict[0].append((name, param))

    return blockwise_dict

@torch.no_grad()
def rge_step_block0(
    model: nn.Module,
    base_loss: torch.Tensor,  
    y1: torch.Tensor,
    y2: torch.Tensor,
    step_size: float,
    args,
):
    """
    ZO RGE implementation 
    可以用blockwise_dict选择用于零阶的参数
    目前可以用rge-block选一个用于零阶的block
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 0) 获取参数分组
    blockwise_dict = get_blockwise_params(model)

    # 1) 备份所有可训练参数
    param_backup = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            param_backup[name] = p.data.clone()

    # 2) 给 指定 block 的参数做随机扰动，并计算差分
    grads = {}  # 用来存储 block 的“零阶梯度”
    perturb_dict = {}
    block_num = args.rge_block
    for param_name, param in blockwise_dict[block_num]:
        # 初始化 0 张量存梯度
        grads[param_name] = torch.zeros_like(param_backup[param_name])

        # 生成归一化扰动
        perturb = torch.randn_like(param)
        perturb /= (torch.norm(perturb) + 1e-8)
        perturb *= step_size

        # 保存一下扰动，后面要求 diff
        perturb_dict[param_name] = perturb

        # 应用扰动
        param.data.add_(perturb)

    # 2.1) forward => 得到扰动后的 loss_list
    perturbed_loss_list = model.forward(y1, y2)
    perturbed_loss = perturbed_loss_list[block_num]

    # 2.2) 计算 block 的 diff
    diff = (perturbed_loss - base_loss) / step_size

    # 2.3) 将 diff_0 * perturb 累加到 grads
    for param_name, _ in blockwise_dict[block_num]:
        grads[param_name] += diff * perturb_dict[param_name]

    # 2.4) 恢复参数
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.data.copy_(param_backup[name])

    # 3) 将估计的梯度存储到参数的 gradient 属性中
    for param_name, param in blockwise_dict[block_num]:
        if not hasattr(param, 'gradient'):
            param.gradient = torch.zeros_like(param.data)
        else:
            param.gradient.zero_()
        param.grad = torch.zeros_like(param.grad)
        param.gradient.add_(grads[param_name])
        
    return

def main():
    args = parser.parse_args()
    assert args.noise_type == "none" or args.noise_std is not None

    args.ngpus_per_node = torch.cuda.device_count()
    
    # Initialize the distributed environment
    args.gpu = 0
    args.world_size = 1
    args.local_rank = 0
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1
    args.rank = int(os.getenv('RANK', 0))

    if "SLURM_NNODES" in os.environ:
        args.local_rank = args.rank % torch.cuda.device_count()
        print(f"SLURM tasks/nodes: {os.getenv('SLURM_NTASKS', 1)}/{os.getenv('SLURM_NNODES', 1)}")
    elif "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.getenv('LOCAL_RANK', 0))

    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    args.world_size = torch.distributed.get_world_size()
    assert int(os.getenv('WORLD_SIZE', 1)) == args.world_size
    
    print(f"Initializing the environment with {args.world_size} processes | Current process rank: {args.local_rank}")

    if args.rank == 0:
        print("Current checkpoint directory:", args.checkpoint_dir)
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    gpu = args.gpu
    torch.backends.cudnn.benchmark = True
    
    # set_seed(args.seed + args.local_rank)  # parallel 
    set_seed(args.seed) 
             
    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=False)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = LARS(trainable_params, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=exclude_bias_and_norm,
                     lars_adaptation_filter=exclude_bias_and_norm)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if torch.distributed.get_rank() == 0:
            print("Resuming model training from the last trained checkpoint...")
    else:
        start_epoch = 0

    dataset = datasets.ImageFolder(args.data / 'train', Transform())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    block_losses = {'block0': [], 'block1': [], 'block2': [], 'block3': []}
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        if epoch == 10:
            break
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
            # if step == 300:
            #     flag = 1
            #     break
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            lr = adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                loss_list = model.forward(y1, y2)

            for i, loss in enumerate(loss_list):
                scaler.scale(loss).backward()

            rge_step_block0(
                model=model,
                base_loss=loss_list[args.rge_block],    # 直接复用这次 forward 的 loss
                y1=y1, y2=y2,
                step_size=args.rge_step_size,   # 零阶步长
                args=args,
            )

            # 用零阶计算的梯度覆盖一阶得到的梯度
            for name, p in model.named_parameters():
                # 这里实现了零阶的sgd 和 sign
                if hasattr(p, 'gradient') and p.requires_grad:
                    if args.rge_optimizer == 'sgd':
                        p.data.sub_(args.rge_lr * p.gradient)
                    elif args.rge_optimizer == 'sign':
                        p.data.sub_(args.rge_lr * p.gradient.sign())

            scaler.step(optimizer)
            scaler.update()

            if step % args.print_freq == 0:
                for loss in loss_list:
                    torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=[loss.item() for loss in loss_list],
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        # if flag == 1:
        #     break
        if args.rank == 0:
            # save checkpoint
            for i in range(0, len(loss_list)):
                block_losses[f'block{i}'].append(loss_list[i].item()) 
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
    if args.rank == 0:
        # save final model
        torch.save(model.module.backbone.state_dict(),
                   args.checkpoint_dir / 'resnet50.pth')
        # 绘制每个block的loss的图像
        plt.figure(figsize=(10, 6))
        for block, losses in block_losses.items():
            plt.plot(losses, label=f'{block} Loss') 
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss for Different Blocks during Training')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.checkpoint_dir / 'block_losses.png')  
        plt.close()

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.learning_rate * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.backbone = block_resnet50(zero_init_residual=True, filter_size=args.filter_size,
                                       noise_type=args.noise_type, noise_std=args.noise_std)
        self.backbone.fc = nn.Identity()
        
        # self.backbone = models.resnet50(zero_init_residual=True)
        # self.backbone.fc = nn.Identity()
        
        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        # self.projector = nn.Sequential(*layers)
        
        # Separate projection head for each output
        self.projector_list = nn.ModuleList()
        projector = nn.Sequential(*layers)
        for i in range(4):
            self.projector_list.append(copy.deepcopy(projector))
        
        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        out_y1 = self.backbone(y1)
        out_y2 = self.backbone(y2)
        
        loss_list = []
        for projector, rep_y1, rep_y2 in zip(self.projector_list, out_y1, out_y2):
            z1 = projector(rep_y1)
            z2 = projector(rep_y2)

            # empirical cross-correlation matrix
            c = self.bn(z1).T @ self.bn(z2)

            # sum the cross-correlation matrix between all gpus
            c.div_(self.args.batch_size)
            torch.distributed.all_reduce(c)

            # use --scale-loss to multiply the loss by a constant factor
            # see the Issues section of the readme
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum().mul(self.args.scale_loss)
            off_diag = off_diagonal(c).pow_(2).sum().mul(self.args.scale_loss)
            loss = on_diag + self.args.lambd * off_diag
            loss_list.append(loss)
        
        return loss_list


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if g['weight_decay_filter'] is None or not g['weight_decay_filter'](p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if g['lars_adaptation_filter'] is None or not g['lars_adaptation_filter'](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def exclude_bias_and_norm(p):
    return p.ndim == 1


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
