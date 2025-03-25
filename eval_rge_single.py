# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# rewrite the evaluate.py with ZO rge

from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import urllib
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch import nn, optim
from torchvision import datasets, transforms

from model import block_resnet50  

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet (Single-GPU RGE version)')
parser.add_argument('data',default='/media/HDD5/personal_files/datasets/imagenet100', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('pretrained',default='/home/zwx1350276/jackzhan/blockwise_ssl_zo/checkpoint_block_sim_expand_pool_fs1_noise_0.25_all/resnet50.pth', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of training set in percent')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=0.3, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint-dir', default='/home/zwx1350276/jackzhan/blockwise_ssl_zo/checkpoint_rge_supervised_4/block-4/lincls/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--seed', default=None, type=int, metavar='N',
                    help='seed')
parser.add_argument('--num-blocks', default=4, type=int, metavar='N',
                    help='number of blocks to be trained in the network')
parser.add_argument('--filter-size', default=1, type=int, metavar='N',
                    help='filter size to be used for reduction')
parser.add_argument('--init-step-size', default=1e-3, type=float, help='initial RGE step size') 
parser.add_argument('--step-size-decay', default=1.0, type=float, help='multiplicative factor for step size decay per epoch')
def main():
    args = parser.parse_args()
    if args.seed is not None:
        print(f"Using seed: {args.seed}")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(seed=args.seed)
        random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
    print(' '.join(sys.argv))
    print(' '.join(sys.argv), file=stats_file)

    torch.backends.cudnn.benchmark = True

    model = block_resnet50(zero_init_residual=True, filter_size=args.filter_size).to(device)
    model.fc = nn.Identity()
    print(f"Loading checkpoint: {args.pretrained}")
    state_dict = torch.load(args.pretrained, map_location='cpu')
    model.load_state_dict(state_dict)
    if args.weights == 'freeze':
        model.requires_grad_(False)

    classifier = nn.Linear(2048, 1000).to(device)
    classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.bias.data.zero_()

    criterion = nn.CrossEntropyLoss().to(device)

    param_groups = [dict(params=classifier.parameters(), lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, lr=0.0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    start_epoch = 0
    best_acc = argparse.Namespace(top1=0, top5=0)
    ckpt_path = args.checkpoint_dir / 'checkpoint.pth'
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        start_epoch = ckpt['epoch']
        best_acc = ckpt['best_acc']
        classifier.load_state_dict(ckpt['classifier'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    traindir = args.data / 'train'
    valdir   = args.data / 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    )
    if args.train_percent in {1, 10}:
        # Subset logic
        lines = urllib.request.urlopen(
            f'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/{args.train_percent}percent.txt'
        ).readlines()
        subset_samples = []
        for fname in lines:
            fname = fname.decode().strip()
            cls = fname.split('_')[0]
            subset_samples.append((traindir/cls/fname, train_dataset.class_to_idx[cls]))
        train_dataset.samples = subset_samples

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    start_time = time.time()
    output_idx = args.num_blocks - 1 

    current_step_size = args.init_step_size 
    print(f"Initial RGE step_size = {current_step_size}, decay factor={args.step_size_decay}")


    for epoch in range(start_epoch, args.epochs):
        if args.weights == 'finetune':
            model.train()
        else:
            model.eval()

        current_lr = scheduler.get_last_lr()[0]

        # RGE Training
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # 1) base forward
            with torch.no_grad():
                feats_base = model(images)[output_idx]
                logits_base = classifier(feats_base)
                base_loss = criterion(logits_base, target)

            # 2) pertube classifier
            param_backup = {}
            perturb_dict = {}
            for name, p in classifier.named_parameters():
                param_backup[name] = p.data.clone()
                direction = torch.randn_like(p)
                direction /= (direction.norm() + 1e-8)
                direction *= current_step_size
                perturb_dict[name] = direction
                p.data.add_(direction)

            # 3) forward again
            with torch.no_grad():
                feats_perturbed = model(images)[output_idx]
                logits_perturbed = classifier(feats_perturbed)
                perturbed_loss = criterion(logits_perturbed, target)

            # 4) compute diff & update
            diff_val = (perturbed_loss.item() - base_loss.item()) / current_step_size
            for name, p in classifier.named_parameters():
                p.data.copy_(param_backup[name])  # restore
                grad_est = diff_val * perturb_dict[name] 
                # if step % args.print_freq == 0:
                #     g_mean = grad_est.mean().item()
                #     g_std  = grad_est.std().item()
                #     g_min  = grad_est.min().item()
                #     g_max  = grad_est.max().item()
                #     g_norm = grad_est.norm().item()
                #     print(f"[Epoch {epoch} Step {step}] Param: {name} | "
                #         f"mean={g_mean:.6f}, std={g_std:.6f}, "
                #         f"min={g_min:.6f}, max={g_max:.6f}, norm={g_norm:.6f}")

                #     grad_numpy = grad_est.view(-1).cpu().numpy()

                #     plt.figure(figsize=(6,4))
                #     plt.hist(grad_numpy, bins=50, density=True, alpha=0.7, color='blue')
                #     plt.title(f"Gradient distribution | {name} @ step {step}")
                #     plt.xlabel("Gradient value")

                #     plt.savefig(f"grad_{name}_epoch{epoch}_step{step}.png")
                #     plt.close()

                # p.data -= current_lr * grad_est
                p.data.sub_(current_lr * grad_est)


            # 5) log
            if step % args.print_freq == 0:
                stats = dict(
                    epoch=epoch, step=step,
                    lr_classifier=current_lr,
                    base_loss=base_loss.item(),
                    perturbed_loss=perturbed_loss.item(),
                    diff=diff_val,
                    time=int(time.time() - start_time)
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)

        # evaluate
        model.eval()
        top1, top5 = evaluate(model, classifier, val_loader, device, output_idx)
        best_acc.top1 = max(best_acc.top1, top1)
        best_acc.top5 = max(best_acc.top5, top5)
        eval_stats = dict(epoch=epoch, acc1=top1, acc5=top5,
                          best_acc1=best_acc.top1, best_acc5=best_acc.top5)
        print(json.dumps(eval_stats))
        print(json.dumps(eval_stats), file=stats_file)

        if args.weights == 'freeze':
            state_dict_ckpt = torch.load(args.pretrained, map_location='cpu')
            for k,v in model.state_dict().items():
                assert torch.equal(v.cpu(), state_dict_ckpt[k]), k

        scheduler.step()

        old_step_size = current_step_size 
        current_step_size *= args.step_size_decay 
        print(f"[Epoch {epoch}] step_size: {old_step_size} -> {current_step_size}", file=stats_file) 

        # save checkpoint
        ckpt_state = dict(
            epoch=epoch+1,
            best_acc=best_acc,
            classifier=classifier.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict()
        )
        torch.save(ckpt_state, ckpt_path)

def evaluate(model, classifier, val_loader, device, output_idx):
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = targets.to(device)
            feats = model(images)[output_idx]
            output = classifier(feats)
            acc1, acc5 = accuracy(output, targets, topk=(1,5))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))
    return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()



