from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from datasets.biased_mnist import ColourBiasedMNIST, SimpleConvNet
from datasets.celebA import *
import torchvision
from irene.utilities import *
from irene.core import *

from tqdm import tqdm
import random
import numpy as np

def train(model, args, train_loader):
    model.train()
    loss_task_tot = AverageMeter('Loss', ':.4e')
    loss_private_tot = AverageMeter('Loss', ':.4e')
    MI_tot = AverageMeter('Regu', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    private_top1 = AverageMeter('Acc@1', ':6.2f')
    tk0 = tqdm(train_loader, total=int(len(train_loader)), leave=True)
    for batch_idx, (data, target, private_label) in enumerate(tk0):
        data = data.to(args.device)
        target = target.to(args.device)
        private_label = private_label.to(args.device)
        output= model(data)
        output_private = args.PH()
        loss_task = args.criterion(output, target)
        loss_private = args.criterion(output_private, private_label)
        MI = args.MI(args.PH, private_label)
        loss_task_tot.update(loss_task.item(), data.size(0))
        loss_private_tot.update(loss_private.item(), data.size(0))
        MI_tot.update(MI.item(), data.size(0))
        loss = args.alpha * loss_task + loss_private + args.gamma * MI
        loss.backward()
        if ((batch_idx + 1) % args.batch_size_accumulation) == 0:
            args.optimizer.step()
            args.optimizer.zero_grad()
            args.PH_optimizer.step()
            args.PH_optimizer.zero_grad()
        acc1 = accuracy(output, target, topk=(1,))
        acc1_private = accuracy(output_private, private_label, topk=(1,))
        top1.update(acc1[0], data.size(0))
        private_top1.update(acc1_private[0], data.size(0))
        tk0.set_postfix(loss_task = loss_task_tot.avg, MI = MI_tot.avg, top1 = top1.avg.item(), top1_private=private_top1.avg.item())


def test(model, args, val_loader):
    model.eval()
    loss_task_tot = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    private_top1 = AverageMeter('Acc@1', ':6.2f')
    tk0 = tqdm(val_loader, total=int(len(val_loader)), leave=True)
    for batch_idx, (data, target, private_label) in enumerate(tk0):
        data = data.to(args.device)
        target = target.to(args.device)
        private_label = private_label.to(args.device)
        output= model(data)
        output_private = args.PH()
        loss_task = args.criterion(output, target)
        loss_task_tot.update(loss_task.item(), data.size(0))
        acc1 = accuracy(output, target, topk=(1,))
        acc1_private = accuracy(output_private, private_label, topk=(1,))
        top1.update(acc1[0], data.size(0))
        private_top1.update(acc1_private[0], data.size(0))
        tk0.set_postfix(loss_task = loss_task_tot.avg, top1 = top1.avg.item(), top1_private=private_top1.avg.item())
    return loss_task_tot.avg


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Information Removal at the bottleneck in Deep Neural Networks')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--batch-size-accumulation', type=int, default=1, metavar='N',
                        help='batchsize accumulation (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr_priv', type=float, default=0.1, metavar='LR',
                        help='learning rate for private head (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--rho', type=float, default=0.99)
    parser.add_argument('--target_celeba', type=str, default='Blond_Hair')
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--dev', default="cpu")
    parser.add_argument('--momentum-sgd', type=float, default=0.9, metavar='M',
                        help='Momentum')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--datapath', default='data/')
    parser.add_argument('--dataset', default='Bmnist')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    args.device = torch.device(args.dev)
    if args.dev != "cpu":
        torch.cuda.set_device(args.device)

    if args.dataset == 'celebA':
        model = torchvision.models.resnet18(pretrained=False).to(args.device)
        model.avgpool = nn.Sequential(model.avgpool, torch.nn.Identity().to(args.device))
        args.PH = Privacy_head(model.avgpool, nn.Sequential(torch.nn.Linear(512, 2))).to(args.device)
        args.MI = MI(device = args.device, privates=2)
        model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True).to(args.device)
        train_dataset = CelebA(args.datapath+"CelebA/", split='train', target=args.target_celeba, bias_attr='Male', unbiased=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        val_dataset = CelebA(args.datapath+"CelebA/", split='valid', target=args.target_celeba, bias_attr='Male', unbiased=True)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )  
    elif args.dataset == 'Bmnist':
        model = SimpleConvNet(num_classes=10).to(args.device)
        model.avgpool = nn.Sequential(model.avgpool, torch.nn.Identity().to(args.device))
        args.MI = MI(device = args.device)
        args.PH = Privacy_head(model.avgpool, nn.Sequential(torch.nn.Linear(128, 10))).to(args.device)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        train_dataset = ColourBiasedMNIST(args.datapath+"MNIST/", train=True, download=True, data_label_correlation=args.rho, n_confusing_labels=9, transform=transform)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        test_dataset = ColourBiasedMNIST(args.datapath+"MNIST/", train=False, data_label_correlation=0.1, n_confusing_labels=9, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    else:
        print('ERROR: Dataset not recognized!')
        return
    
    args.criterion = torch.nn.CrossEntropyLoss(reduction='mean').to(args.device)
    args.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
    args.PH_optimizer = torch.optim.SGD(args.PH.parameters(), lr=args.lr_priv, momentum=args.momentum_sgd, weight_decay=args.weight_decay)
    if (args.dataset == 'celebA'):
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(args.optimizer, mode='min', factor=0.1, patience=10, threshold=0, cooldown=5)
    else:
        sched = torch.optim.lr_scheduler.MultiStepLR(args.optimizer, milestones=[40, 60], gamma=0.1, verbose=True)
    for epoch in range(1, args.epochs+1):
        train(model, args, train_loader)
        with torch.no_grad():
            if (args.dataset == 'celebA') :
                sched.step(test(model, args, val_loader))
            else:
                sched.step()
            test(model, args, test_loader)
        if (args.dataset == 'celebA') :
            if args.optimizer.param_groups[0]['lr'] < 0.001: break;
    if args.dataset=='Bmnist':
        torch.save(model.state_dict(), 'models/'+args.dataset+'_'+str(args.rho)+'_'+str(args.gamma)+'_'+str(args.seed)+'.pth')
    elif args.dataset=='celebA':
        torch.save(model.state_dict(), 'models/'+args.dataset+'_'+args.target_celeba+'_'+str(args.alpha)+'_'+str(args.gamma)+'_'+str(args.seed)+'.pth')

if __name__ == '__main__':
    main()
