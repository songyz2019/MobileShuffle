""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
import random
import numpy as np

def get_network(args):
    from mobile_shuffle import mobileshuffle
    if args.net == 'mobileshuffle' and args.dataset == 'cifar100':
        net = mobileshuffle(variant="s0", num_classes=100, inference_mode=False)
    elif args.net == 'mobileshuffle' and args.dataset == 'tinyimagenet':
        net = mobileshuffle(variant="s0", num_classes=200, inference_mode=False)
    elif args.net == 'mobileshuffle' and args.dataset == 'GF3':
        net = mobileshuffle(variant="s0", num_classes=8, inference_mode=False)
    return net
# 设置随机种子以保证结果可复现
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label
    
def get_dataloader(args, mean, std, num_workers=64, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    set_random_seed(2024)
    if args.dataset == 'cifar100': # 5:1的比例
        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        train_loader = DataLoader(train_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=args.b)
        test_loader = DataLoader(test_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=args.b)
    elif args.dataset == 'tinyimagenet': # 10：1的比例
        from create_DataLoader import LoadData
        train_path = '/remote-home/xt/AMC/amc-master/scripts/train.txt'
        val_path = '/remote-home/xt/AMC/amc-master/scripts/val.txt'
        train_dataset = LoadData(txt_path=train_path,train_flag=True)
        test_dataset = LoadData(txt_path=val_path,train_flag=False)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.b,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.b,
                                               num_workers=num_workers,
                                               pin_memory=True,
                                               shuffle=False)
    elif args.dataset == 'GF3': # 3：1的比例
        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
        full_dataset = ImageFolder(root='/attached/remote-home2/xt/DCNN-master/GF3_ChinaSciencePaper_Classify_png/')
        # 分割数据集为训练集和测试集
        train_indices, test_indices = [], []
        train_ratio = 0.75
        for class_index in range(len(full_dataset.classes)):
            '遍历整个数据集的索引,找到和该类索引相同的那一类的数据,随后shuffle并记录索引到train和test'
            class_indices = [i for i, (path, index) in enumerate(full_dataset.samples) if index == class_index]
            np.random.shuffle(class_indices)
            split = int(train_ratio * len(class_indices))
            train_indices.extend(class_indices[:split])
            test_indices.extend(class_indices[split:])
        train_dataset = CustomDataset(full_dataset, train_indices, transform=transform_train)
        test_dataset = CustomDataset(full_dataset, test_indices, transform=transform_test)
        train_loader = DataLoader(train_dataset, batch_size=args.b, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.b, num_workers=num_workers, shuffle=False)

    return train_loader, test_loader


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
