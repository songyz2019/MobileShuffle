#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader 
from GF3.create_DataLoader_GF3 import LoadData
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-GPU_num', type=int, default=0, help='GPU selected')
    args = parser.parse_args()

    net = get_network(args)

    device = torch.device(f'cuda:{args.GPU_num}')
    torch.cuda.set_device(device)
    

    # 设置随机种子以保证结果可复现
    import random
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
    
    #data preprocessing:
    set_random_seed(2024)
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

    train_loader = DataLoader(train_dataset, batch_size=args.b, num_workers=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.b, num_workers=64, shuffle=False)



    # test_dataset = LoadData(txt_path="./GF3/GF3_test.txt",train_flag=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    #                                            batch_size=args.b,
    #                                            num_workers=4,
    #                                            shuffle=False)

    net.load_state_dict(torch.load(args.weights))
    net.to(device)
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
