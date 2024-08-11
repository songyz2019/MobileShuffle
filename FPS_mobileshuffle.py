import argparse
import json
import torch
from torch.utils.data import DataLoader
import time
import engine.utils as utils
import torch.nn.functional as F

# mobilenet以及tiny_imagenet的接口
from models.mobilenet import mobilenet
from models.MobileNet_for_224 import MobileNet
from models.mobilenetv2 import mobilenetv2


# from models.mobileone_channel_shuffle1 import mobileone
# from models.mobileone_channel_shuffle import mobileone
# from models.mobile.mobile_shuffle import mobileone # 改进后

from models.mobileone import mobileone #原mobileone
# device = torch.device('cpu')
device = torch.device(f'cuda:{4}')
torch.cuda.set_device(device)

# model = mobilenet(alpha=1, class_num=200).to(device)
model = MobileNet(n_class=200).to(device)
# model = mobileone(variant="s0", inference_mode=True, num_classes=200).to(device) 
# model = MobileNet(n_class=200, profile='normal')
# model = MobileNet(n_class=200, profile='0.75flops')

# model = MobileNetV2(n_class=200)
# model = resnet56()


# print(model)
random_input = torch.randn(1,3,224,224).to(device)
from thop import profile
from thop import clever_format
flops, params = profile(model, inputs=(random_input, ))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)



def obtain_avg_forward_time(input, model, repeat=10):

    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time, output


def eval(model, test_loader, device=None):
    avg = []
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            start = time.time()
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
            avg_infer_time = (time.time() - start) / 128
            print("time:{}".format(avg_infer_time))
            print("FPS:{}".format(1/avg_infer_time))
            avg.append(avg_infer_time)
    avg_infer_time = sum(avg) / len(avg)
    return avg_infer_time


import registry
num_classes, train_loader, test_loader, input_size = registry.get_dataset(
    'tiny_imagenet', data_root="data")

avg_infer_time = eval(model, test_loader, device = device)
print("avg_time:{}".format(avg_infer_time))
print("avg_FPS:{}".format(1/avg_infer_time))