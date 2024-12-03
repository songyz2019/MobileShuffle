import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torch.nn.functional import cross_entropy
from model import MobileShuffle
from dataloader import FusarShip

# dataset
# testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=T.Compose([
#     T.Resize((224,224)),
#     T.RandomHorizontalFlip(),
#     T.RandomRotation(15),
#     T.ToTensor(),
#     T.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
# ]))
testset = FusarShip(root_dir='data/fusar_ship')

# Model and Optimizer
model = MobileShuffle(num_classes = len(testset.CLASS_LIST), inference_mode = False, width_multipliers= [2.0, 2.0, 2.0, 2.0],num_conv_branches= 4, num_blocks_per_stage= [2, 4, 6, 1],use_se= True).cuda()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Training
n_correct = 0
n_incorrect = 0
for x, y in DataLoader(testset, batch_size=1, shuffle=False):
    # Move x and y to GPU
    x, y = x.cuda(), y.cuda()

    # Forward
    y_hat = model(x)

    # Log
    if torch.argmax(y) == torch.argmax(y_hat):
        n_correct += 1
    else:
        n_incorrect += 1
    print("correct=%d  incorrect=%d" % (n_correct, n_incorrect))