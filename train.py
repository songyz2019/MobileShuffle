import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from torch.nn.functional import cross_entropy
from model import MobileShuffle
from dataloader import FusarShip
# dataset
# trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=T.Compose([
#     T.Resize((224,224)),
#     T.RandomHorizontalFlip(),
#     T.RandomRotation(15),
#     T.ToTensor(),
#     T.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
# ]))
trainset = FusarShip()

# Model and Optimizer
model = MobileShuffle(num_classes = len(trainset.CLASS_LIST), inference_mode = False, width_multipliers= [2.0, 2.0, 2.0, 2.0],num_conv_branches= 4, num_blocks_per_stage= [2, 4, 6, 1],use_se= True).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(50):
    for x, y in DataLoader(trainset, batch_size=64, shuffle=True):
        # Move x and y to GPU
        x, y = x.cuda(), y.cuda()

        # Forward
        optimizer.zero_grad()
        y_hat = model(x)
        loss = cross_entropy(y_hat, y)

        # Backward
        loss.backward()
        optimizer.step()

        # Log
        print("epoch = %d, loss = %f" % (epoch, loss))

# Save model
torch.save(model.state_dict(), 'model.pt')