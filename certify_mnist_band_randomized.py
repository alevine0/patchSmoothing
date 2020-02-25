from __future__ import print_function

import utils_band_randomized as utils

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from prog_bar import progress_bar

parser = argparse.ArgumentParser(description='L0 Certificate Evaluation')
parser.add_argument('--band_size',  required=True, type=int, help='size of band to keep')
parser.add_argument('--attack_size',  required=True, type=int, help='size of attack')

parser.add_argument('--model',  required=True, help='checkpoint to certify')
parser.add_argument('--alpha', default=0.05, type=float, help='Certify to 1-alpha probability')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--predsamples', default=1000, type=int, help='samples for prediction')
parser.add_argument('--boundsamples', default=10000, type=int, help='samples for bound')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
radii_dir = 'radii'
if not os.path.exists('./radii'):
    os.makedirs('./radii')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

net = nn.Sequential(
        nn.Conv2d(2, 64, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 4, stride=2, padding=1),
        nn.ReLU(),
        Flatten(),
        nn.Linear(128*7*7,500),
        nn.ReLU(),
        nn.Linear(500,100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.model)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])

net.eval()
all_batches = []
for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        #breakpoint()
        batch_radii = utils.certify(inputs, targets, net, args.alpha, args.band_size, args.attack_size, args.predsamples, args.boundsamples )
        all_batches.append(batch_radii)
        progress_bar(batch_idx, len(testloader))
out = torch.cat(all_batches)
print('band size: ' + str(args.band_size))
print('certify correct: ' + str(float((out == 1).sum())/out.shape[0]))
print('certify wrong: ' + str(float((out == -1).sum())/out.shape[0]))