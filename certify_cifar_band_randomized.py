from __future__ import print_function

import utils_band_randomized as utils

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import pytorch_cifar.models.resnet as resnet

import os
import argparse

from prog_bar import progress_bar

parser = argparse.ArgumentParser(description='Certificate Evaluation')
parser.add_argument('--band_size',  required=True, type=int, help='size of band to keep')
parser.add_argument('--attack_size',  required=True, type=int, help='size of attack')
parser.add_argument('--model', default='resnet18', type=str, help='model')

parser.add_argument('--checkpoint',  required=True, help='checkpoint to certify')
parser.add_argument('--alpha', default=0.05, type=float, help='Certify to 1-alpha probability')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--predsamples', default=1000, type=int, help='samples for prediction')
parser.add_argument('--boundsamples', default=10000, type=int, help='samples for bound')
parser.add_argument('--test', action='store_true', help='Use test set (vs validation)')

args = parser.parse_args()
checkpoint_dir = 'checkpoints'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

val_indices = torch.load('validation.t7')
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if (args.test):
    val_indices = list(set(range(len(testset))) - set(val_indices.numpy().tolist()))
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,val_indices), batch_size=2, shuffle=False, num_workers=2)

# Model
print('==> Building model..')

if (args.model == 'resnet50'):
    net = resnet.ResNet50()
elif (args.model == 'resnet18'):
    net = resnet.ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.checkpoint)
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