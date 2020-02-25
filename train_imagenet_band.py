from __future__ import print_function

import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse
import resnet_imgnt as resnet

from prog_bar import progress_bar
import utils_band as utils

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--band_size', default=25, type=int, help='band size')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--regularization', default=0.0005, type=float, help='weight decay')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--end_epoch', default = 199, type=int, help='end_epoch')
#parser.add_argument('--trainpath', default='imagenet-train/train', type=str, help='Path to ImageNet training set')
#parser.add_argument('--valpath', default='imagenet-val/val', type=str, help='Path to ImageNet validation set')
parser.add_argument('--trainpath', default='imagenet-train/train', type=str, help='Path to ImageNet training set')
parser.add_argument('--valpath', default='imagenet-val/val', type=str, help='Path to ImageNet validation set')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_file = f'./{checkpoint_dir}/imagenet_one_band_lr_{args.lr}_regularization_{args.regularization}_model_{args.model}__band_{args.band_size}_epoch_{{}}.pth'

print("==> Checkpoint directory", checkpoint_dir)
print("==> Saving to", checkpoint_file)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
traindir = args.trainpath
valdir = args.valpath
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=196, shuffle=True,
    num_workers=2, pin_memory=True)

nomtestloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=196, shuffle=False,
    num_workers=2, pin_memory=True)

# Model
print('==> Building model..')
if (args.model == 'resnet50'):
    net = resnet.resnet50()
elif (args.model == 'resnet18'):
    net = resnet.resnet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
    resume_file = '{}/{}'.format(checkpoint_dir, args.resume)
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    checkpoint_file = './{}/imagenet_one_band_lr_{}_regularization_{}_band_{}_epoch_{}_resume_{}.pth'.format(checkpoint_dir, args.lr,args.regularization, args.band_size,'{}', args.resume)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.regularization)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total_epsilon = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(utils.random_mask_batch_one_sample(inputs, args.band_size, reuse_noise=True))
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total += targets.size(0)

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
def test_nominal(epoch):
    print('\nEpoch: %d' % epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(nomtestloader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            #breakpoint()
            outputs = net(utils.random_mask_batch_one_sample(inputs, args.band_size, reuse_noise=True))
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
            total += targets.size(0)
            progress_bar(batch_idx, len(nomtestloader), 'Nominal Test Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch
    }
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_file.format(epoch))
    torch.save(state, "checkpoints/imagenet_" + str(epoch)+".pth")



for epoch in range(start_epoch,  args.end_epoch+1):
    train(epoch)
    test_nominal(epoch)
