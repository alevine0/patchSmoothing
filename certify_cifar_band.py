import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pytorch_cifar.models.resnet as resnet

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import utils_band as utils
from prog_bar import progress_bar
parser = argparse.ArgumentParser(description='PyTorch CIFAR Certification')


parser.add_argument('--band_size', default=4, type=int, help='size of each smoothing band')
parser.add_argument('--size_to_certify', default=5, type=int, help='size_to_certify')
parser.add_argument('--checkpoint', help='checkpoint')
parser.add_argument('--threshhold', default=0.2, type=float, help='threshold for smoothing abstain')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--test', action='store_true', help='Use test set (vs validation)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'



transform_test = transforms.Compose([
    transforms.ToTensor(),
  #  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
val_indices = torch.load('validation.t7')
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
if (args.test):
    val_indices = list(set(range(len(testset))) - set(val_indices.numpy().tolist()))
testloader = torch.utils.data.DataLoader(torch.utils.data.Subset(testset,val_indices), batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
if (args.model == 'resnet50'):
    net = resnet.ResNet50()
elif (args.model == 'resnet18'):
    net = resnet.ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
#assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
resume_file = '{}/{}'.format(checkpoint_dir, args.checkpoint)
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)
net.load_state_dict(checkpoint['net'])
net.eval()


def test():
    global best_acc
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            predictions,  certyn = utils.predict_and_certify(inputs, net,args.band_size, args.size_to_certify, 10,threshold =  args.threshhold)

            correct += (predictions.eq(targets)).sum().item()
            cert_correct += (predictions.eq(targets) & certyn).sum().item()
            cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()


          #  progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d) Cert: %.3f%% (%d/%d)'
           #     %  ((100.*correct)/total, correct, total, (100.*cert_correct)/total, cert_correct, total))
    print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(args.threshhold))
    print('Total images: ' + str(total))
    print('Correct: ' + str(correct) + ' (' + str((100.*correct)/total)+'%)')
    print('Certified Correct class: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
    print('Certified Wrong class: ' + str(cert_incorrect) + ' (' + str((100.*cert_incorrect)/total)+'%)')



test()
