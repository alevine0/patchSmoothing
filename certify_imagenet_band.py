import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import resnet_imgnt as resnet
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import utils_band as utils
from prog_bar import progress_bar
parser = argparse.ArgumentParser(description='PyTorch ImageNet Certification')


parser.add_argument('--band_size', default=25, type=int, help='size of each smoothing band')
parser.add_argument('--size_to_certify', default=42, type=int, help='size_to_certify')
parser.add_argument('--checkpoint', help='checkpoint')
parser.add_argument('--threshhold', default=0.2, type=float, help='threshold for smoothing abstain')
parser.add_argument('--model', default='resnet50', type=str, help='model')
parser.add_argument('--valpath', default='imagenet-val/val', type=str, help='Path to ImageNet validation set')
parser.add_argument('--skip', default=50,type=int, help='Number of images to skip')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
valdir = args.valpath
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

valset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        normalize,
    ]))

skips = list(range(0, len(valset), args.skip))

valset_1 = torch.utils.data.Subset(valset, skips)
testloader = torch.utils.data.DataLoader(
    valset_1,
    batch_size=100, shuffle=False,
    num_workers=2, pin_memory=True)


# Model
print('==> Building model..')
checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
if (args.model == 'resnet50'):
    net = resnet.resnet50()
elif (args.model == 'resnet18'):
    net = resnet.resnet18()

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
            predictions,  certyn = utils.predict_and_certify(inputs, net,args.band_size, args.size_to_certify, 1000,threshold =  args.threshhold)

            correct += (predictions.eq(targets)).sum().item()
            cert_correct += (predictions.eq(targets) & certyn).sum().item()
            cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()


            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d) Cert: %.3f%% (%d/%d)'  %  ((100.*correct)/total, correct, total, (100.*cert_correct)/total, cert_correct, total))
    print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(args.threshhold))
    print('Certifying For Patch ' +str(args.size_to_certify) + '*'+str(args.size_to_certify))
    print('Total images: ' + str(total))
    print('Correct: ' + str(correct) + ' (' + str((100.*correct)/total)+'%)')
    print('Certified Correct class: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
    print('Certified Wrong class: ' + str(cert_incorrect) + ' (' + str((100.*cert_incorrect)/total)+'%)')



test()
