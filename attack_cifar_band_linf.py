import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pytorch_cifar.models.resnet as resnet
import torchvision.datasets as datasets

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

import os
import argparse
import utils_band as utils
from prog_bar import progress_bar
from patch_attacker_parallelized import PatchAttacker
parser = argparse.ArgumentParser(description='PyTorch CIFAR Attack')


parser.add_argument('--band_size', default=4, type=int, help='size of each smoothing band')
parser.add_argument('--steps', default=50, type=int, help='Attack steps')
parser.add_argument('--step_size', default=0.5/255., type=float, help='Attack step size')
parser.add_argument('--epsilon', default=1./255., type=float, help='Attack epsilon')
parser.add_argument('--checkpoint', help='checkpoint')
parser.add_argument('--threshhold', default=0.3, type=float, help='threshold for smoothing abstain')
parser.add_argument('--model', default='resnet18', type=str, help='model')
parser.add_argument('--test', action='store_true', help='Use test set (vs validation)')
parser.add_argument('--skip', default=10,type=int, help='Number of images to skip')

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
val_indices = val_indices[::args.skip]
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
attacker = PatchAttacker(net, [0.,0.,0.],[1.,1.,1.], {
    'epsilon':args.epsilon,
    'random_start':False,
    'steps':args.steps,
    'step_size':args.step_size,
    'block_size':args.band_size,
    'threshhold': args.threshhold,
    'num_classes':10,
    'patch_l':32,
    'patch_w':32
})

def test():
    global best_acc
    correctclean = 0
    correctattacked =0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        save_image(make_grid(inputs, nrow=10),"baseline_linf_"+str(batch_idx)+".jpg")
        attacked = attacker.perturb(inputs,targets,float('inf'),random_count=1)
        predictionsclean,  certyn = utils.predict_and_certify(inputs, net,args.band_size, 1, 10,threshold =  args.threshhold)
        predictionsattacked,  certynx = utils.predict_and_certify(attacked, net,args.band_size, 1, 10,threshold =  args.threshhold)

        correctclean += (predictionsclean.eq(targets)).sum().item()
        correctattacked += (predictionsattacked.eq(targets)).sum().item()
        save_image(make_grid(attacked, nrow=10),"attacks_linf_"+str(batch_idx)+".jpg")


        progress_bar(batch_idx, len(testloader), 'Clean Acc: %.3f%% (%d/%d)  Adv Acc: %.3f%% (%d/%d)'  %  ((100.*correctclean)/total, correctclean, total,  (100.*correctattacked)/total, correctattacked, total))
    print('Using band size ' + str(args.band_size) + ' with threshhold ' + str(args.threshhold))
    print('L_inf attack magnitude' +str(args.epsilon))
    print('Total images: ' + str(total))
    print('Clean Correct: ' + str(correctclean) + ' (' + str((100.*correctclean)/total)+'%)')
    print('Attacked Correct: ' + str(correctattacked) + ' (' + str((100.*correctattacked)/total)+'%)')



test()
