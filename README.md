# (De)Randomized Smoothing for Certifiable Defense against Patch Attacks

Code for the paper [(De)Randomized Smoothing for Certifiable Defense against Patch Attacks](http://arxiv.org/abs/2002.10733) by Alexander Levine and Soheil Feizi.

Files are provided for training and evaluation of classifiers robust to patch attacks on MNIST, CIFAR-10, and ImageNet datasets.

For MNIST and CIFAR-10, both the "block" and "band" methods are supported. On MNIST, there is also a supported "row" method.

For MNIST and CIFAR-10, there are additional certification options, using just the "top one" classification or (in the case of band smoothing) using randomized (as opposed to derandomized) smoothing. Additionally, multi-block  and  multi-band smoothing is supported for MNIST.

ImageNet code expects the ILSVRC2012 training and validation sets to be in the directories 'imagenet-train/train' and 'imagenet-val/val', respectively. This can be changed using the '--trainpath' and '--valpath' options.

Explanation of files: (substitute 'mnist' for 'cifar' or 'imagenet' appropriately;  similarly substitute 'block' for 'band')

```
- train_mnist_band.py # Will train the base classifier, and save the model to the 'checkpoints' directory

- certify_mnist_band.py # Will load a model from 'checkpoints', and calculate and print clean and certified accuracies. The '--test' option will use the test set, rather than the validation set.
```

Example Usage: 

```
python3 train_mnist_band.py --band_size 4 --lr 0.01 --end_epoch 199
python3 train_mnist_band.py --band_size 4 --lr 0.001 --end_epoch 399 --resume mnist_one_band_lr_0.01_regularization_0.0005_band_4_epoch_199.pth
python3 certify_mnist_band.py --band_size 4 --size_to_certify 5 --checkpoint mnist_one_band_lr_0.001_regularization_0.0005_band_4_epoch_399_resume_mnist_one_band_lr_0.01_regularization_0.0005_band_4_epoch_199.pth.pth
python3 certify_mnist_band.py --band_size 4 --size_to_certify 5 --test --checkpoint mnist_one_band_lr_0.001_regularization_0.0005_band_4_epoch_399_resume_mnist_one_band_lr_0.01_regularization_0.0005_band_4_epoch_199.pth.pth
```

There is also code to attack column-smoothed CIFAR-10 models:

```
attack_cifar_band.py  -- Patch attack on smoothed classifier
attack_cifar_band_linf.py  -- L-infinity attack on smoothed classifier
attack_cifar_baseline.py  -- Patch attack on baseline classifier
attack_cifar_band_linf.py  -- L-infinity attack on baseline classifier
```

Attributions:
- Code in the `pytorch_cifar` directory is from https://github.com/kuangliu/pytorch-cifar, with slight modification to allow for 6-channel input.
- The file `resnet_imgnt.py` is modified from the PyTorch torchvision implementation of ResNet (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py), again with slight modification to allow for 6-channel input.
- Attack code is modified from https://github.com/Ping-C/certifiedpatchdefense.
