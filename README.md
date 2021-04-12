# Senior Thesis: "A Defense against Adversarial Patch Attacks on Neural Network Classifiers"
By Nicholas Sum

with [Chong Xiang](http://xiangchong.xyz/), [Saeed Mahloujifar](https://smahloujifar.github.io/), [Prattek Mittal](https://www.princeton.edu/~pmittal/)


Code for Princeton Senior Thesis: "A Defense against Adversarial Patch Attacks on Neural Network Classifiers" [link]

- For fall work, majority of work can be found in eval_defense_cifar.py, patch_attack_cifar_dual.py, patch_attack_cifar.py, and PatchAttacker.py. 
- [fall report]
- For spring work, and final thesis work, code can be found in nets/resnet.py, test_mod_resnet.py, test_mod_resnet_eval.py, and PatchAttacker.py. 
- [thesis]
- link to google drive and/or the library page

## Requirements
The code is tested with Python 3.6 and PyTorch 1.3.0. The complete list of required packages are available in `requirement.txt`, and can be installed with `pip install -r requirement.txt`. The code should be compatible with other versions of packages.

## Files
```shell
├── README.md                        #this file 
├── requirement.txt                  #required package
├── example_cmd.sh                   #example command to run the code
├── mask_bn_imagenet.py              #mask-bn for imagenet(te)
├── mask_bn_cifar.py                 #mask-bn for cifar
├── mask_ds_imagenet.py              #mask-ds for imagenet(te)
├── mask_ds_cifar.py                 #mask-ds for cifar
├── nets
|   ├── bagnet.py                    #modified bagnet model for mask-bn
|   ├── resnet_original.py           #modified resnet model for mask-bn
|   ├── resnet.py                    #modified resnet model for this thesis
|   ├── dsresnet_imgnt.py            #ds-resnet-50 for imagenet(te)
|   └── dsresnet_cifar.py            #ds-resnet-18 for cifar
├── utils
|   ├── defense_utils.py             #utils for different defenses
|   ├── normalize_utils.py           #utils for nomrlize images stored in numpy array (unused in the paper)
|   ├── cutout.py                    #utils for CUTOUT training (unused)
|   └── progress_bar.py              #progress bar (used in train_cifar.py)
├── test_acc_imagenet.py             #test clean accuracy of resnet/bagnet on imagenet(te); support clipping, median operations
├── test_acc_cifar.py                #test clean accuracy of resnet/bagnet on cifar; support clipping, median operations
├── train_imagenet.py                #train resnet/bagnet for imagenet(te)
├── train_cifar.py                   #train resnet/bagnet for cifar
├── patch_attack_imagenet.py         #empirically attack resnet/bagnet trained on imagenet(te)
├── patch_attack_cifar.py            #empirically attack resnet/bagnet trained on cifar
├── patch_attack_cifar_dual.py       #empirically attack resnet/bagnet trained on cifar on two models
├── PatchAttacker.py                 #untargeted adversarial patch attack 
├── ds_imagenet.py                   #ds for imagenet(te)
├── ds_cifar.py                      #ds for imagenet(te)
├── test_mod_resnet.py               #testing script for adaptive attack on defense
├── test_mod_resnet_eval.py          #use modified resnet model to evaluate defense from this thesis
|
└── checkpoints                      #directory for checkpoints
    ├── README.md                    #details of each checkpoint
    └── ...                          #model checkpoints
```
NOTE: except for `test_mod_resnet.py` and `test_mod_resnet_eval.py`, do not use `nets/resnet.py`, use `nets/resnet_original`!

## Datasets
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Usage
- See **Files** for details of each file. 

- Download data in **Datasets** and specify the data directory to the code.

- (optional) Download checkpoints from Google Drive [link](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT?usp=sharing) and move them to `checkpoints`.

- See `example_cmd.sh` for example commands for running the code.

If anything is unclear, please open an issue or contact Nicholas Sum (nsum@princeton.edu / nicksum107@gmail.com) or Chong Xiang (cxiang@princeton.edu).

## Related Repositories
- [certifiedpatchdefense](https://github.com/Ping-C/certifiedpatchdefense)
- [patchSmoothing](https://github.com/alevine0/patchSmoothing)
- [bag-of-local-features-models](https://github.com/wielandbrendel/bag-of-local-features-models)
- [PatchGuard](https://github.com/inspire-group/PatchGuard)
