import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from  torchvision import datasets, transforms

import numpy as np 

from tqdm import tqdm
import joblib

import nets.bagnet
import nets.resnet
import PIL

from PatchAttacker import PatchAttacker
import os 

import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='patch_adv',type=str,help="directory to save attack results")
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument("--model",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--data_dir",default='./data/cifar',type=str,help="path to data")
parser.add_argument("--patch_size",default=30,type=int,help="size of the adversarial patch")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. one of mean, median, cbn")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)
DUMP_DIR=os.path.join('dump',args.dump_dir+'_{}'.format(args.model))
if not os.path.exists('dump'):
	os.mkdir('dump')
if not os.path.exists(DUMP_DIR):
	os.mkdir(DUMP_DIR)

img_size=192
#prepare data
transform_test = transforms.Compose([
	transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # noramlize channel to have mean, std dev
])
testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

val_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

#build and initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.clip > 0:
	clip_range = [0,args.clip]
else:
	clip_range = None
    
if 'bagnet17' in args.model:
    model = nets.bagnet.bagnet17(clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet33' in args.model:
    model = nets.bagnet.bagnet33(clip_range=clip_range,aggregation=args.aggr)
elif 'bagnet9' in args.model:
    model = nets.bagnet.bagnet9(clip_range=clip_range,aggregation=args.aggr)
elif 'resnet50' in args.model:
    model = nets.resnet.resnet50(clip_range=clip_range,aggregation=args.aggr)

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)
model = model.to(device)
if device == 'cuda':
	model = torch.nn.DataParallel(model)
	cudnn.benchmark = True
print('restoring model from checkpoint...')
checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
model.load_state_dict(checkpoint['net'])
model = model.to(device)
model.eval()


# load in a logits model
if 'bagnet17' in args.model:
    logits_model = nets.bagnet.bagnet17(clip_range=clip_range,aggregation='none')
elif 'bagnet33' in args.model:
    logits_model = nets.bagnet.bagnet33(clip_range=clip_range,aggregation='none')
elif 'bagnet9' in args.model:
    logits_model = nets.bagnet.bagnet9(clip_range=clip_range,aggregation='none')
elif 'resnet50' in args.model:
    logits_model = nets.resnet.resnet50(clip_range=clip_range,aggregation='none')
   
num_ftrs = logits_model.fc.in_features
logits_model.fc = nn.Linear(num_ftrs, 10)
logits_model = logits_model.to(device)
if device == 'cuda':
	logits_model = torch.nn.DataParallel(logits_model)
	cudnn.benchmark = True
print('restoring logits_model from checkpoint...')
checkpoint = torch.load(os.path.join(MODEL_DIR,args.model+'.pth'))
logits_model.load_state_dict(checkpoint['net'])
logits_model = logits_model.to(device)
logits_model.eval()



attacker = PatchAttacker(model, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],patch_size=args.patch_size,step_size=0.05,steps=500)

adv_list=[]
error_list=[]
accuracy_list=[]
patch_loc_list=[]

print("here")
for data,labels in tqdm(val_loader):
	
	data,labels=data.to(device),labels.to(device)
	data_adv,patch_loc = attacker.perturb(data, labels)

	# print(data.shape, data_adv.shape, data.detach().cpu().numpy())
	np.save("temp_data", data.detach().cpu().numpy())
	np.save("temp_data_adv", data_adv.detach().cpu().numpy())

	logits_data = logits_model(data)
	logits_data_adv = logits_model(data_adv)
	np.save("logits_data", logits_data.detach().cpu().numpy())
	np.save("logits_data_adv", logits_data_adv.detach().cpu().numpy())

	break 

	output_adv = model(data_adv)
	error_adv=torch.sum(torch.argmax(output_adv, dim=1) != labels).cpu().detach().numpy()/ data.size(0)
	output_clean = model(data)
	acc_clean=torch.sum(torch.argmax(output_clean, dim=1) == labels).cpu().detach().numpy()/ data.size(0)

	data_adv=data_adv.cpu().detach().numpy()
	patch_loc=patch_loc.cpu().detach().numpy()

	patch_loc_list.append(patch_loc)
	adv_list.append(data_adv)
	error_list.append(error_adv)
	accuracy_list.append(acc_clean)


adv_list = np.concatenate(adv_list)
patch_loc_list = np.concatenate(patch_loc_list)
joblib.dump(adv_list,os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
joblib.dump(patch_loc_list,os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
print("Attack success rate:",np.mean(error_list))
print("Clean accuracy:",np.mean(accuracy_list))
	
