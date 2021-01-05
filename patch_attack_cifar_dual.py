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

from PatchAttacker import DualPatchAttacker
import os 

import matplotlib.pyplot as plt

import argparse

def load_model(model_type, model_dir, clipping, device, aggr):
	if clipping > 0:
		clip_range = [0,clipping]
	else:
		clip_range = None
		
	if 'bagnet17' in model_type:
		model = nets.bagnet.bagnet17(clip_range=clip_range,aggregation=aggr)
	elif 'bagnet33' in model_type:
		model = nets.bagnet.bagnet33(clip_range=clip_range,aggregation=aggr)
	elif 'bagnet9' in model_type:
		model = nets.bagnet.bagnet9(clip_range=clip_range,aggregation=aggr)
	elif 'resnet50' in model_type:
		model = nets.resnet.resnet50(clip_range=clip_range,aggregation=aggr)
	
	num_ftrs = model.fc.in_features
	model.fc = nn.Linear(num_ftrs, 10)
	model = model.to(device)
	if device == 'cuda':
		model = torch.nn.DataParallel(model)
		cudnn.benchmark = True
	print('restoring model from checkpoint...')
	checkpoint = torch.load(os.path.join(model_dir,model_type+'.pth'))
	model.load_state_dict(checkpoint['net'])
	model = model.to(device)
	model.eval()
	return model
	
parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='dual_patch_adv',type=str,help="directory to save attack results")
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument("--model1",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--model2",default='resnet50_192_cifar',type=str,help="model name")
parser.add_argument("--data_dir",default='./data/cifar',type=str,help="path to data")
parser.add_argument("--patch_size",default=30,type=int,help="size of the adversarial patch")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. one of mean, median, cbn")
parser.add_argument('--recalc', action='store_true', default=False, help='debug output')
print('patch attack dual')
args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)
DUMP_DIR=os.path.join('dump',args.dump_dir+'_{}_{}'.format(args.model1, args.model2))
if not os.path.exists('dump'):
	os.mkdir('dump')
if not os.path.exists(DUMP_DIR):
	os.mkdir(DUMP_DIR)

if not os.path.exists(os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size))):
	args.recalc = True 
	print('force recalc')

img_size=192
#prepare data
transform_test = transforms.Compose([
	transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # noramlize channel to have mean, std dev
])

if not args.recalc: 
	current_adv = joblib.load(os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
	current_loc = joblib.load(os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
	print('loaded: ', current_adv.shape)

testset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

val_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)


#build and initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
model1 = load_model(args.model1, MODEL_DIR, args.clip, device, 'mean')
model2 = load_model(args.model2, MODEL_DIR, args.clip, device, 'mean')

attacker = DualPatchAttacker(model1, model2, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],patch_size=args.patch_size,step_size=0.01,steps=750)

if args.recalc:
	adv_list=[]
	patch_loc_list=[]
else:
	adv_list=[current_adv]
	patch_loc_list=[current_loc]

accuracy_list1=[]
error_list1=[]
accuracy_list2=[]
error_list2=[]

im_number = 0 
for data,labels in tqdm(val_loader):
	if (not args.recalc) and (im_number < current_adv.shape[0]):
		print('skip', im_number)
		im_number += 16
		continue 
	data,labels=data.to(device),labels.to(device)
	data_adv,patch_loc = attacker.perturb(data, labels)


	output_adv = model1(data_adv)
	error_adv=torch.sum(torch.argmax(output_adv, dim=1) != labels).cpu().detach().numpy()/ data.size(0)
	output_clean = model1(data)
	acc_clean=torch.sum(torch.argmax(output_clean, dim=1) == labels).cpu().detach().numpy()/ data.size(0)

	error_list1.append(error_adv)
	accuracy_list1.append(acc_clean)

	output_adv = model2(data_adv)
	error_adv = torch.sum(torch.argmax(output_adv,dim=1) != labels).cpu().detach().numpy()/data.size(0)
	output_clean = model2(data)
	acc_clean = torch.sum(torch.argmax(output_clean, dim=1) == labels).cpu().detach().numpy()/data.size(0)

	error_list2.append(error_adv)
	accuracy_list2.append(acc_clean)
	
	data_adv=data_adv.cpu().detach().numpy()
	patch_loc=patch_loc.cpu().detach().numpy()
	patch_loc_list.append(patch_loc)
	adv_list.append(data_adv)
	# dump every 16 batches
	if (int(im_number/16) % 16 == 0):
		temp_adv_list = np.concatenate(adv_list)
		temp_patch_loc_list = np.concatenate(patch_loc_list)
		print('dumping', temp_adv_list.shape, temp_patch_loc_list.shape)
		joblib.dump(temp_adv_list,os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
		joblib.dump(temp_patch_loc_list,os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
		print('finish dump')
		# free memory of these np arrays
		temp_adv_list = None 
		temp_patch_loc_list = None 
		
		print("Attack success rate model 1:",np.mean(error_list1))
		print("Clean accuracy model 1:",np.mean(accuracy_list1))
		print("Attack success rate model 2:",np.mean(error_list2))
		print("Clean accuracy model 2:",np.mean(accuracy_list2))
		# if im_number > 0: 
		# 	break 
	im_number += 16
	


print("Attack success rate model 1:",np.mean(error_list1))
print("Clean accuracy model 1:",np.mean(accuracy_list1))
print("Attack success rate model 2:",np.mean(error_list2))
print("Clean accuracy model 2:",np.mean(accuracy_list2))    
	

temp_adv_list = np.concatenate(adv_list)
temp_patch_loc_list = np.concatenate(patch_loc_list)
print('dumping', temp_adv_list.shape)
joblib.dump(temp_adv_list,os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
joblib.dump(temp_patch_loc_list,os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
print('finish dump')
# free memory of these np arrays
temp_adv_list = None 
temp_patch_loc_list = None 