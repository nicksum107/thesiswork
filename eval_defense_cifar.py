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
    return model

parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='patch_adv',type=str,help="directory to save attack results")
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument("--attacked_model",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--final_model",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--data_dir",default='./data/cifar',type=str,help="path to data")
parser.add_argument("--patch_size",default=30,type=int,help="size of the adversarial patch")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--defense", default='avg')
batch_size = 16
# parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. one of mean, median, cbn")

args = parser.parse_args()

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)
DUMP_DIR=os.path.join('dump',args.dump_dir+'_{}'.format(args.attacked_model))
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

data = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)

val_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

has_attacked = False 
if os.path.exists(os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size))):
    data_adv = joblib.load(os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
    true_patch_locs = joblib.load(os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
    has_attacked = True 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

attacked_model = load_model(args.attacked_model, MODEL_DIR, args.clip, device, 'mean')
attacked_model.eval()
defense_model = load_model(args.defense_model, MODEL_DIR, args.clip, device, 'none')

if not has_attacked: 
    attacker = PatchAttacker(attacked_model, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],patch_size=args.patch_size,step_size=0.05,steps=500)

batch_num = 0
for clean_data, labels in tqdm(val_loader):
    data, labels = data.to(device), labels.to(device) 

    clean_output = attacked_model(data)
    clean_predict = torch.argmax(clean_output, dim=1)
    clean_acc = torch.sum(clean_predict == labels).cpu().detach().numpy() / data.size(0)
    del clean_output


    if has_attacked: 
        data_attacked = data_adv[batch_num*batch_size:(batch_num+1)*batch_size]
        true_locs = true_patch_locs[batch_num*batch_size:(batch_num+1)*batch_size]
        
        attacked_output = attacked_model(torch.from_numpy(data_attacked))
        attacked_predictions = torch.argmax(attacked_output, dim=1)
        attacked_acc = torch.sum(attacked_predictions == labels).cpu().detach().numpy()/ data.size(0)
        del attacked_output
    else: 
        data_attacked, true_locs = attacker.perturb(data, labels)
        attacked_output = attacked_model(data_attacked)
        attacked_predictions = torch.argmax(attacked_output, dim=1)
        attacked_acc = torch.sum(attacked_predictions == labels).cpu().detach().numpy()/ data.size(0)
        del attacked_output

    print(clean_acc, attacked_acc)
    print(true_locs)
    


    batch_num += 1
    break