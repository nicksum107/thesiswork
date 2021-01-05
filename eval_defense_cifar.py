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
import scipy.signal

import argparse

# mask assuming square mask
def mask(data, patch_locs, do_mask, sizes):
    for i in range(len(data)):
        if do_mask[i]:
            r = patch_locs[i, 0]
            c = patch_locs[i, 1]
            data[i, :, r:r+sizes[i], c:c+sizes[i]] = 0
    return data 

# find size in pixels based on size in feature space
def size_convert(sizes, model=17):
    if model==17:
        return (((sizes*2)+2)*2+2)*2+2
    else: 
        return sizes * 8
def perfect_defense(data, true_locs, patch_size): 
    return mask(data_attacked, true_locs, np.ones(len(data), dtype=bool), np.ones(data.shape[0], dtype=int) * args.patch_size)

def threshold(heatmaps, thresh_val=1.5):
    means = np.mean(heatmaps, axis=(1,2))
    std_devs = np.mean(heatmaps, axis=(1,2))
    thresholds = np.add(means, np.multiply(std_devs, thresh_val))
    
    #todo use numpy to do oneliner
    for i in range(len(thresholds)):
        heatmaps[i, heatmaps[i] < thresholds[i]] = 0
        
    return heatmaps, thresholds

def defense_window(data_attacked, heatmaps, size=5, window_thresh=1/2): 
    heatmaps, _ = threshold(heatmaps) 
    nz = np.zeros((heatmaps.shape[0], heatmaps.shape[1]-size+1, heatmaps.shape[2]-size+1))
    
    for i in range(len(nz)):
        for j in range(len(nz[0])):
            nz[:, i,j] = np.count_nonzero(heatmaps[:, i:size+i, j:size+j], axis=(1,2))
    indices = np.zeros((len(nz), 2), dtype=int)
    maxes = np.zeros(len(nz))
    for i in range(len(indices)):
        indices[i] = np.unravel_index(np.argmax(nz[i]), nz[i].shape)
        maxes[i] = nz[i, indices[i,0], indices[i,1]]
        
    return mask(data_attacked, np.multiply(indices, 8), (maxes >= (size **2 * window_thresh)), size_convert(size))

def defense_avg(data_attacked, heatmaps, thresh2=0.3):
    heatmaps, ori_thresh = threshold(heatmaps)

    indices =  np.zeros((len(heatmaps), 2), dtype=int)
    sizes = np.zeros((len(heatmaps)), dtype=int)
    to_mask = np.zeros(len(heatmaps), dtype=bool)
    for i in range(len(heatmaps)):
        max_ = -1 
        r,c= 0,0
        size = 5

        for s in range(3, 8):
            f = np.ones((s,s))/s# / (s ** 2) divide by square root of the size of the mask? 
            smoothed = scipy.signal.convolve2d(heatmaps[i], f, mode='valid')
            curr_max = smoothed.max()
            # curr_r, curr_c = np.count_nonzero(heatmaps[i])
            if curr_max > max_: 
                max_ = curr_max
                r,c = np.unravel_index(smoothed.argmax(), smoothed.shape)
                size = s 
        # print(max_, r, c, size)
        if max_ > -1:
            sd = np.std(heatmaps[i])
            to_mask[i] = max_ >= (ori_thresh[i] + thresh2 * sd)
            indices[i,0], indices[i,1] = r,c
            sizes[i] = size 
    return mask(data_attacked, np.multiply(indices, 8), to_mask, size_convert(sizes))

def defend(defense, defense_model, data_attacked, predict_attacked, true_locs, patch_size):
    if defense=='perfect': 
        return perfect_defense(data_attacked, true_locs, patch_size=patch_size)
    
    heatmaps_attacked = defense_model(torch.from_numpy(data_attacked)).cpu().detach().numpy()
    top_logits_attacked = np.empty(heatmaps_attacked.shape[0:3])
    for i in range(len(top_logits_attacked)):
        top_logits_attacked[i] = heatmaps_attacked[i, :,:, predict_attacked[i]]
    
    if defense == 'window': 
        return defense_window(data_attacked, top_logits_attacked)
    elif defense == 'avg': 
        return defense_avg(data_attacked, top_logits_attacked)
    else: # default: no defense
        return data_attacked

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
parser.add_argument("--dump_dir",default='patch_adv',type=str,help="directory to save attack results")
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument("--attacked_model",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--defense_model",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--final_model",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--data_dir",default='./data/cifar',type=str,help="path to data")
parser.add_argument("--patch_size",default=30,type=int,help="size of the adversarial patch")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
parser.add_argument("--defense", default='perfect', type=str, help="defense type")
parser.add_argument('--debug', action='store_true', default=False, help='debug output')
batch_size = 16
# parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. one of mean, median, cbn")

args = parser.parse_args()
print('eval defense: size', args.patch_size, 'defense', args.defense)

MODEL_DIR=os.path.join('.',args.model_dir)
DATA_DIR=os.path.join(args.data_dir)
# DUMP_DIR=os.path.join('dump',args.dump_dir+'_{}'.format(args.attacked_model))
DUMP_DIR=os.path.join('dump','dual_patch_adv_bagnet17_192_resnet50_192_cifar')
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

val_loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False)

has_attacked = False 
if os.path.exists(os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size))):
    data_adv = joblib.load(os.path.join(DUMP_DIR,'patch_adv_list_{}.z'.format(args.patch_size)))
    true_patch_locs = joblib.load(os.path.join(DUMP_DIR,'patch_loc_list_{}.z'.format(args.patch_size)))
    has_attacked = True 
    print('Has attacked dataset')
    if args.debug:
        print('Got', DUMP_DIR)
        print(data_adv.shape, true_patch_locs.shape)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

attacked_model = load_model(args.attacked_model, MODEL_DIR, args.clip, device, 'mean')
defense_model = load_model(args.defense_model, MODEL_DIR, args.clip, device, 'none')

different_final =  not (args.final_model == args.attacked_model)

if different_final:  
    print('loading different final') 
    final_model = load_model(args.final_model, MODEL_DIR, args.clip, device, 'mean')
    final_model.eval()
    fin_cln_acc_list = []
    fin_att_acc_list = []
    fin_def_acc_list = []
    fin_def_cln_acc_list = []


if not has_attacked: 
    attacker = PatchAttacker(attacked_model, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],patch_size=args.patch_size,step_size=0.05,steps=500)

batch_num = 0
print("starting")
cln_acc_list = [] 
def_acc_list = [] 
att_acc_list = []
cln_def_acc_list = []

for clean_data, labels in tqdm(val_loader):
    # if batch_num > 360: 
    #     break
    clean_data, labels = clean_data.to(device), labels.to(device) 

    clean_output = attacked_model(clean_data)
    clean_predict = torch.argmax(clean_output, dim=1)
    clean_acc = torch.sum(clean_predict == labels).cpu().detach().numpy() / clean_data.size(0)
    
    # del clean_output
    del clean_predict 
    if args.debug:
        print('clean', clean_acc)
    cln_acc_list.append(clean_acc)
    del clean_acc 

    if different_final:
        fin_clean_output = (clean_output + final_model(clean_data))/2
        
        
        # fin_clean_output = final_model(clean_data)
        fin_clean_predict = torch.argmax(fin_clean_output, dim=1)
        fin_clean_acc = torch.sum(fin_clean_predict == labels).cpu().detach().numpy() / clean_data.size(0)
        del clean_output###

        del fin_clean_output
        del fin_clean_predict
        if args.debug:
            print('fin clean acc', fin_clean_acc)
        fin_cln_acc_list.append(fin_clean_acc)
        del fin_clean_acc

    if has_attacked: 
        data_attacked = data_adv[batch_num*batch_size:(batch_num+1)*batch_size]
        true_locs = true_patch_locs[batch_num*batch_size:(batch_num+1)*batch_size]

        attacked_output = attacked_model(torch.from_numpy(data_attacked))
        attacked_predict = torch.argmax(attacked_output, dim=1)
        attacked_acc = torch.sum(attacked_predict == labels).cpu().detach().numpy()/ data_attacked.shape[0]
        # del attacked_output
        if args.debug:
            print('attacked', attacked_acc)
        att_acc_list.append(attacked_acc)
        del attacked_acc
    else: 
        # must calculate data if we don't have 
        data_attacked, true_locs = attacker.perturb(clean_data, labels)
        attacked_output = attacked_model(data_attacked)
        attacked_predictions = torch.argmax(attacked_output, dim=1)
        attacked_acc = torch.sum(attacked_predictions == labels).cpu().detach().numpy()/ data_attacked.size(0)

        data_attacked = data_attacked.cpu().detach().numpy()
        true_locs = true_locs.cpu().detach().numpy()
    
    if different_final:
        fin_att_output = (attacked_output + final_model(torch.from_numpy(data_attacked))) / 2
        # fin_att_output = final_model(torch.from_numpy(data_attacked))
        fin_att_predict = torch.argmax(fin_att_output, dim=1)
        fin_att_acc = torch.sum(fin_att_predict==labels).cpu().detach().numpy() / data_attacked.shape[0]
        del attacked_output ###

        del fin_att_output 
        del fin_att_predict 
        if args.debug:
            print('fin attacked', fin_att_acc)
        fin_att_acc_list.append(fin_att_acc)
        del fin_att_acc

    # masking
    defended_data = defend(args.defense, defense_model, data_attacked, attacked_predict, true_locs, args.patch_size)

    defended_output = attacked_model(torch.from_numpy(defended_data))
    defended_predict = torch.argmax(defended_output, dim=1)
    defended_acc = torch.sum(defended_predict == labels).cpu().detach().numpy()/ defended_data.shape[0]
    # del defended_output 
    del defended_predict 
    if args.debug:
        print(args.defense, 'defense attacked', defended_acc)
    def_acc_list.append(defended_acc)
    del defended_acc

    if different_final:
        fin_def_output = (defended_output + final_model(torch.from_numpy(defended_data)))/2
        # fin_def_output = final_model(torch.from_numpy(defended_data))
        fin_def_predict = torch.argmax(fin_def_output, dim=1)
        fin_def_acc = torch.sum(fin_def_predict == labels).cpu().detach().numpy() / defended_data.shape[0]
        del defended_output ##
        del fin_def_output
        del fin_def_predict
        if args.debug:
            print('fin def acc', fin_def_acc)
        fin_def_acc_list.append(fin_def_acc)
        del fin_def_acc
    del defended_data

    # defense on clean image
    clean_def_data = defend(args.defense, defense_model, clean_data.cpu().detach().numpy(), attacked_predict, true_locs, args.patch_size)
    del attacked_predict

    clean_def_output = attacked_model(torch.from_numpy(clean_def_data))
    clean_def_predict = torch.argmax(clean_def_output, dim=1)
    clean_def_acc = torch.sum(clean_def_predict == labels).cpu().detach().numpy()/ clean_def_output.shape[0]
    # del clean_def_output
    del clean_def_predict 
    if args.debug:
        print(args.defense, 'defense clean', clean_def_acc)
    cln_def_acc_list.append(clean_def_acc)
    del clean_def_acc

    if different_final:
        fin_def_cln_output = (clean_def_output + final_model(torch.from_numpy(clean_def_data)))/2
        
        # fin_def_cln_output = final_model(torch.from_numpy(clean_def_data))
        fin_def_cln_predict = torch.argmax(fin_def_cln_output, dim=1)
        fin_def_cln_acc = torch.sum(fin_def_cln_predict == labels).cpu().detach().numpy() / clean_def_data.shape[0]
        del clean_def_output##
        del fin_def_cln_output
        del fin_def_cln_predict
        if args.debug:
            print('fin def cln acc', fin_def_cln_acc)
        fin_def_cln_acc_list.append(fin_def_cln_acc)
        del fin_def_cln_acc
    del clean_def_data

    if args.debug or batch_num % 16 == 0:
        print('cumulative acc: clean acc | def acc | att acc | cln_def')
        print('attacked model:', np.mean(np.array(cln_acc_list)), np.mean(np.array(def_acc_list)), np.mean(np.array(att_acc_list)), np.mean(np.array(cln_def_acc_list)))
        if different_final:
            print('final model:', np.mean(np.array(fin_cln_acc_list)), np.mean(np.array(fin_def_acc_list)), np.mean(np.array(fin_att_acc_list)), np.mean(np.array(fin_def_cln_acc_list)))


    batch_num += 1

print('cumulative acc: clean acc | def acc | att acc | cln_def')

print('attacked model:', np.mean(np.array(cln_acc_list)), np.mean(np.array(def_acc_list)), np.mean(np.array(att_acc_list)), np.mean(np.array(cln_def_acc_list)))
if different_final:
    print('final model:', np.mean(np.array(fin_cln_acc_list)), np.mean(np.array(fin_def_acc_list)), np.mean(np.array(fin_att_acc_list)), np.mean(np.array(fin_def_cln_acc_list)))