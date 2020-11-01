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

def upscale_coords(i, j): 
    # 3 downsamples occur in bagnet17
    return 

def threshold_torch(heatmaps, thresh_val=1.5):
    # print(heatmaps.shape)
    means = torch.mean(heatmaps, (1,2))
    std_devs = torch.std(heatmaps, (1,2))
    thresholds = torch.add(means, torch.mul(std_devs, thresh_val))
    ret = torch.zeros(heatmaps.shape)
    # print(thresholds)
    for i in range(len(heatmaps)): 
        thresher = nn.Threshold(thresholds[i].item(), 0)
        ret[i] = thresher(heatmaps[i])
    return ret

def threshold(heatmaps, thresh_val=1.5):
    means = np.mean(heatmaps, axis=(1,2))
    std_devs = np.mean(heatmaps, axis=(1,2))
    thresholds = np.add(means, np.multiply(std_devs, thresh_val))
    # print(thresholds)
    #todo use numpy to do oneliner
    for i in range(len(thresholds)):
        heatmaps[i, heatmaps[i] < thresholds[i]] = 0
        
    # heatmaps[heatmaps < thresholds] = 0
    # print(heatmaps.shape)
    return heatmaps 

def twod_max_argmax(tensor):
    flat_index = torch.argmax(torch.flatten(tensor, start_dim=1), dim=1)
    maxes = torch.flatten(tensor, start_dim=1)[:, flat_index]
    unraveled = torch.zeros((flat_index.shape[0], 2))
    for i in flat_index:
        unraveled[i] = [int(i/tensor.shape[1]), int(i/tensor.shape[2])]
    return unraveled, maxes

#finds a square patch on the heatmap of the given size (in logits)
def locate_patch_window(heatmaps, size=5, window_thresh=1/2):
    heatmaps = threshold(heatmaps)

    nz = np.zeros((heatmaps.shape[0], heatmaps.shape[1]-size+1, heatmaps.shape[2]-size+1))
    
    for i in range(len(nz)):
        for j in range(len(nz[0])):
            nz[:, i,j] = np.count_nonzero(heatmaps[:, i:size+i, j:size+j], axis=(1,2))
    indices = np.zeros((len(nz), 2), dtype=int)
    maxes = np.zeros(len(nz))
    for i in range(len(indices)):
        indices[i] = np.unravel_index(np.argmax(nz[i]), nz[i].shape)
        maxes[i] = nz[i, indices[i,0], indices[i,1]]

    # return false if the candidate does not have many | upscale
    return (maxes >= (size ** 2 * window_thresh)), np.multiply(indices, 8)

# returns the masked images 
def mask_image(data, detections, locations, patch_size=int(5*192/22)):
    
    for i in range(len(detections)):
        if detections[i]:
            r = locations[i, 0]
            c = locations[i, 1]
            data[i, :, r:r+patch_size,c:c+patch_size] = 0
    return data
    


parser = argparse.ArgumentParser()
parser.add_argument("--dump_dir",default='patch_adv',type=str,help="directory to save attack results")
parser.add_argument("--model_dir",default='checkpoints',type=str,help="path to checkpoints")
parser.add_argument("--model",default='bagnet17_192',type=str,help="model name")
parser.add_argument("--data_dir",default='./data/cifar',type=str,help="path to data")
parser.add_argument("--patch_size",default=30,type=int,help="size of the adversarial patch")
parser.add_argument("--clip",default=-1,type=int,help="clipping value; do clipping when this argument is set to positive")
# parser.add_argument("--aggr",default='mean',type=str,help="aggregation methods. one of mean, median, cbn")

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

attacked_model = load_model(args.model, MODEL_DIR, args.clip, device, 'mean')
attacked_model.eval()
logits_model = load_model(args.model, MODEL_DIR, args.clip, device, 'none')

attacker = PatchAttacker(attacked_model, [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010],patch_size=args.patch_size,step_size=0.05,steps=500)

attcked_data_list=[]
clean_acc_list=[]
true_patch_loc_list=[]
attacked_acc_list=[]

defended_clean_acc_list = []
defended_attacked_acc_list = []

defended_clean_patch_loc_list = []
defended_attacked_patch_loc_list = []

batch_num = 0 
for data,labels in tqdm(val_loader):
    # get data
    data,labels=data.to(device),labels.to(device)
    data_attacked,true_patch_loc = attacker.perturb(data, labels)    

    # attacked output
    attacked_output = attacked_model(data_attacked)
    attacked_predictions = torch.argmax(attacked_output, dim=1)
    attacked_acc = torch.sum(attacked_predictions == labels).cpu().detach().numpy()/ data.size(0)
    
    # clean output
    clean_output = attacked_model(data)
    clean_predictions = torch.argmax(clean_output, dim=1)
    clean_acc = torch.sum(clean_predictions == labels).cpu().detach().numpy()/ data.size(0)
    

    # defense (attacked)
    logits_attacked = logits_model(data_attacked)
    top_logits_attacked = torch.empty(logits_attacked.shape[0:3])
    
    # get the top predicted logit in each image
    for i in range(len(attacked_predictions)):
        top_logits_attacked[i] = logits_attacked[i,:,:,attacked_predictions[i]] 

    # pytorch count nonzero is busted, so we swtich to numpy
    attacked_detections, attacked_indices = locate_patch_window(top_logits_attacked.cpu().detach().numpy())
    del logits_attacked
    del top_logits_attacked

    data_attacked = mask_image(data_attacked, attacked_detections, attacked_indices)
    defended_attacked_output = attacked_model(data_attacked)

    defended_attacked_predictions = torch.argmax(defended_attacked_output, dim=1)
    defended_attacked_acc = torch.sum(defended_attacked_predictions == labels).cpu().detach().numpy()/ data_attacked.size(0)
    del defended_attacked_predictions
    del defended_attacked_output
    

    # defense (clean)
    logits_clean = logits_model(data)
    top_logits_clean = torch.empty(logits_clean.shape[0:3])
    
    # get the top predicted logit in each image
    for i in range(len(clean_predictions)):
        top_logits_clean[i] = logits_clean[i,:,:,clean_predictions[i]] 

    # pytorch count nonzero is busted, so we swtich to numpy
    clean_detections, clean_indices = locate_patch_window(top_logits_clean.cpu().detach().numpy())
    del logits_clean
    del top_logits_clean

    data = mask_image(data, clean_detections, clean_indices)
    defended_clean_output = attacked_model(data)

    defended_clean_predictions = torch.argmax(defended_clean_output, dim=1)
    defended_clean_acc = torch.sum(defended_clean_predictions == labels).cpu().detach().numpy()/ data.size(0)
    del defended_clean_predictions
    del defended_clean_output
    
    # post process 
    true_patch_loc=true_patch_loc.cpu().detach().numpy()

    attcked_data_list.append(data_attacked)
    true_patch_loc_list.append(true_patch_loc)

    attacked_acc_list.append(attacked_acc)
    clean_acc_list.append(clean_acc)
    
    defended_clean_acc_list.append(defended_clean_acc)
    defended_attacked_acc_list.append(defended_attacked_acc)

    defended_clean_patch_loc_list.append(clean_indices)
    defended_attacked_patch_loc_list.append(attacked_indices)

    print('cumulative vals | batch', batch_num, 
          '| clean acc', np.mean(clean_acc_list), 
          '| attacked acc', np.mean(attacked_acc_list), 
          '| defended clean acc', np.mean(defended_clean_acc_list), 
          '| defended attacked acc', np.mean(defended_attacked_acc_list))
    batch_num+=1 


data_attacked = np.concatenate(data_attacked)
true_patch_loc_list = np.concatenate(true_patch_loc_list)
defended_clean_patch_loc_list = np.concatenate(defended_clean_patch_loc_list)
defended_attacked_patch_loc_list = np.concatenate(defended_attacked_patch_loc_list)
joblib.dump(adv_list,os.path.join(DUMP_DIR,'patch_att_list_{}.z'.format(args.patch_size)))
joblib.dump(true_patch_loc_list,os.path.join(DUMP_DIR,'true_patch_loc_list_{}.z'.format(args.patch_size)))
joblib.dump(defended_clean_patch_loc_list,os.path.join(DUMP_DIR,'defended_clean_patch_loc_list_{}.z'.format(args.patch_size)))
joblib.dump(defended_attacked_patch_loc_list,os.path.join(DUMP_DIR,'defended_att_patch_loc_list_{}.z'.format(args.patch_size)))

print("Clean accuracy:",np.mean(clean_acc_list))
print("Attacked accuracy:",np.mean(attacked_acc_list))
print("Defended clean accuracy:",np.mean(defended_clean_acc_list))
print("Defended attacked accuracy:",np.mean(defended_attacked_acc_list))

