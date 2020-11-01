import numpy as np
import matplotlib.pyplot as plt
from utils.normalize_utils import unnormalize_np
import scipy.ndimage
import cv2

def scale_logits(i, j): 
    # 3 downsamples occur in bagnet17
    return i *8, j *8

# look for connected areas in the image
# heatmap = thresholded heatmap, 
# visited = which areas have we visited yet
# i,j = coordinates searched
# max_travel = how big of a patch do we want to find 
def fill (heatmap, visited, i, j, max_travel_x, max_travel_y): 
    if visited[i,j]==1: 
        return False, 0, 0
    if max_travel_x == 0 and max_travel_y == 0: 
        return True, 5-max_travel_x, 5-max_travel_y
    visited[i,j] = 1 

    if heatmap[i,j] > 0: 
        max_max_travel_x = []
        max_max_travel_y = []
        if i < len(heatmap)-1:
            yes, thismax_x, thismax_y = fill(heatmap, visited,i+1, j, max_travel_x-1, max_travel_y)
            if yes:
                max_max_travel_x.append(thismax_x)
        if i > 0:
            yes,  thismax_x, thismax_y =fill(heatmap, visited, i-1, j, max_travel_x-1, max_travel_y)
            if yes:
                max_max_travel_x.append(thismax_x)
        if j < len(heatmap[0])-1:
            yes,  thismax_x, thismax_y = fill(heatmap, visited, i, j+1, max_travel_x, max_travel_y-1)
            
            if yes: 
                max_max_travel_y.append(thismax_y)
        if j > 0: 
            yes,  thismax_x, thismax_y = fill (heatmap, visited, i, j-1, max_travel_x, max_travel_y-1)
            if yes:
                max_max_travel_y.append(thismax_y)
        max_max_travel_x.append(max_travel_x)
        max_max_travel_y.append(max_travel_y)
        return True, max(max_max_travel_x), max(max_max_travel_y)

    return False, 5-max_travel_x, 5-max_travel_y

# in future, return candidates with corresponding sizes 
def locate_patch_fill(logit): 
    thresh = np.mean(logit) + thresh_val * np.mean(logit)
    threshed = (logit > thresh) * logit
    visited = np.zeros(threshed.shape)
    found = False 
    largest_size_x = 0
    largest_size_y = 0
    largest_loc = (0,0)
    for i in range(len(threshed)):
        for j in range(len(threshed[i])):
            if threshed[i,j] > 0 and visited[i,j] == 0:
                patch_found, size_x, size_y = fill(threshed, visited, i,j, 5,5)
                if patch_found and (size_x > 3) and (size_y > 3): 
                    found = True 
                    largest_size_x = max(largest_size_x, size_x)
                    largest_size_y = max(largest_size_y, size_y)
                    largest_loc = (i,j)
                    return found, largest_loc, largest_size_x, largest_size_y
            visited[i,j] = 1
    # print (found, largest_loc, largest_size_x, largest_size_y)
    return found, largest_loc, largest_size_x, largest_size_y


#finds a square patch on the heatmap of the given size
def locate_patch_window(heatmap, size):
    window_thresh = 1/2


    nz = np.zeros((heatmap.shape[0]-size+1, heatmap.shape[1]-size+1))
    for i in range(len(nz)):
        for j in range(len(nz[0])):
            nz[i,j] = np.count_nonzero(heatmap[i:size+i, j:size+j])
    i, j = np.unravel_index(np.argmax(nz), nz.shape)
    # note i and j also correspond to the top left in the logit too 
    masked_log = heatmap
    masked_log[i:i+size, j:j+size] = 0

    # translate i,j to the larger image
    scaled_i, scaled_j = scale_logits(i,j)

    # return false if the candidate does not have many 
    return (nz[i,j] >= (size ** 2 * window_thresh)), scaled_i, scaled_j


# load
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
thresh_val = 1.5 #hyperparam to set

data = np.load('temp_data.npy')
data_attacked = np.load('temp_data_adv.npy')

# norm data
norm_data = np.zeros(data.shape)
norm_data_attacked = np.zeros(data_attacked.shape)
for i in range(len(data)):
    for j in range(3):
        norm_data[i,j] = np.add(np.multiply(data[i,j], std[j]), mean[j])
        norm_data_attacked[i,j] = np.add(np.multiply(data_attacked[i,j], std[j]), mean[j])

print(norm_data.shape)
todisp = np.transpose(norm_data, (0,2,3,1))
todisp_attacked = np.transpose(norm_data_attacked, (0,2,3,1))

logits = np.load('logits_data.npy')
logits_attacked = np.load('logits_data_adv.npy')

print(logits.shape)

logits_disp = np.transpose(logits, (0,3,1,2))
logits_disp_attacked = np.transpose(logits_attacked, (0,3,1,2))

# aggregation 
class_score = np.mean(logits,axis=(1,2))
class_score_attacked = np.mean(logits_attacked, axis=(1,2))
score_std = np.std(logits, axis=(1,2))
score_std_attacked = np.std(logits_attacked, axis=(1,2))

# THRESHOLDING: based on mean + std deviation: this should probably be ok
# because aggregation function is mean; appendix of bagnet has normal-ish distr
thresh = class_score + thresh_val * score_std 
thresh_attacked = class_score_attacked + thresh_val * score_std_attacked

print('predicted labels for each class')
predict = np.argmax(class_score, axis=1) 
predict_attacked = np.argmax(class_score_attacked, axis=1)

print(predict, predict_attacked)
print(predict == predict_attacked)

print(data.shape, data_attacked.shape, logits_disp.shape, logits_disp_attacked.shape)

threshed = np.zeros(logits_disp.shape)
threshed_attacked = np.zeros(logits_disp_attacked.shape)
for i in range(len(logits_disp)):
    for j in range(len(logits_disp[0])):
        threshed[i,j] = (logits_disp[i,j] > thresh[i,j]) * logits_disp[i,j]
        threshed_attacked[i,j] = (logits_disp_attacked[i,j] > thresh_attacked[i,j]) * logits_disp_attacked[i,j]

#nonneg
for i in range(len(logits_disp)):
    for j in range(len(logits_disp[0])):
        logits_disp_attacked[i,j]  = (logits_disp_attacked[i,j] > 0) * np.abs(logits_disp_attacked[i,j])

cleanacc = 0 
defenseacc = 0
for i in range(len(data)):

    # plt.imshow(todisp[i])
    # plt.figure()
    # plt.imshow(todisp_attacked[i])
    # plt.figure()

    # plt.imshow(logits_disp[i,predict[i]])
    # plt.figure()
    # plt.imshow(logits_disp[i, predict_attacked[i]])
    # plt.figure()
    # plt.imshow(logits_disp_attacked[i, predict[i]])
    # plt.figure()
    # plt.imshow(logits_disp_attacked[i, predict_attacked[i]])
    # plt.figure()

    # plt.imshow(threshed[i, predict_attacked[i]])
    # plt.figure()
    # plt.imshow(threshed_attacked[i, predict_attacked[i]])
    # plt.figure()
    
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_original.png', np.clip(todisp[i], 0,1))
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_attacked.png', np.clip(todisp_attacked[i],0,1))
    
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_correct_logits.png', logits_disp[i,predict[i]])
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_correct_logits_target.png', logits_disp[i, predict_attacked[i]])
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_attacked_logits.png', logits_disp_attacked[i,predict[i]])
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_attacked_logits_target.png', logits_disp_attacked[i, predict_attacked[i]])

    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_correct_logits_thresh.png', threshed[i,predict[i]])
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_correct_logits_target_thresh.png', threshed[i, predict_attacked[i]])
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_attacked_logits_thresh.png', threshed_attacked[i,predict[i]])
    plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_attacked_logits_target_thresh.png', threshed_attacked[i, predict_attacked[i]])


    
    

    # found, (patch_x, patch_y), size_x, size_y = locate_patch(logits_disp_attacked[i, predict_attacked[i]])
    patch_size = 5
    
    found, r, c = locate_patch_window(threshed[i, predict_attacked[i]], patch_size) 
    if found: 
        print(i, 'clean found')
        adding = int(patch_size * 192/22)
        masked = np.array(todisp[i])
        # print(r, c)
        masked[r:r+adding, c:c+adding, :] = 0
        # plt.imshow(masked)
        # plt.title(str(i) + ' clean')
        # plt.figure() 
        
        plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_clean_badmask.png', np.clip(masked, 0,1))
    else: 
        cleanacc+=1
        print(i, 'clean not found')


    found, r, c = locate_patch_window(threshed_attacked[i, predict_attacked[i]], patch_size) 
    if found: 
        print(i, 'attacked found')
        adding = int(patch_size * 192/22)
        masked = np.array(todisp_attacked[i])
        # print(r, c)
        masked[r:r+adding, c:c+adding, :] = 0
        # plt.imshow(masked)
        # plt.title(str(i) + ' attacked')
        # plt.figure()
        plt.imsave('./image_dumps/known_size_window/image_' + str(i) + '_attacked_goodmask.png', np.clip(masked, 0,1))
        defenseacc += 1
    else:
        
        print(i, 'attacked not found')
    # break
plt.show()

print('defenseacc', defenseacc/len(data))
print('cleanacc', cleanacc/len(data))

# ordered_labels = np.argsort(labels, axis=1)
# ordered_labels_attacked = np.argsort(labels_attacked, axis=1)

# print(ordered_labels, ordered_labels_attacked)


#     blob_params = cv2.SimpleBlobDetector_Params()
#     blob_params.minThreshold = thresh[i, predict_attacked[i]] / np.max(logits_disp_attacked[i, predict_attacked[i]]) * 255 
#     blob_params.maxThreshold = thresh[i, predict_attacked[i]] / np.max(logits_disp_attacked[i, predict_attacked[i]]) * 255 
#     blob_params.filterByArea = 1 
#     #these would depend on image size and patch size
#     blob_params.minArea = 3
#     blob_params.maxArea = 5
#     blob_params.minDistBetweenBlobs = 3

#     ver = (cv2.__version__).split('.')
#     if int(ver[0]) < 3 :
#         detector = cv2.SimpleBlobDetector(blob_params)
#     else :
#         detector = cv2.SimpleBlobDetector_create(blob_params)

#     # print(logits_disp_attacked[i, predict_attacked[i]].dtype)
#     # print(logits_disp_attacked[i, predict_attacked[i]])
#     scaled = np.array(logits_disp_attacked[i, predict_attacked[i]] / np.max(logits_disp_attacked[i, predict_attacked[i]]) * 255, dtype=np.uint8)

#     keypoints = detector.detect(scaled)
#     print(keypoints)