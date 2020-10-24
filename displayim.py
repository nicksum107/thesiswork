import numpy as np
import matplotlib.pyplot as plt
from utils.normalize_utils import unnormalize_np
# load
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

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

logits_disp = np.transpose(logits, (0,3,1,2))
logits_disp_attacked = np.transpose(logits_attacked, (0,3,1,2))
# aggregation 
labels = np.mean(logits,axis=(1,2))
labels_attacked = np.mean(logits_attacked, axis=(1,2))

print('predicted labels')
predict = np.argmax(labels, axis=1) 
predict_attacked = np.argmax(labels_attacked, axis=1)

print(predict, predict_attacked)
print(predict == predict_attacked)

# for i in range(1, len(data)):
#     plt.figure()
#     plt.imshow(todisp[i])
#     plt.figure()
#     plt.imshow(todisp_attacked[i])
#     plt.figure()
#     plt.imshow(logits_disp[i,predict[i]])
#     plt.figure()
#     plt.imshow(logits_disp[i, predict_attacked[i]])
#     plt.figure()
#     plt.imshow(logits_disp_attacked[i, predict[i]])
#     plt.figure()
#     plt.imshow(logits_disp_attacked[i, predict_attacked[i]])
#     break
# plt.show()

print(data.shape, data_attacked.shape, logits.shape, logits_attacked.shape)

ordered_labels = np.argsort(labels, axis=1)
ordered_labels_attacked = np.argsort(labels_attacked, axis=1)

print(ordered_labels, ordered_labels_attacked)