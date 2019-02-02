import numpy as np
from scipy import ndimage
import glob
import math
import matplotlib.pyplot as plt

from colour_demosaicing import (
    demosaicing_CFA_Bayer_bilinear,
    demosaicing_CFA_Bayer_DDFAPD,
    demosaicing_CFA_Bayer_Malvar2004,
    mosaicing_CFA_Bayer)

import nn_model

comparison_data_path = "data/Comparison"
weights_path = "checkpoints/weights.100-0.00.hdf5"

# =========== Utility ==========

def calc_cpsnr(gt_image, test_image):
    '''Calculates color peak SNR between two images.  Returns a value in decibels.'''
    mse = ((test_image - gt_image) ** 2).mean()
    if mse != 0:
        return 20 * math.log10(1.0) - 10 * math.log10(mse)
    else:
        return math.inf

# =========== Initialize NN model object ==========
nn_m = nn_model.nn_model()
model = nn_m.setup_model()

# ==================== Data preparation ====================

# For evaluation, all test image patches must be equal in shape to the learned CFA.

types = (comparison_data_path + '/*.png', comparison_data_path + '/*.jpg')  # the tuple of file types
files_grabbed = []
for ext in types:
    files_grabbed.extend(glob.glob(ext))

print("Reading images...")
dataset = np.zeros((len(files_grabbed), nn_m.image_shape[0], nn_m.image_shape[1], 3))

idx = 0
for file in files_grabbed:
    dataset[idx] = ndimage.imread(file, mode='RGB').astype(np.float32) / 255.0  # Normalize images 0-1
    idx += 1

# Create a RGGB-mosaiced version of the input dataset to compare against more traditional demosaicing
dataset_mosaiced = np.zeros((dataset.shape[0], dataset.shape[1], dataset.shape[2]))

for i in range(dataset.shape[0]):
    dataset_mosaiced[i] = mosaicing_CFA_Bayer(dataset[i], pattern="RGGB")

# ==================== Evaluation ====================

desc_list = ["Bilinear",
             "Malvar (2004)",
             "DDFAPD",
             "Learned"]

# Indexing:  Algorithm - Test Image - Row - Column - RGB channel
results = np.zeros((len(desc_list), dataset.shape[0], dataset.shape[1], dataset.shape[2], dataset.shape[3]))

model.compile(optimizer='adam',
              loss='mean_squared_error')
model.load_weights(weights_path)
model.summary()
results_nn = model.predict(dataset)


for i in range(dataset.shape[0]):
    results[0][i] = demosaicing_CFA_Bayer_bilinear(dataset_mosaiced[i], pattern="RGGB")
    results[1][i] = demosaicing_CFA_Bayer_Malvar2004(dataset_mosaiced[i], pattern="RGGB")
    results[2][i] = demosaicing_CFA_Bayer_DDFAPD(dataset_mosaiced[i], pattern="RGGB")
    results[3][i] = results_nn[i]

psnrs = np.zeros((dataset.shape[0], len(desc_list)))

for alg_idx in range(len(desc_list)):
    for img_idx in range(dataset.shape[0]):
        psnrs[img_idx][alg_idx] = calc_cpsnr(dataset[img_idx], results[alg_idx][img_idx])

for img_idx in range(dataset.shape[0]):
    print(files_grabbed[img_idx] + ":")
    for alg_idx in range(len(desc_list)):
        print(desc_list[alg_idx] + ": " + str(psnrs[img_idx][alg_idx]) + " dB")

    print("")

# plot everything

fig, ax = plt.subplots(nrows=dataset.shape[0], ncols=len(desc_list)+1 )

for img_idx in range(0, dataset.shape[0]):
    gt_idx = img_idx * (len(desc_list) + 1)
    ax[img_idx, 0].imshow(dataset[img_idx])
    ax[img_idx, 0].xaxis.set_visible(False)
    ax[img_idx, 0].yaxis.set_visible(False)
    ax[img_idx, 0].set_title("Ground Truth")

    for alg_idx in range(0, len(desc_list)):
        ax_idx = img_idx * (len(desc_list) + 1) + 1 + alg_idx

        ax[img_idx, alg_idx+1].imshow(results[alg_idx][img_idx])
        ax[img_idx, alg_idx+1].xaxis.set_visible(False)
        ax[img_idx, alg_idx+1].yaxis.set_visible(False)
        ax[img_idx, alg_idx+1].set_title(desc_list[alg_idx] + "\nCPSNR: %0.02f" % psnrs[img_idx][alg_idx] )

plt.show()
