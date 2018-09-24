import numpy as np
from scipy import ndimage
import glob
from tqdm import tqdm


def create_training_data(patch_shape):
    """Loads and prepares a dataset from stored images on disk."""

    data_path = "./data/Flickr500"

    num_patch_centers_per_image = 16

    types = (data_path + '/*.png', data_path + '/*.jpg')  # the tuple of file types
    files_grabbed = []
    for ext in types:
        files_grabbed.extend(glob.glob(ext))

    print("Generating dataset...")
    dataset = np.zeros((len(files_grabbed) * num_patch_centers_per_image * patch_shape[0] * patch_shape[1],
                        patch_shape[0], patch_shape[1], 3))

    idx = 0
    for file in tqdm(files_grabbed):
        img = ndimage.imread(file, mode='RGB').astype(np.float32) / 255.0  # Normalize images 0-1

        for i in range(num_patch_centers_per_image):
            start_row = np.random.randint(0, img.shape[0]-patch_shape[0]*2)
            start_col = np.random.randint(0, img.shape[1]-patch_shape[1]*2)

            for dr in range(0,patch_shape[0]):
                for dc in range(0, patch_shape[1]):
                    dataset[idx] = img[start_row + dr:start_row + dr + patch_shape[0], start_col + dc:start_col + dc + patch_shape[1], :]
                    idx += 1

    print("Generated dataset of %d image patches from %d files." % (len(dataset), len(files_grabbed)))
    return dataset

if __name__ == "__main__":
    data = create_training_data((32,32))
    print(data.shape)
