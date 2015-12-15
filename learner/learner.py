from os import listdir
import input_data
from os.path import exists
import logging
import numpy as np
from cv2 import imread, imshow, waitKey
import improc.features.preprocess as preprocess

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.info('Starting logger for image grabber.')

TRAINING_SET_DIR = '../grabber/out'

if not (exists(TRAINING_SET_DIR)) or listdir(TRAINING_SET_DIR) == []:
    print "Training set folder does not exist or is empty."
    exit()


# LOGGER.info("Sorting imgs in training set folder...")
# training_set_img_id = input_data.sort_imgs_in_path(TRAINING_SET_DIR)
# training_set_img_id = input_data.clean_set(training_set_img_id)
# LOGGER.info("Sorting complete!")
#
# input_data.from_dict_to_arrays(training_set_img_id)

dataset_array = np.empty((0, 100, 100))
img1 = input_data.acquire_img('55c59ece27aae77a50b3a2a9', TRAINING_SET_DIR)
img2 = input_data.acquire_img('55c59ece27aae77a50b3a2a9', TRAINING_SET_DIR)
dataset_array = np.append(dataset_array, img1, axis=0)
dataset_array = np.append(dataset_array, img2, axis=0)
print dataset_array.shape


