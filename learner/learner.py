from os import listdir
import input_data
from os.path import exists
import logging
import numpy as np
from cv2 import imread
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

dataset_array = np.ndarray((100, 100, 1))
img = imread('/home/alessio/tensor/grabber/out/543d4b96186d8961e28caa35_55c59ece27aae77a50b3a2a9.jpg')
img = preprocess.grey(img)
img = preprocess.scale_max(img, 100, 100)
img = np.resize(img, (100, 100, 1))
label = np.zeros((5, 1))
data = np.append(dataset_array, img, axis=2)
print data

