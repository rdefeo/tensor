from os import listdir
from os.path import isfile, join
from pymongo import MongoClient
import numpy as np
from cv2 import imread
import logging
from collections import defaultdict
from bson.objectid import ObjectId
import improc.features.preprocess as preprocess


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.info('Starting logger for image grabber.')

IMG_SIZE = 100


def pymongo_init_working_collection():
    client = MongoClient('localhost', 27017)
    test_db = client.jemboo
    collection = test_db.shoes
    return collection


def remove_dict_key(d, key):
    r = dict(d)
    del r[key]
    return r


def get_dict_size(sorted_dict):
    size = 0
    for key in sorted_dict:
        size += len(sorted_dict[key])
    return size


def get_imgs_id(path):
    """Gets the images ids.

    Given a path, it returns a list of the
    id of the images present in the folder.

    Args:
        path (str): images directory

    Returns:
        list: list of images ids
    """
    ids = []
    for i in listdir(path):
        product_id = i[:24]
        img_id = i[25:49]
        ids.append((product_id, img_id))
    return ids


def retrieve_img_data_from_db(img_id, product_id, collection):
    """Retrieve img data from db.

    Given a mongodb collection and product and image ids,
    it returns the data associated with the queried img.

    Args:
        img_id (str): id of the image
        product_id (str): mongodb collection where to look for
        collection (pymongo.collection): dictionary of img data from the db

    Returns:
        dict: img data stored in the db
    """
    db_products = collection.find({"_id": ObjectId(product_id)})
    for db_product in db_products:
        for db_img in db_product['images']:
            if str(db_img['_id']) == img_id:
                return db_img
        return None


def get_img_file_path(img_id, path):
    """
    Args:
        img_id (str): image id
        path (str): img filepath

    Returns:
        str: img filepath
    """
    for i in listdir(path):
        if isfile(join(path, i)) and img_id in i:
            return join(path, i)
    return None


def sort_imgs_in_path(path):
    """Sort images by orientation.

    Images in a folder are sorted by their orientation.
    It returns a dictionary whose keys are defined as:
    "roll_pitch_yaw".

    Args:
        path (str): imgs directory

    Returns:
        dict: dictionary of imgs sorted by orientation
    """
    imgs_by_rpy = defaultdict(list)
    local_imgs = get_imgs_id(path)
    coll = pymongo_init_working_collection()

    for (product, img) in local_imgs:
        img_db = retrieve_img_data_from_db(img, product, coll)
        if img_db is not None:
            if 'x' in img_db and 'y' in img_db and 'z' in img_db:
                rpy = str(img_db['x']) + '_' + str(img_db['y']) + '_' + str(img_db['z'])
                imgs_by_rpy[rpy].append(str(img))
            else:
                imgs_by_rpy['invalid_orientation'].append(str(img))
    return imgs_by_rpy


def clean_set(sorted_images_dict):
    """Delete unwanted orientations from a dict.

    Args:
        sorted_images_dict (dict):

    Returns:
        dict: cleaned dictionary
    """
    sorted_images_dict = remove_dict_key(sorted_images_dict, '0_90_270')
    sorted_images_dict = remove_dict_key(sorted_images_dict, '0_0_315')
    sorted_images_dict = remove_dict_key(sorted_images_dict, 'invalid_orientation')
    return sorted_images_dict


def create_label_one_hot_mapping(sorted_dict):
    """Generates a mapping from labels to one-hot vectors.

    Automagically generates a dictionary that maps a
    set of keys in a dictionary to a one-hot vector.

    Example:
        mapping["roll_pitch_yaw"] = [1, 0, 0, 0, 0]

    Args:
        sorted_dict (dict): dict of images sorted by orientation

    Returns:
        dict: dict that maps each key into a one-hot vector
    """
    mapping = defaultdict(lambda: np.ndarray((1, len(sorted_dict.keys()))))
    one_hot = np.zeros((1, len(sorted_dict.keys())))
    one_hot[0][-1] = 1
    for key in sorted_dict:
        one_hot = np.roll(one_hot, 1)
        mapping[key] = one_hot
    return mapping


def acquire_img(img_id, path):
    """Retrieve and process img from folder.

    Retrieve image from folder, scale it, applies
    grayscale, and reshape it to (1, 100, 100)

    Args:
        img_id (str):
        path (str):

    Returns:
        numpy.array: img array
    """
    img_path = get_img_file_path(img_id, path)
    img = imread(img_path)
    img_gray = preprocess.grey(img)
    img_resized = preprocess.scale_max(img_gray, IMG_SIZE, IMG_SIZE)
    img_reshaped = np.reshape(img_resized, (1, 100, 100))
    return img_reshaped


def shuffle_in_unison(a, b):
    """Shuffles two arrays maintaining relative ordering.

    Args:
        a (numpy.array): 1st array
        b (numpy.array): 2nd array

    Returns:
        numpy.array: 1st shuffled array
        numpy.array: 2nd shuffled array
    """
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def from_dict_to_arrays(sorted_dict, path, mapping):
    """Generates arrays to be fed to the net.

    Transform a sorted dictionary in two consistently ordered arrays.
    Each dictionary key ("label") is translated into a one-hot array
    according to a mapping function.

    Args:
        sorted_dict (dict): dict of img id sorted by orientation
        path (str): img files path
        mapping (dict): dict mapping orientations to one-hot vectors

    Returns:
        array: images array
        array: labels array
    """
    i = 0
    dataset_size = get_dict_size(sorted_dict)
    dataset_img = np.empty((dataset_size, IMG_SIZE, IMG_SIZE))
    dataset_label = np.empty((dataset_size, len(mapping.keys())))
    for individual in xrange(dataset_size):
        for orientation, img_list in sorted_dict.iteritems():
            for img_id in img_list:
                img = acquire_img(img_id, path)
                label = mapping[orientation]
                dataset_img[individual] = img
                dataset_label[individual] = label
                i += 1
                if i % 100 == 0:
                    print i
    return dataset_img, dataset_label


def part_data(sorted_dict, boundary_percentage):
    """Parts lists belonging to a dict.

    Each list is parted in two. "boundary percentage" specify
    the size of the first list compared to the original one.
    Two dict are returned, one containing all right hand lists
    and one containing all left hand ones.

    Args:
        sorted_dict (dict):
        boundary_percentage (float):

    Returns:
        dict: dict right hand lists
        dict: dict left hand lists
    """
    right_set = defaultdict(list)
    left_set = defaultdict(list)
    for key in sorted_dict:
        left_set[key] = sorted_dict[key][:(int(len(sorted_dict[key]) * boundary_percentage / 100))]
        right_set[key] = sorted_dict[key][(int(len(sorted_dict[key]) * boundary_percentage / 100)):]
    return left_set, right_set


def save_dataset_array_to_file(dataset, destination):
    np.save(destination, dataset)


def load_dataset_from_file(source):
    dataset = np.load(source)
    return dataset


class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(784)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class DataSets(object):
        pass


def read_data_sets(train_dir, test_percentage, validation_percentage, fake_data=False):
    """Reads the images in a directory and sort them in three sets for ANN feeding.

    Args:
        train_dir (str): Data points images directory
        test_percentage (float): Percentage of data points to be used exclusively for test
        validation_percentage (float): Percentage of training data points to be used specifically for validation
        fake_data (bool): True if you need to initialize a dummy element

    Returns:
        DataSets: Incorporates training, validation and test set
    """

    data_sets = DataSets()

    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets

    LOGGER.info("Sorting imgs in training set folder...")
    sorted_imgs = sort_imgs_in_path(train_dir)
    sorted_imgs = clean_set(sorted_imgs)

    LOGGER.info("Partitioning data into training, validation and test groups...")
    label_to_one_hot_map = create_label_one_hot_mapping(sorted_imgs)
    test_dict, train_and_validation_dict = part_data(sorted_imgs, test_percentage)
    validation_dict, train_dict = part_data(train_and_validation_dict, validation_percentage)
    del train_and_validation_dict
    LOGGER.info("Retrieving and processing %i train images...", get_dict_size(train_dict))
    train_images, train_labels = from_dict_to_arrays(train_dict, train_dir, label_to_one_hot_map)
    LOGGER.info("Retrieving and processing %i validation images...", get_dict_size(validation_dict))
    validation_images, validation_labels = from_dict_to_arrays(validation_dict, train_dir, label_to_one_hot_map)
    LOGGER.info("Retrieving and processing %i test images...", get_dict_size(test_dict))
    test_images, test_labels = from_dict_to_arrays(test_dict, train_dir, label_to_one_hot_map)
    LOGGER.info("Shuffling all data...")
    train_images, train_labels = shuffle_in_unison(train_images, train_labels)
    validation_images, validation_labels = shuffle_in_unison(validation_images, validation_labels)
    test_images, test_labels = shuffle_in_unison(test_images, test_labels)
    LOGGER.info("Datasets ready.")

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.validation = DataSet(validation_images, validation_labels)
    data_sets.test = DataSet(test_images, test_labels)

    return data_sets
