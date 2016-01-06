from os import listdir
from os.path import isfile, join
from pymongo import MongoClient
import numpy as np
from cv2 import imread
import logging
import cPickle
from collections import defaultdict
from bson.objectid import ObjectId
import improc.features.preprocess as preprocess


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
logging.info('Starting logger for image grabber.')

IMG_SIZE = 100
ALLOWED_ORIENTATIONS = ['0_0_270',
                        '0_0_0',
                        '0_0_90',
                        '0_0_45',
                        '0_0_180']


def pymongo_init_working_collection():
    client = MongoClient('localhost', 27017)
    test_db = client.jemboo
    collection = test_db.shoes
    return collection


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
    id_list = listdir(path)
    for i in id_list:
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


def sort_imgs_in_path(path, max_num_imgs):
    """Sort images by orientation.

    Images in a folder are sorted by their orientation.
    It returns a dictionary whose keys are defined as:
    "roll_pitch_yaw".

    Args:
        path (str): imgs directory
        max_num_imgs (int): maximum number of images to process

    Returns:
        dict: dictionary of imgs sorted by orientation
    """
    imgs_by_rpy = defaultdict(list)
    local_imgs = get_imgs_id(path)
    coll = pymongo_init_working_collection()
    sorted_imgs_counter = 0

    for (product, img) in local_imgs:
        img_db = retrieve_img_data_from_db(img, product, coll)
        if img_db is not None:
            if 'x' in img_db and 'y' in img_db and 'z' in img_db:
                rpy = str(img_db['x']) + '_' + str(img_db['y']) + '_' + str(img_db['z'])
                if rpy in ALLOWED_ORIENTATIONS:
                    imgs_by_rpy[rpy].append(str(img))
                    sorted_imgs_counter += 1
                    if sorted_imgs_counter >= max_num_imgs:
                        break
    return imgs_by_rpy


def create_label_one_hot_mapping(sorted_dict):
    """Generates a mapping from labels to one-hot vectors.

    Automagically generates a dictionary that maps a
    set of keys in a dictionary to a one-hot vector.

    Example:
        mapping["0_90_90"] = numpy.array(1, 0, 0, 0, 0)
        mapping["0_0_180"] = numpy.array(0, 1, 0, 0, 0)
        ...

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
    dataset_size = get_dict_size(sorted_dict)
    dataset_img = np.empty((dataset_size, IMG_SIZE, IMG_SIZE))
    dataset_label = np.empty((dataset_size, len(mapping.keys())))
    individual = 0
    for orientation, img_list in sorted_dict.iteritems():
        for img_id in img_list:
            if individual % 100 == 0:
                LOGGER.info("%i images processed", individual)
            img = acquire_img(img_id, path)
            label = mapping[orientation]
            dataset_img[individual] = img
            dataset_label[individual] = label
            individual += 1
    return dataset_img, dataset_label


def split_array(array, size):
    """Splits the array in two parts. The left hand part length is given by "size"
    Args:
        array (numpy.array):
        size (int):
    Returns:
        numpy.array:
        numpy.array:
    """
    left_array = array[:size]
    right_array = array[size:]
    return left_array, right_array


def split_array_percent(array, percentage):
    """Split an array in two parts. The left part length is expressed as a percentage of the original one.
    Args:
        array (numpy.array)
        percentage (int):
    Returns:
        numpy.array:
        numpy.array:
    """
    array_dim = array.shape[0]
    left_array = array[:(array_dim * percentage / 100)]
    right_array = array[(array_dim * percentage / 100):]
    return left_array, right_array


def compare_dataset_dict(old_dict, new_dict):
    """Compare two dictionaries of lists. Each list is treated as a set to check which
    elements of the first dictionary list appears in the respective list of the second dict.
    It return the dictionaries of elements belonging uniquely to the new dictionary.
    Args:
        old_dict (dict(list)):
        new_dict (dict(list)):
    Returns:
        dict(list): dictionary of elements belonging to new_dict but not to old_dict
    """
    assert set(old_dict.keys()) == set(new_dict.keys())
    diff_dict = defaultdict(list)
    for key in old_dict:

        diff_dict[key] = list(set(new_dict[key]) - set(old_dict[key]))
    return diff_dict


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
            assert len(images.shape) == 3
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

    def increment_dataset_size(self, images, labels):
        """Increments the number of images and labels stored in a dataset."""
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape,
                                                   labels.shape))
        assert len(images.shape) == len(self._images.shape)
        assert images.shape[1] == self._images.shape[1] and images.shape[2] == self._images.shape[2]
        assert len(labels.shape) == len(self._labels.shape)
        assert labels.shape[1] == self._labels.shape[1]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._num_examples += images.shape[0]
        self._images = np.append(self._images, images, axis=0)
        self._labels = np.append(self._labels, labels, axis=0)


class DataSets(object):
    def __init__(self, train_dataset, validation_dataset, test_dataset):
        self._train = train_dataset
        self._validation = validation_dataset
        self._test = test_dataset

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test


def save_datasets_to_file(datasets, filepath):
    with open(filepath, 'wb') as output_file:
        cPickle.dump(datasets, output_file, cPickle.HIGHEST_PROTOCOL)


def load_datasets_from_file(filepath):
    with open(filepath, 'rb') as input_file:
        datasets = cPickle.load(input_file)
    return datasets


def get_datasets(train_dir, test_set_size, validation_percentage, max_num_imgs, savedir=""):
    """Reads the images in a directory and sort them in three sets for ANN feeding.
       If a save directory is specified, all datasets are saved for later use.

    Args:
        train_dir (str): Data points images directory
        test_set_size (float): Amount of data to be reserved for testing
        validation_percentage (float): Percentage of training data points to be used specifically for validation
        max_num_imgs (int): maximum number of images to process, negative values implies no bound.
        savedir (str): path where to save tmp arrays

    Returns:
        DataSets: Incorporates training, validation and test set
    """

    LOGGER.info("Sorting imgs in training set folder...")
    sorted_imgs = sort_imgs_in_path(train_dir, max_num_imgs)
    if savedir != "":
        save_datasets_to_file(sorted_imgs, savedir + "/processed_imgs.rcrd")

    LOGGER.info("Mapping labels to one hot vectors...")
    label_to_one_hot_map = create_label_one_hot_mapping(sorted_imgs)

    LOGGER.info("Retrieving images and processing into numpy arrays...")
    images, labels = from_dict_to_arrays(sorted_imgs, train_dir, label_to_one_hot_map)
    images, labels = shuffle_in_unison(images, labels)

    LOGGER.info("Slicing and saving test images...")
    test_images, train_images = split_array(images, test_set_size)
    test_labels, train_labels = split_array(labels, test_set_size)
    test_dataset = DataSet(test_images, test_labels)
    if savedir != "":
        save_datasets_to_file(test_dataset, savedir + "/test.dtst")

    LOGGER.info("Slicing train and validation set...")
    validation_images, training_images = split_array_percent(train_images, validation_percentage)
    validation_labels, training_labels = split_array_percent(train_labels, validation_percentage)
    training_dataset = DataSet(training_images, training_labels)
    validation_dataset = DataSet(validation_images, validation_labels)

    LOGGER.info("Saving train and validation set...")
    if savedir != "":
        save_datasets_to_file(training_dataset, savedir + "/training.dtst")
        save_datasets_to_file(validation_dataset, savedir + "/validation.dtst")

    LOGGER.info("Datasets ready.")

    return DataSets(training_dataset, validation_dataset, test_dataset)


def reload_session(savedir):
    training_dataset = load_datasets_from_file(savedir + "/training.dtst")
    validation_dataset = load_datasets_from_file(savedir + "/validation.dtst")
    test_dataset = load_datasets_from_file(savedir + "/test.dtst")
    return DataSets(training_dataset, validation_dataset, test_dataset)


def retrain_session(train_dir, savedir, validation_percentage, new_max_num_imgs):
    LOGGER.info("Sorting imgs in training set folder...")
    sorted_imgs = sort_imgs_in_path(train_dir, new_max_num_imgs)
    previously_sorted_imgs = load_datasets_from_file(savedir + "/processed_imgs.rcrd")
    save_datasets_to_file(sorted_imgs, savedir + "/processed_imgs.rcrd")
    new_sorted_imgs = compare_dataset_dict(previously_sorted_imgs, sorted_imgs)

    LOGGER.info("Mapping labels to one hot vectors...")
    label_to_one_hot_map = create_label_one_hot_mapping(sorted_imgs)
    del sorted_imgs
    del previously_sorted_imgs

    LOGGER.info("Retrieving test set...")
    test_dataset = load_datasets_from_file(savedir + "/test.dtst")

    LOGGER.info("Retrieving images and processing into numpy arrays...")
    images, labels = from_dict_to_arrays(new_sorted_imgs, train_dir, label_to_one_hot_map)
    images, labels = shuffle_in_unison(images, labels)
    del new_sorted_imgs

    LOGGER.info("Slicing train and validation set...")
    validation_images, training_images = split_array_percent(images, validation_percentage)
    validation_labels, training_labels = split_array_percent(labels, validation_percentage)
    training_dataset = DataSet(training_images, training_labels)
    validation_dataset = DataSet(validation_images, validation_labels)

    LOGGER.info("Saving train and validation set...")

    old_training_dataset = load_datasets_from_file(savedir + "/training.dtst")
    old_training_dataset.increment_dataset_size(training_images, training_labels)
    save_datasets_to_file(old_training_dataset, savedir + "/training.dtst")
    del old_training_dataset

    old_validation_dataset = load_datasets_from_file(savedir + "/validation.dtst")
    old_validation_dataset.increment_dataset_size(validation_images, validation_labels)
    save_datasets_to_file(old_validation_dataset, savedir + "/validation.dtst")
    del old_validation_dataset

    LOGGER.info("Datasets ready.")

    return DataSets(training_dataset, validation_dataset, test_dataset)
