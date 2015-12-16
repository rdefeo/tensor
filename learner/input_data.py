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


def get_imgs_id(path):
    """Given a path, it returns a list of the ids of the
       images present in the folder.
    :param path: images directory.
    :return:     list of images id.
    """
    ids = []
    for i in listdir(path):
        product_id = i[:24]
        img_id = i[25:49]
        ids.append((product_id, img_id))
    return ids


def retrieve_img_data_from_db(img_id, product_id, collection):
    """Given a mongodb collection and product and image ids,
       it returns the data associated with the queried img.
    :param img_id:     id of the image.
    :param product_id: id of the product the image belongs to.
    :param collection: mongodb collection where to look for.
    :return:           dictionary of img data from the db.
    """
    db_products = collection.find({"_id": ObjectId(product_id)})
    for db_product in db_products:
        for db_img in db_product['images']:
            if str(db_img['_id']) == img_id:
                return db_img
        return None


def get_img_file_path(img_id, path):
    """
    :param img_id: image id
    :param path:   images directory
    :return:       img filepath
    """
    for i in listdir(path):
        if isfile(join(path, i)) and img_id in i:
            return join(path, i)
    return None


def sort_imgs_in_path(path):
    """Sort images id from a directory into a dictionary
       by their orientation.
    :param path: imgs directory.
    :return:     sorted dictionary.
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
    """Delete some invalid orientations for a sorted image dictionary.
    :param sorted_images_dict: dict to be cleaned.
    :return: clean dict.
    """
    sorted_images_dict = remove_dict_key(sorted_images_dict, '0_90_270')
    sorted_images_dict = remove_dict_key(sorted_images_dict, '0_0_315')
    sorted_images_dict = remove_dict_key(sorted_images_dict, 'invalid_orientation')
    return sorted_images_dict


def create_label_one_hot_mapping(sorted_dict):
    """Automagically generates a dictionary that maps a
       set of keys in a dictionary to a one-hot vector.
    :param sorted_dict:
    :return:
    """
    mapping = defaultdict(lambda: np.ndarray((1, len(sorted_dict.keys()))))
    one_hot = np.zeros((1, len(sorted_dict.keys())))
    one_hot[0][-1] = 1
    for key in sorted_dict:
        one_hot = np.roll(one_hot, 1)
        mapping[key] = one_hot
    return mapping


def acquire_img(img_id, path):
    """Looks in a path for an img file given its id,
       and applies a small preprocessing
    :param img_id: img id.
    :param path:   img directory.
    :return:       processed cvimage (numpy array).
    """
    img_path = get_img_file_path(img_id, path)
    img = imread(img_path)
    img_gray = preprocess.grey(img)
    img_resized = preprocess.scale_max(img_gray, IMG_SIZE, IMG_SIZE)
    return img_resized


def shuffle_in_unison(a, b):
    """Shuffle two arrays along their first dimension
       so that their relative order is kept unchanged.
    :param a: 1st array.
    :param b: 2nd array.
    :return:  Shuffled arrays.
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
    """Transform a sorted dictionary in two consistently ordered arrays.
       Each dictionary key ("label") is translated into a one-hot array
       according to a mapping function.
    :param sorted_dict: Dictionary to be processed ("label": "img_id").
    :param path: Path where to search for img files ("img_id" is in their name).
    :param mapping: Mapping function from label to one-hot vector.
    :return: img and label dataset.
    """
    dataset_img = np.empty((0, IMG_SIZE, IMG_SIZE))
    dataset_label = np.empty((0, len(mapping.keys())))
    for orientation, img_list in sorted_dict.iteritems():
        for img_id in img_list:
            img = acquire_img(img_id, path)
            label = mapping[orientation]
            dataset_img = np.append(dataset_img, img, axis=0)
            dataset_label = np.append(dataset_label, label, axis=0)
    return dataset_img, dataset_label


def part_data(sorted_dict, boundary_percentage):
    """Parts a dictionary of lists in two parts,
       so that every element is similarly resized.
    :param sorted_dict:         Dictionary to be parted.
    :param boundary_percentage: Size of the left element expressed as
                                percentage of the original dictionary.
    :return: Right and left partition.
    """
    right_set = defaultdict(list)
    left_set = defaultdict(list)
    for key in sorted_dict:
        right_set[key].append(sorted_dict[key][:(int(len(sorted_dict[key]) * boundary_percentage / 100))])
        left_set[key].append(sorted_dict[key][(int(len(sorted_dict[key]) * boundary_percentage / 100)):])
    return left_set, right_set


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


def read_data_sets(train_dir, test_percentage, validation_percentage, fake_data=False):
    """Reads the images in a directory and sort them in three sets for ANN feeding.
    :param train_dir: Data points images directory
    :param test_percentage: Percentage of data points to be used exclusively for test
    :param validation_percentage: Percentage of training data points to be used specifically for validation
    :param fake_data: Set True if you need to initialize a dummy element
    :return: Object that incorporates the three sets
    """
    class DataSets(object):
        pass

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
    LOGGER.info("Retrieving and processing images...")
    train_images, train_labels = from_dict_to_arrays(train_dict, train_dir, label_to_one_hot_map)
    validation_images, validation_labels = from_dict_to_arrays(validation_dict, train_dir, label_to_one_hot_map)
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
