from os import listdir
from os.path import isfile, join, exists
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

TRAINING_SET_DIR = '../grabber/out'
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
    ids = []
    for i in listdir(path):
        product_id = i[:24]
        img_id = i[25:49]
        ids.append((product_id, img_id))
    return ids


def retrieve_img_data_from_db(img_id, product_id, collection):
    db_products = collection.find({"_id": ObjectId(product_id)})
    for db_product in db_products:
        for db_img in db_product['images']:
            if str(db_img['_id']) == img_id:
                return db_img
        return None


def get_img_file_path(img_id, path):
    for i in listdir(path):
        if isfile(join(path, i)) and img_id in i:
            return join(path, i)
    return None


def sort_imgs_in_path(path):
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
    sorted_images_dict = remove_dict_key(sorted_images_dict, '0_90_270')
    sorted_images_dict = remove_dict_key(sorted_images_dict, '0_0_315')
    sorted_images_dict = remove_dict_key(sorted_images_dict, 'invalid_orientation')
    return sorted_images_dict


def create_label_one_hot_mapping(sorted_dict):
    mapping = defaultdict(lambda: np.ndarray((1, len(sorted_dict.keys()))))
    one_hot = np.zeros((1, len(sorted_dict.keys())))
    one_hot[0][-1] = 1
    for key in sorted_dict:
        one_hot = np.roll(one_hot, 1)
        mapping[key] = one_hot
    print mapping


def acquire_img(img_id, path):
    img_path = get_img_file_path(img_id, path)
    img = imread(img_path)
    img_gray = preprocess.grey(img)
    img_resized = preprocess.resize(img_gray, IMG_SIZE)
    return img_resized


def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def from_dict_to_arrays(sorted_dict, path, mapping):
    dataset_img = np.ndarray((IMG_SIZE, IMG_SIZE, 1))
    dataset_label = np.ndarray((1, len(mapping.keys())))
    for orientation, img_id in sorted_dict:
        img = acquire_img(img_id, path)
        label = mapping[orientation]
        # TODO: check how to stack imgs and labels


def part_data(sorted_dict, test_percentage):
    # TODO return 2 sets from 1
    return sorted_dict


class DataSet(object):
    def __init__(self, images, labels, fake_data=False, one_hot=False):
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


# def read_data_sets(train_dir, fake_data=False, one_hot=False):
#     class DataSets(object):
#         pass
#
#     data_sets = DataSets()
#
#     if fake_data:
#         data_sets.train = DataSet([], [], fake_data=True, one_hot=one_hot)
#         data_sets.validation = DataSet([], [], fake_data=True, one_hot=one_hot)
#         data_sets.test = DataSet([], [], fake_data=True, one_hot=one_hot)
#         return data_sets
#
#     sorted_imgs = sort_imgs_in_path(train_dir)
#
#     VALIDATION_SIZE = 5000
#
#     local_file = maybe_download(TRAIN_IMAGES, train_dir)
#     train_images = extract_images(local_file)
#
#     local_file = maybe_download(TRAIN_LABELS, train_dir)
#     train_labels = extract_labels(local_file, one_hot=one_hot)
#
#     local_file = maybe_download(TEST_IMAGES, train_dir)
#     test_images = extract_images(local_file)
#
#     local_file = maybe_download(TEST_LABELS, train_dir)
#     test_labels = extract_labels(local_file, one_hot=one_hot)
#
#     validation_images = train_images[:VALIDATION_SIZE]
#     validation_labels = train_labels[:VALIDATION_SIZE]
#     train_images = train_images[VALIDATION_SIZE:]
#     train_labels = train_labels[VALIDATION_SIZE:]
#
#     data_sets.train = DataSet(train_images, train_labels)
#     data_sets.validation = DataSet(validation_images, validation_labels)
#     data_sets.test = DataSet(test_images, test_labels)
#
#     return data_sets
