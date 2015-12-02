from os import listdir
from os.path import isfile, join

import itertools as it
import numpy as np

from cv2 import waitKey
from cv2 import imshow
from cv2 import imread
from cv2 import cvtColor
from cv2 import COLOR_BGR2RGB
from cv2 import imwrite

import improc.features.preprocess as preprocess
from improc.features.descriptor import ZernikeDescriptor


def grouper(n, iterable, fillvalue=None):
    '''grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx'''
    args = [iter(iterable)] * n
    return it.izip_longest(fillvalue=fillvalue, *args)


def mosaic(w, imgs):
    '''Make a grid from images.

    w    -- number of grid columns
    imgs -- images (must have same size and format)
    '''
    imgs = iter(imgs)
    img0 = imgs.next()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = grouper(w, imgs, pad)
    return np.vstack(map(np.hstack, rows))


def preprocess_basic(_image_files):
    processed_images = []

    for image_file in _image_files:
        img = imread(image_file)
        img = preprocess.autocrop(img)
        img = preprocess.scale_max(img)
        img = preprocess.make_square(img)
        processed_images.append(img)

    mosaic_image = mosaic(len(processed_images), processed_images)
    return  mosaic_image


def preprocess_super_simple(_image_files):
    processed_images = []
    for image_file in _image_files:
        img = imread(image_file)
        img = preprocess.autocrop(img)
        img = preprocess.scale_max(img)
        img = preprocess.make_square(img)
        img = preprocess.blur(
            img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
            "sigmaX": 0}
        )
        img = preprocess.grey(img)
        img = preprocess.bitwise(img)
        img = preprocess.canny(img)
        processed_images.append(img)

    mosaic_image = mosaic(len(processed_images), processed_images)
    return mosaic_image


def preprocess_zernike(_image_files):
    descriptor = ZernikeDescriptor()
    procesessed_images = []
    for image_file in _image_files:
        img = imread(image_file)
        img = preprocess.autocrop(img)
        img = descriptor.do_preprocess(img)
        procesessed_images.append(img)
        print img.shape
    mosaic_images = mosaic(len(procesessed_images), procesessed_images)
    return mosaic_images


shoes_path = 'shoes'
image_files = [join(shoes_path, f) for f in listdir(shoes_path) if isfile(join(shoes_path, f))]

original_imgs = preprocess.grey(preprocess_basic(image_files))  # made gray just to be shown with the others
gray_imgs = preprocess_super_simple(image_files)
zernike_images = preprocess_zernike(image_files)

print original_imgs.shape
print gray_imgs.shape
print zernike_images.shape

comparison = np.concatenate((original_imgs, gray_imgs, zernike_images), axis=0)
imshow("Descriptors comparison", comparison)
waitKey(0)

imwrite("out/comparison.jpg", comparison)

