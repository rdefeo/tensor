# from scipy.ndimage import imread
from scipy.misc import imshow
from cv2 import imread
import improc.features.preprocess as preprocess
from improc.features.descriptor import ZernikeDescriptor



img = imread('/home/alessio/Desktop/shoe.jpg')
# imshow(img)
# img = img.convertTo(img, CV_32SC1)
# img = preprocess.autocrop(img)
# img = preprocess.blur(
#     img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
#     "sigmaX": 0}
# )

img = preprocess.autocrop(img)
img = preprocess.blur(
    img, gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
    "sigmaX": 0}
)
imshow(img)
img = preprocess.grey(img)
img = preprocess.bitwise(img)
img = preprocess.canny(img)
imshow(img)

# descriptor = ZernikeDescriptor(
#     preprocess=True,
#     radius=84,
#     resize={"enabled": False, "width": 250, "height": 250},
#     grey={"enabled": True},
#     autocrop={"enabled": True},
#     outline_contour={"enabled": True},
#     add_border={"enabled": True, "color_value": 0, "border_size": 15,
#                 "fill_dimensions": True},
#     bitwise_info={"enabled": True},
#     thresh={"enabled": False},
#     scale_max={"enabled": True, "width": 250, "height": 250},
#     dilate={"enabled": True, "width": 7, "height": 7, "iterations": 1},
#     closing={"enabled": True, "width": 5, "height": 5},
#     canny={"enabled": True, "threshold1": 100, "threshold2": 200},
#     gaussian_blur={"enabled": True, "ksize_width": 5, "ksize_height": 5,
#                    "sigmaX": 0},
#     laplacian={"enabled": False})
#
# img = descriptor.do_preprocess(img)
# imshow(img)
