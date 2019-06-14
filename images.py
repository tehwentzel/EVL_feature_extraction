import cv2
import numpy as np
import glob
from copy import copy
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
from skimage.feature import local_binary_pattern
from constants import Constants

def get_image_files(root = 'data\images_2\**\*.jpg',
                    classes = None):
    if classes is None:
        classes = Constants.classes
    files = glob.glob(root, recursive = True)
    files = list(set(files))
    images = OrderedDict()
    for c in classes:
        images[classes.index(c)] = []
        pattern = re.compile(c)
        new_files = copy(files)
        for file in files:
            if pattern.search(file) is not None:
                image = cv2.imread(file)
                if image is not None and image.std() > 0.0001:
                    images[classes.index(c)].append(file)
                new_files.remove(file)
        files = new_files
    return images

def get_unsegmented_image_files(root = "data\samples*\*.bmp"):
    files = glob.glob(root)
    bad_files = []
    for file in files:
        if cv2.imread(file) is None:
            bad_files.append(file)
    for file in bad_files:
        files.remove(file)
    return files

def gaussian_blur_channels(image, kernel = (3,3), mean = 0):
    if len(image.shape) <= 2:
        return cv2.GaussianBlur(image, kernel, mean)
    else:
        n_channels = image.shape[2]
        for channel in range(n_channels):
            image[:,:,channel] = cv2.GaussianBlur(image[:,:,channel], kernel, mean)
    return image

def bilateral_blur(img, diameter = 20, sigmaColor = 20, sigmaSpace = 20):
    image = copy(img)
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)

def edge_extraction(img, d1 = 10, d2 = 10, t1 = 20, t2 = 60):
    image = denoise(img, d1, d2)
    edges = cv2.Canny(image,t1,t2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges

def word_extraction(img, max_area_fraction = .4, 
                    lower_canny_thresh = 200, 
                    upper_canny_thresh = 300,
                    dilation_kernel = np.ones((5,5))):
    new_img = copy(img)
    if len(img.shape) > 2:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
#    new_img = cv2.bilateralFilter(new_img, 20,20, 20)
    edges = cv2.Canny(copy(new_img), lower_canny_thresh, upper_canny_thresh)
    edges = cv2.dilate(edges, dilation_kernel)
    contours, heirarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good_contours = []
    max_area = max_area_fraction*img.shape[0]*img.shape[1]
    for contour in contours:
        if cv2.contourArea(contour) < max_area:
            good_contours.append(contour)
    cv2.drawContours(new_img, good_contours, -1, 0, cv2.FILLED)
    return crop_image(img, new_img)
    
def lbp(img):
    image = denoise(img)
    lbp_image = local_binary_pattern(image, 1, 2, method = 'uniform')
    return lbp_image

def crop_image(img, reference_image = None, upper_bound = 220, lower_bound = 80):
    if reference_image is None:
        reference_image = copy(img)
#    reference_image = denoise(reference_image)
    if len(reference_image.shape) > 2:
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    image_size = (img.shape[0], img.shape[1])
    valid_coords = np.argwhere( (reference_image > lower_bound)*(reference_image < upper_bound) )
    if len(valid_coords) <= Constants.image_area/3:
        if upper_bound < 245 and lower_bound > 10:
            return crop_image(img, reference_image, upper_bound + 10, lower_bound - 10)
        else:
            print('error in cropping image')
            return img
    x0, y0 = valid_coords.min(axis = 0)
    x1, y1 = valid_coords.max(axis = 0) + 1
    if len(reference_image.shape) > 2:
        cropped_image = img[x0:x1, y0:y1]
    else:
        cropped_image = img[x0:x1, y0:y1, :]
    return cv2.resize(cropped_image, image_size)

def denoise(img, d1 = 8, d2 = 8):
    if len(img.shape) > 2:
        denoiser = lambda x: cv2.fastNlMeansDenoisingColored(x, None, d1, d2)
    else:
        denoiser = lambda x: cv2.fastNlMeansDenoising(x, None, d1, d2)
    return denoiser(img)

def show_images(image_set, compare_func = None):
    window_name = 'images'
    cv2.namedWindow(window_name)
    if compare_func is None:
        on_change = lambda x: cv2.imshow(window_name, image_set[x])
    else:
        on_change = lambda x: cv2.imshow(window_name, 
                                         np.hstack([image_set[x], compare_func(copy(image_set[x]))]))
    cv2.createTrackbar(window_name, window_name, 0, len(image_set) - 1, on_change)
    on_change(0)
    
def preprocess_dict(image_file_dict):
    images = []
    for files in image_file_dict.values():
        image_set = preprocess_list(files)
        images.extend(image_set)
    return images

def preprocess_list(image_file_list):
    image_set = []
    for file in image_file_list:
        new_img = cv2.imread(file)
        new_img = cv2.resize(new_img, Constants.image_size)
        new_img = denoise(new_img)
        new_img = word_extraction(new_img)
        image_set.append(new_img)
    return image_set

imshow = lambda x: cv2.imshow('image', x)
imshow_big = lambda x: imshow(cv2.pyrUp(x))
imshow_func = lambda function, img: imshow( np.hstack([img, function(copy(img))]) )
