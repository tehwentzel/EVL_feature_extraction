# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:59:27 2019

@author: Andrew
"""
import cv2
import numpy as np
from copy import copy
from skimage.feature import local_binary_pattern
from constants import Constants
from images import *

def get_histogram(image, bins = 5):
    ##extracts a histogram from all the color channels in an image
    if len(image.shape) > 2: #check if grayscale
        n_channels = image.shape[2]
    else:
        n_channels = 1
    hists = []
    for channel in range(n_channels):
        if n_channels > 1:
            img_slice = image[:,:,channel]
        else:
            img_slice = image
        color_histogram = np.histogram(img_slice, 
                                       bins = bins, 
                                       density = True, 
                                       range = (0, 255))[0]
        hists.append(color_histogram.astype('float32'))
    return np.hstack(hists)


def get_color_histograms(images, 
                         bins = 5,
                         additional_color_spaces = [
                                 cv2.COLOR_BGR2YCrCb,
                                 cv2.COLOR_BGR2GRAY]):
    #assumes this is passed a list of images in bgr color space
    all_features = []
    for image in images:
        features = []
        features.append( get_histogram(image, bins = bins) )
        for color_space in additional_color_spaces:
            converted_image = cv2.cvtColor(copy(image), color_space)
            features.append( get_histogram(converted_image, bins = bins) )
        features = np.hstack(features)
        all_features.append(features)
    return np.vstack(all_features)

def get_lbp_texture(file_dict, bin_size = 15):
    pass

get_classes = lambda file_dict: np.hstack([k*np.ones((len(v), )) for k,v in file_dict.items()]).astype('int32')
