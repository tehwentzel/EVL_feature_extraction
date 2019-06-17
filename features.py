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


def get_color_histograms(image, 
                         bins = 5,
                         additional_color_spaces = [
                                 cv2.COLOR_BGR2YCrCb,
                                 cv2.COLOR_BGR2GRAY]):
    #assumes this is passed a list of images in bgr color space
    features = []
    features.append( get_histogram(image, bins = bins) )
    for color_space in additional_color_spaces:
        converted_image = cv2.cvtColor(copy(image), color_space)
        features.append( get_histogram(converted_image, bins = bins) )
    features = np.hstack(features)
    return features

##much slower than opencv
def get_hog(image, n_bins = 15, segments = 10, cell_grid = (8,8)):
    if len(image.shape) > 2:
        img = cv2.cvtColor(copy(image), cv2.COLOR_BGR2GRAY)
    else:
        img = copy(image)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx,gy)
    magnitude = magnitude.ravel()
    angle = (angle % np.pi).ravel() #I don't really thing we need positive/negative gradients here?
    histogram = np.zeros((n_bins,))
    bin_width = np.pi/n_bins
    for idx in range(len(magnitude)):
        histogram[ int(angle[idx]//bin_width) ] += magnitude[idx]  
    return histogram/np.linalg.norm(histogram)
    
def get_hog_descriptors(image):
    hog = cv2.HOGDescriptor('hog.xml')
    value =  hog.compute(image)
    return value.ravel()

def get_lbp_texture(file_dict, bin_size = 15):
    pass

def get_features(images, functions = [get_hog, get_color_histograms]):
    #assumes functions is a list of functions that all return numpy 1d arrays (histograms basically)
    all_features = []
    for image in images:
        features = []
        for func in functions:
            features.append( func(image) )
        features = np.hstack(features)
        all_features.append(features)
    return np.vstack(all_features)

get_classes = lambda file_dict: np.hstack([k*np.ones((len(v), )) for k,v in file_dict.items()]).astype('int32')
