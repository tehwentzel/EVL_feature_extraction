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

class FeatureGenerator(ImageGenerator):
    
    def get_features(self, num_images, classes = None):
        images, labels = self.get_images(num_images, classes)
        x = []
        for image in images:
            img_features = self.image_features(image)
            x.append(img_features)
        x = np.vstack(x)
        return x, labels
    
    def image_features(self, image):
        color_histograms = self.get_color_histogram(image)
        return color_histograms
        
    def get_color_histogram(self, image, bins = 5, 
                            additional_color_spaces = [
                                 cv2.COLOR_BGR2YCrCb,
                                 cv2.COLOR_BGR2GRAY]):
        features = []
        features.append( self.get_colorspace_histogram(image, bins = bins) )
        for color_space in additional_color_spaces:
            converted_image = cv2.cvtColor(copy(image), color_space)
            features.append( self.get_colorspace_histogram(converted_image, bins = bins) )
        features = np.hstack(features)
        return features

    def get_colorspace_histogram(self, image, bins = 5):
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
    
get_classes = lambda file_dict: np.hstack([k*np.ones((len(v), )) for k,v in file_dict.items()]).astype('int32')
