# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:59:27 2019

@author: Andrew
"""
import cv2
import numpy as np
from copy import copy
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from constants import Constants
from images import *
from scipy.fftpack import rfft
from skimage.transform import radon

class FeatureGenerator(ImageGenerator):

    def get_features(self, num_images, classes = None):
        images, labels = self.get_images(num_images, classes)
        x = []
        for image in images:
            img_features = self.image_features(image)
            x.append(img_features)
        x = np.vstack(x)
        return x, labels, images

    def image_features(self, image):
        color_feature_functions = [get_color_histogram, sobel_hist]
        gray_feature_functions = [lbp, gabor_sums, radon_hists]
        fft_feature_functions = [lambda x: im2hist(x, 10), radon_hists]
        features = []
        def add_features(i, funcs):
            for func in funcs:
                featureset = func(i).ravel()
                features.append(featureset)
        add_features(image, color_feature_functions)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        add_features(gray_img, gray_feature_functions)
        fft_image = rfft(gray_img)
        add_features(fft_image, fft_feature_functions)
        return np.hstack(features).ravel()

def get_color_histogram(image, bins = 5,
                        additional_color_spaces = [
                             cv2.COLOR_BGR2YCrCb,
                             cv2.COLOR_BGR2GRAY]):
    features = []
    features.append( get_colorspace_histogram(image, bins = bins) )
    for color_space in additional_color_spaces:
        converted_image = cv2.cvtColor(copy(image), color_space)
        features.append( get_colorspace_histogram(converted_image, bins = bins) )
    features = np.hstack(features)
    return features

def get_colorspace_histogram(image, bins = 10):
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

def radon_hists(image, bins = 5):
    radon_image = radon(image,
          theta = np.linspace(0, 180, np.max(image.shape)),
          circle = False)
    return im2hist(radon_image, bins)

def lbp(img, bins = 5):
    lbp_image = local_binary_pattern(img, 4, 2, method = 'uniform')
    return im2hist(lbp_image, bins)

def im2hist(img, bins):
    return np.histogram(img, bins = bins, density = True)[0].astype('float32')

def gabor_sums(img, bins = 5):
    #gabor based features based on wndcharm
    vector = np.empty((bins,))
    integrate = lambda i: np.sum(i)/np.ones(i.shape).sum()
    gab = lambda f: integrate(gabor(img, f, theta = np.pi/2)[0])
    lower_bound = 0
    f = 0
    while lower_bound <= 0:
        f += .1
        lower_bound = gab(f + .1)
    for f in np.arange(1, bins + 1):
        vector[f-1] = gab(f)/lower_bound
    return vector.astype('float32')

def sobel_hist(img, bins = 5):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    hist = np.hstack([im2hist(gx,bins), im2hist(gy,bins)])
    return hist


get_classes = lambda file_dict: np.hstack([k*np.ones((len(v), )) for k,v in file_dict.items()]).astype('int32')
