# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:07:21 2019

@author: Andrew
"""

import cv2
import numpy as np
import glob
from copy import copy
import re
import matplotlib.pyplot as plt
from collections import OrderedDict

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern, haar_like_feature
from skimage.transform.integral import integral_image
from scipy.ndimage.filters import convolve


classes = [
        'Electron-Mix',
        'Transmission',
        'ReporterANDImmunoHistochem',
        'Fluorescence-Mix',
        'Light-Mix',
        'InSituHybridization'
           ]

def get_image_files(root = 'data\images_2\**\*.jpg'):
    files = glob.glob(root, recursive = True)
    files = list(set(files))
    images = dict()
    for c in classes:
        images[classes.index(c)] = []
        pattern = re.compile(c)
        new_files = copy(files)
        for file in files:
            if pattern.search(file) is not None:
                if cv2.imread(file) is not None:
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

def get_histogram(file, bins = 5, color_space = cv2.COLOR_BGR2YCrCb):
    image = cv2.imread(file)
    image = cv2.bilateralFilter(image, 15, 20, 20)
    image = cv2.cvtColor(image, color_space)
    if len(image.shape) > 2:
        n_channels = image.shape[2]
    else:
        n_channels = 1
    hists = []
#    scale = lambda x: (x - x.min())/(x.max() - x.min())
    for channel in range(n_channels):
        if n_channels > 1:
            img_slice = image[:,:,channel]
        else:
            img_slice = image
#        image[:,:,channel] = scale(image[:,:,channel])
        color_histogram = np.histogram(img_slice, bins = bins, density = True)[0]
        hists.append(color_histogram.astype('float32'))
    return(hists)

def get_histograms(file_dict = None,
                     color_space = cv2.COLOR_BGR2YCrCb, 
                     bins = 15):
    if file_dict is None:
        file_dict = get_image_files()
    hist_distance_dict = {}
    for c in file_dict.keys():
        image_files  = file_dict[c]
        correlation = []
        position = 0
        for image_file in image_files:
            all_hists = []
            hists = get_histogram(image_file, color_space = color_space, bins = bins)
            for histogram in hists:
                all_hists.extend(histogram)
            position += 1
            correlation.append(np.hstack(all_hists).ravel())
        print(np.vstack(correlation).shape)
        hist_distance_dict[c] = np.vstack(correlation)
    return hist_distance_dict

def kernel_histogram(image, kernels, bins = 10, density = True):
    def get_hist(kernel):
        edges = convolve(image, kernel)
        hist = np.histogram(edges.ravel(), 
                            bins = bins, 
                            density = density)[0]
        return hist
    
    if isinstance(kernels, list):
        histograms = []
        for kernel in kernels:
            histograms.append(get_hist(kernel))
        return np.hstack(histograms)
    else:
        return get_hist(kernels)

top_edge_kernel = np.array([
            [1,1,1],
            [0,0,0],
            [-1,-1,-1]
        ])
bottom_edge_kernel = -1*top_edge_kernel
left_edge_kernel = np.array([
            [1,1,-1],
            [1,0,-1],
            [1,0,-1]
        ])
right_edge_kernel = -1*left_edge_kernel
all_edges = [top_edge_kernel, bottom_edge_kernel, left_edge_kernel, right_edge_kernel]

file_dict = get_image_files()

features = []
for color_space in [cv2.COLOR_BGR2GRAY, 
                    cv2.COLOR_BGR2RGB, 
                    cv2.COLOR_BGR2HSV
                    ]:
    color_hists = get_histograms(file_dict = file_dict, 
                                 color_space = color_space)
    color_matrix = np.vstack([v for v in color_hists.values()])
    features.append(color_matrix)
class_labels = np.hstack([c*np.ones((v.shape[0],)) for c,v in color_hists.items()])
bin_size = 6
n_files = np.sum([len(fileset) for fileset in file_dict.values()])
textures = np.zeros((n_files, bin_size*len(all_edges)))
t_position = 0
for c in range(max(file_dict.keys())):
    files = file_dict[c]
    matrix = np.zeros((len(files), bin_size*len(all_edges)))
    position = 0
    for file in files:
        img = cv2.imread(file)
        img = cv2.bilateralFilter(img, 20, 10, 10)
#        img = cv2.GaussianBlur(img, (3,3), 0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        histogram = kernel_histogram(img, all_edges, bins = bin_size)
#        lbp = local_binary_pattern(img, bin_size - 3, 2, method = 'ror')
#        histogram = np.histogram(lbp.ravel(), 
#                                 density = True, 
#                                 bins = bin_size,
#                                 range = (0, bin_size - 1))[0]
        matrix[position, :] = histogram
        position += 1
    end = t_position + len(files)
    textures[t_position:end, :] = matrix
    t_position = end
x = np.hstack(features)
x = np.hstack([x, textures])

x_train, x_test, y_train, y_test = train_test_split(x, class_labels, stratify = class_labels)
#pca = PCA(n_components = None)
#x_train = pca.fit_transform(x_train)
#x_test = pca.transform(x_test)

model = RandomForestClassifier(n_estimators = 60,
                               max_depth = 25,
                               min_samples_split = 2,
                               random_state = 0)
model.fit(x_train,y_train)
print(model.score(x_test, y_test))
print(model.feature_importances_)
