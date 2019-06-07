import cv2
import numpy as np
import glob
from copy import copy
import re
import matplotlib.pyplot as plt

classes = ['Electron-Mix','Transmission',
                               'ReporterANDImmunoHistochem', 'Fluorescence-Mix',
                               'Light-Mix', 'InSituHybridization']

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

def get_histogram(file, bins = 5, n_channels = 3, color_space = cv2.COLOR_BGR2YCrCb):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, color_space)
    hists = []
    scale = lambda x: (x - x.min())/(x.max() - x.min())
    for channel in range(n_channels):
#        image[:,:,channel] = scale(image[:,:,channel])
        color_histogram = np.histogram(image[:,:,channel], bins = bins, density = True)[0]
        hists.append(color_histogram.astype('float32'))
    return(hists)

def get_correlations(file_dict = None, method = cv2.HISTCMP_CHISQR, color_space = cv2.COLOR_BGR2YCrCb):
    if file_dict is None:
        file_dict = get_image_files()
    hist_distance_dict = {}
    for c in file_dict.keys():
        image_files  = file_dict[c]
        correlation = np.zeros((len(image_files),3))
        position = 0
        for image_file in image_files:
            y,r,g = get_histogram(image_file, color_space = color_space)
            correlation[position, 0] = cv2.compareHist(r,g, method)
            correlation[position, 1] = cv2.compareHist(y,r, method)
            correlation[position, 2] = cv2.compareHist(y,g, method)
            position += 1
        hist_distance_dict[c] = correlation
    return hist_distance_dict

d = get_correlations(file_dict = file_dict, 
#                     color_space = cv2.COLOR_BGR2RGB, 
                     method = cv2.HISTCMP_BHATTACHARYYA)
for c, v in d.items():
    print(classes[c], v.mean(axis = 0))