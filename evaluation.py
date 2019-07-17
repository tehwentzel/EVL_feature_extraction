# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:07:21 2019

@author: Andrew
"""

from images import *
from constants import Constants
from features import *
import cv2
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier


def show_images(image_set, compare_func = None):
    window_name = 'images'
    cv2.namedWindow(window_name)
    max_dim = max(image_set[0].shape)
    clahe = cv2.createCLAHE(clipLimit = 100, tileGridSize = (100,100))
    if image_set[0].ndim == 3:
        dummy_image = np.ones((max_dim, max_dim, 3)).astype('uint16')
    else:
        dummy_image = np.ones((max_dim, max_dim)).astype('uint16')
    process = lambda x: x
    if compare_func is not None:
        dummy_image = np.hstack([dummy_image, dummy_image])
        process = lambda x: clahe.apply(np.hstack([x, compare_func(copy(x))]).astype('uint16'))
    def on_change(n):
        nonlocal dummy_image
        i = image_set[n]
        cv2.imshow(window_name, dummy_image)
        cv2.imshow(window_name, process(i))
        dummy_image = np.ones(i.shape).astype('uint16')
    cv2.createTrackbar(window_name, window_name, 0, len(image_set) - 1, on_change)
    on_change(0)

def print_feature_importances(generator, importances, plot = True):
    feature_names = generator.get_feature_positions()
    labels = []
    sums = []
    for name, idx in feature_names.items():
        fs = importances[idx]
        print(name)
        print('mean', np.round(fs.mean(), 4))
        print('max', np.round(fs.max(), 4))
        print('sum', np.round(fs.sum(), 4),'\n')
        labels.append(name)
        sums.append(fs.sum())
    if plot:
        ranks = sorted(zip(labels, sums), key = lambda x: x[1])
        labels, sums = list(zip(*ranks))
        x = np.arange(len(sums))
        plt.barh(x,sums, tick_label = labels)
        plt.show()

def classwise_importances(x, y, generator):
    model = ExtraTreesClassifier(n_estimators = 100)
    feature_names = generator.get_feature_positions()
    for c in range(len(generator.class_names)):
        name = generator.class_names[c]
        binary_y = (y == c).astype('int32')
        xtrain, xtest, ytrain, ytest = train_test_split(x, binary_y, stratify = binary_y)
        model.fit(xtrain, ytrain.ravel())
        print(name, model.score(xtest, ytest.ravel()))
        for fname, idx in feature_names.items():
            print(fname, np.round(model.feature_importances_[idx].sum(), 4))
        print()

def evaluate(x, y, model, generator, importances = False):
    model.fit(x, y.ravel())
    if importances:
        print_feature_importances(generator, model.feature_importances_)
        plt.bar(np.arange(len(model.feature_importances_)), model.feature_importances_)
    metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    score = cross_validate(model, x, y.ravel(), cv = 5, scoring=metrics)
    for metric in metrics:
        key = 'test_' + metric
        print(metric, score[key].mean())
    print()
    return score

def get_loaded_images(file_path = 'data\cleaned_images.pickle'):
    with open(file_path, 'rb') as f:
        images = pickle.load(f)
    return images

def get_featureset(generator, total = None, batch_size = 20):
    all_features = []
    all_labels = []
    all_images = []
    if total is None:
        total = generator.num_images
    for _ in range(int(total/batch_size)):
        print(_*batch_size/total, '%')
        features, labels, images = generator.get_features(batch_size)
        all_features.extend(features)
        all_labels.extend(labels)
        all_images.extend(images)
    return np.vstack(all_features), all_labels, all_images


#files = []
#for fs in generator.file_dict.values():
#    files.extend(fs)
#
#parent = 'Microscopy'
#parent_re = re.compile(parent)
#children_classes = Constants.class_hierarchy[parent]
#class_res = [re.compile[c] for c in children_classes]
#labels = []
#for file in files:
#    if parent_re.search(file) is not None:
#        for c_idx in range(len(children_classes)):
#            if class_res[c_idx].search(file) is not None:
#                labels.append(c_idx)
#
def get_classes(x, files, parent):
    parent_pattern = re.compile(parent)
    classes = Constants.class_heirarchy[parent]
    class_patterns = [re.compile(c) for c in classes]
    y = -np.ones((len(files),))
    for idx in range(len(files)):
        file = files[idx]
        if parent_pattern.search(file) is None:
            continue
        for c in range(len(classes)):
            if class_patterns[c].search(file) is not None:
                y[idx] = c
                break
    good_files = np.argwhere(y > -1).ravel()
    return x[good_files], y[good_files]



generator = FeatureGenerator(classes = Constants.test_classes, denoise = False, crop = True,
                             remove_borders= True)


features, files, images = get_featureset(generator, total = 120)
#
#tree = ExtraTreesClassifier(n_estimators = 100)
#evaluate(x,y,tree,generator, True)
#classwise_importances(x, y, generator)
#show_images(all_images)