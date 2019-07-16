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
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesClassifier


def show_images(image_set, compare_func = None):
    window_name = 'images'
    cv2.namedWindow(window_name)
    if compare_func is None:
        on_change = lambda x: cv2.imshow(window_name, image_set[x])
    else:
        on_change = lambda x: cv2.imshow(window_name,
                                         np.hstack([image_set[x],
                                        compare_func(copy(image_set[x]))]))
    cv2.createTrackbar(window_name, window_name, 0, len(image_set) - 1, on_change)
    on_change(0)

def print_feature_importances(generator, importances):
    feature_names = generator.get_feature_positions()
    for name, idx in feature_names.items():
        fs = importances[idx]
        print(name)
        print('mean', np.round(fs.mean(), 4))
        print('max', np.round(fs.max(), 4))
        print('sum', np.round(fs.sum(), 4),'\n')

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


generator = FeatureGenerator(classes = Constants.test_classes, denoise = False, crop = True,
                             remove_borders= True)

all_features = []
all_labels = []
all_images = []
batch_size = 15
total = generator.num_images
for _ in range(int(total/batch_size)):
    print(_*batch_size/total, '%')
    features, labels, images = generator.get_features(batch_size)
    all_features.extend(features)
    all_labels.extend(labels)
    all_images.extend(images)

x = np.vstack(all_features)
y = np.vstack(all_labels)

tree = ExtraTreesClassifier(n_estimators = 100)
evaluate(x,y,tree,generator, True)
classwise_importances(x, y, generator)
show_images(all_images)