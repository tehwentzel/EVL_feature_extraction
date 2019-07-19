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
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
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

def classwise_importances(x, y, generator, parent = None):
    model = ExtraTreesClassifier(n_estimators = 100)
    feature_names = generator.get_feature_positions()
    if parent is None:
        for parent, child_classes in Constants.class_hierarchy.items():
            for item in set(y):
                if item in child_classes:
                    class_names = child_classes
                    break
    for c in range(len(class_names)):
        name = class_names[c]
        if name not in set(y):
            continue
        binary_y = (y == name).astype('int32')
        xtrain, xtest, ytrain, ytest = train_test_split(x, binary_y, stratify = binary_y)
        model.fit(xtrain, ytrain.ravel())
        print(name)#, model.score(xtest, ytest.ravel()))
        for fname, idx in feature_names.items():
            print(fname, np.round(model.feature_importances_[idx].sum(), 4))
        print()

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


def get_class_args(files, parent = None):
    if parent is None:
        classes = list( Constants.class_hierarchy.keys() )
        parent_pattern = re.compile('')
    else:
        classes = Constants.class_hierarchy[parent]
        parent_pattern = re.compile(parent)
    class_patterns = [re.compile(c) for c in classes]
    y = []
    good_files = []
    for idx in range(len(files)):
        file = files[idx]
        if parent_pattern.search(file) is None:
            continue
        for c in range(len(classes)):
            if class_patterns[c].search(file) is not None:
                y.append(classes[c])
                good_files.append(idx)
                break
    return np.array(good_files), np.array(y)

def crossvalidate(x,y,model):
    kfold = StratifiedKFold(n_splits = 5)
    y_pred = np.empty(y.shape).astype('str')
    importances = []
    for train_index, test_index in kfold.split(x,y):
       model.fit(x[train_index], y[train_index])
       y_pred[test_index] = model.predict(x[test_index])
       importances.append(model.feature_importances_)
    return y_pred, np.mean(importances, axis = 0)


def save_result(ytrue, ypred, x_importances, feature_names):
    scores = {}
    scores['accuracy'] = accuracy_score(ytrue, ypred)
    scores['f1_weighted'] = f1_score(ytrue, ypred, average = 'weighted')
    importances = {}
    for name, pos in feature_names.items():
        importances[name] = x_importances[pos].sum()
    scores['importances'] = importances
    return scores

def get_cascade_classifier_results(features, files, generator, top_level = None, model = None):
    feature_names = generator.get_feature_positions()
    all_args, y_true = get_class_args(files, parent = top_level)
    if len(all_args) == 0:
        return None
    good_files = list(np.array(files)[all_args])
    good_features = features[all_args]

    if model is None:
        model = ExtraTreesClassifier(n_estimators = 100)
    y_pred, importances = crossvalidate(good_features, y_true, model)
    staged_results = {str(set(y_true)): save_result(y_true, y_pred, importances, feature_names)}
    while True:
        yset = set(y_true)
        parents = [label for label in yset if (label in Constants.class_hierarchy.keys())]
        if len(parents) == 0:
            break
        args, y = get_class_args(good_files, parents[0])
        y_true[args] = y

        to_classify = np.argwhere(y_pred == parents[0]).ravel()
        if len(to_classify) == 0:
            continue
        y_pred, importances = crossvalidate(good_features, y_true, model)
        staged_results[str(set(y_true))] = save_result(y_true, y_pred, importances, feature_names)
    return staged_results


generator = FeatureGenerator(denoise = False, crop = True,
                             remove_borders= True, class_roots = 'Experimental')
features, files, images = get_featureset(generator, total = 800)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
gbc = ExtraTreesClassifier(n_estimators = 200)
accuracys = {'Top': get_cascade_classifier_results(features, files, generator, None, gbc)}
for level in Constants.class_hierarchy.keys():
    results = get_cascade_classifier_results(features, files, generator, level, gbc)
    if results is not None:
        accuracys[level] = results
print(accuracys['Experimental'])