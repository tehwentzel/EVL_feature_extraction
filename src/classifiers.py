# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:14:43 2019

@author: Andrew Wentzel
"""
import numpy as np
import re
from constants import Constants
from sklearn.model_selection import cross_validate, cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, make_scorer
import copy

class NodeClassifier():

    def __init__(self, base_classifier, node):
        self.parent_node = node
        if node is None:
            self.classes = Constants.top_level_classes
        else:
            self.classes = Constants.class_hierarchy[node]
        self.n_classes = len(self.classes)
        self.classifier = copy.copy(base_classifier)
        self.is_fit = False

    def get_xy(self, features, files):
        args, labels = get_class_args(files, self.parent_node)
        if len(args) < 1:
            return None, None, None
        return features[args], labels, args

    def fit(self, features, files):
#        print('fitting', self.parent_node)
        x, y, args = self.get_xy(features, files)
#        print('files found')
        if x is None:
#            print('skipping', self.parent_node)
            return
        if len(set(y)) == 1:
#            print('only one class for', self.parent_node)
            self.constant_class = y[0]
            self.is_fit = True
            return
        self.select_features(x,y)
        x = x[:, self.features_to_use]
#        print('features selected', x.shape[1]/features.shape[1])
        self.classifier.fit(x,y)
#        print('fitted')
        if hasattr(self.classifier, 'feature_importances_'):
            self.feature_importances_[self.features_to_use] = self.classifier.feature_importances_
        self.is_fit = True
        self.constant_class = False
        return args

    def predict(self, features):
        assert(self.is_fit)
        if self.constant_class is not False:
            return np.full((features.shape[0],), self.constant_class)
        x = features[:, self.features_to_use]
        return self.classifier.predict(x)

    def predict_proba(self, features):
        assert(self.is_fit)
        x = features[:,self.features_to_use]
        return self.classifier.predict_proba(x)

    def cross_validate(self, features, files, cv = 5):
        #scores, but only on valide one (so we can get accuracies at each level
        if self.is_fit is False:
            return None
        x, y, args = self.get_xy(features, files)
        scoring = {}
        scoring['accuracy'] = accuracy_score
        scoring['balanced_accuracy'] = balanced_accuracy_score
        scoring['f1_micro'] = lambda y1, y2: f1_score(y1, y2, average = 'micro')
        scoring = {key: make_scorer(val) for key,val in scoring.items()}
        results = cross_validate(self.classifier, x, y, cv = cv, scoring = scoring)
        return results['test_score']


    def select_features(self, x, y):
        args = np.arange(x.shape[1])
        self.feature_importances_ = np.zeros((x.shape[1],))
        feature_scores = mutual_info_classif(x,y)
        args = np.argwhere(feature_scores > 0).ravel()
#        best_score = 0
#        for percentile in np.linspace(50, 100, 6):
#            selector = SelectPercentile(mutual_info_classif, percentile)
#            x_subset = selector.fit_transform(x,y)
#            score = cross_val_score(self.classifier, x_subset, y, cv = 3).mean()
#            if score > best_score:
#                best_score = score
#                args = selector.get_support(True)
#                self.feature_percentile = percentile
        self.features_to_use = args
        self.feature_importances_ = feature_scores

class CascadeClassifier():

    def __init__(self, base_estimator):
        nodes = Constants.class_hierarchy.keys()
        self.classifiers = {node: NodeClassifier(base_estimator, node) for node in nodes}
        self.classifiers['Top'] = NodeClassifier(base_estimator, None)

    def fit(self, features, files):
        for parent_node, classifier in self.classifiers.items():
            #this is a little unneeded but I always make error with dicts when directly mutating them
            print(parent_node)
            classifier.fit(features, files)
            self.classifiers[parent_node] = classifier

    def predict(self, features):
        y_pred = self.classifiers['Top'].predict(features)
        y_set = set(y_pred)
        while True:
            parents = [c for c in y_set if c in Constants.class_hierarchy.keys()]
            parents = [p for p in parents if self.classifiers[p].is_fit]
            if len(parents) == 0:
                break
            to_classify = np.argwhere(y_pred == parents[0]).ravel()
            classifier = self.classifiers[parents[0]]
            y_pred[to_classify] = classifier.predict(features[to_classify])
            y_set = set(y_pred)
        return y_pred

#
#skf = StratifiedKFold(n_splits = 4)
#y = classes_from_files(files)
#skf.get_n_splits(features, y)
#accuracys =[]
#balanced_accuracys = []
#for train_ind, test_ind in skf.split(features, y):
#    x_train, files_train = features[train_ind], np.array(files)[train_ind]
#    x_test, files_test = features[test_ind], np.array(files)[test_ind]
#    y_test = classes_from_files(files_test)
#    cc.fit(x_train, files_train)
#    y_pred = cc.predict(x_test)
#    accuracys.append( accuracy_score(y_test, y_pred))
#    balanced_accuracys.append( balanced_accuracy_score(y_test, y_pred))

    def cv_score(self, features, files, n_splits = 3):
        skf = StratifiedKFold(n_splits = n_splits)
        y = classes_from_files(files)
        skf.get_n_splits(features, y)
        accuracys = {c: 0 for c in self.classifiers.keys()}
        balanced_accuracys = {c: 0 for c in self.classifiers.keys()}
        overall_accuracys = []
        overall_balanced = []
        concat = lambda x,y,c: x[c] + y[c]
        for train_ind, test_ind in skf.split(features, y):
            x_train, files_train = features[train_ind], np.array(files)[train_ind]
            x_test, files_test = features[test_ind], np.array(files)[test_ind]
            x_train, x_test = normalize(x_train, x_test)
            self.fit(x_train, files_train)
            y_pred = self.predict(x_test)
            y_test = classes_from_files(files_test)
            accuracy, balanced_accuracy = self.score(x_test, files_test)
            for classifier in self.classifiers.keys():
                accuracys[classifier] = concat(accuracys, accuracy, classifier)
                balanced_accuracys[classifier] = concat(balanced_accuracys, balanced_accuracy, classifier)
            overall_accuracys.append(accuracy_score(y_test, y_pred))
            overall_balanced.append( balanced_accuracy_score(y_test, y_pred))
        dict_divide = lambda d: {k: v/n_splits for k, v in d.items()}
        accuracys = dict_divide(accuracys)
        balanced_accuracys = dict_divide(balanced_accuracys)
        accuracys['All'] = np.mean(overall_accuracys)
        balanced_accuracys['All'] = np.mean(overall_balanced)
        return accuracys, balanced_accuracys

    def score(self, features, files):
        accuracys = {c: 0 for c in self.classifiers.keys()}
        balanced_accuracys = {c: 0 for c in self.classifiers.keys()}
        for cname, classifier in self.classifiers.items():
            valid_args, _ = get_class_args(files, classifier.parent_node)
            if len(valid_args) <= 1:
                continue
            x = features[valid_args]
            valid_files = files[valid_args]
            y = classes_from_files(valid_files, stop_nodes = classifier.classes)
            y_pred = classifier.predict(x)
            accuracys[cname] = accuracy_score(y, y_pred)
            balanced_accuracys[cname] = balanced_accuracy_score(y, y_pred)
#        print(accuracys)
        return accuracys, balanced_accuracys


def normalize(x1, x2 = None):
    reg_coeffs = np.nan_to_num(x1.mean(axis = 0)/x1.std(axis = 0))
    regularize = lambda v: v*reg_coeffs
    if x2 is not None:
        x2 = regularize(x2)
    return regularize(x1), regularize(x2)

def classes_from_files(files, parent = None,  stop_nodes = None, depth = 100):
    all_args, labels = get_class_args(files, parent)
    good_files = list(np.array(files)[all_args])
    parent_nodes = set(Constants.class_hierarchy.keys())
    if stop_nodes is not None:
        parent_nodes = parent_nodes - set(stop_nodes)
    for d in range(depth):
        labelset = set(labels)
        parents = [label for label in labelset if (label in parent_nodes)]
        if len(parents) == 0:
            break
        args, new_labels = get_class_args(good_files, parents[0])
        labels[args] = new_labels
    return labels

def get_class_args(files, parent = None):
    if parent is None:
        classes = Constants.top_level_classes
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