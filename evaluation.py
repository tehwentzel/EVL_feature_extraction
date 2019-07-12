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


from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, RFECV


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


x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y)

tree = ExtraTreesClassifier(n_estimators = 25).fit(x_train, y_train.ravel())
print(tree.score(x_test, y_test.ravel()))
plt.bar(np.arange(len(tree.feature_importances_)), tree.feature_importances_)
show_images(all_images)
