# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:07:21 2019

@author: Andrew
"""

from images import *
from constants import Constants
from features import *

    
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE, RFECV


generator = FeatureGenerator(classes = Constants.test_classes, denoise = False, crop = True, 
                             remove_borders= False)
all_features = []
all_labels = []
all_images = []
batch_size = 15
total = np.sum(generator.inverse_class_positions)
for _ in range(int(total/batch_size)):
    features, labels = generator.get_features(batch_size) 
    all_features.extend(features)
#    images, labels = generator.get_images(batch_size)
#    all_images.extend(images)
    all_labels.extend(labels)
x = np.vstack(all_features)
y = np.vstack(all_labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y)

tree = ExtraTreesClassifier(n_estimators = 25).fit(x_train, y_train.ravel())
print(tree.feature_importances_)
print(tree.score(x_test, y_test.ravel()))