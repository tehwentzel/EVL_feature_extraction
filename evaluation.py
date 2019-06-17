# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:07:21 2019

@author: Andrew
"""

from images import *
from constants import Constants
from features import *

    
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
    
def test_classifier(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y)
#    model = AdaBoostClassifier(n_estimators = 200,
#                               learning_rate = .1)
    model = RandomForestClassifier(n_estimators = 100,
                                   max_depth = x.shape[0]//3,
                                   min_samples_split = 2,
                                   random_state = 0)
    model.fit(x_train,y_train)
    result = model.score(x_test, y_test)
    return (result, model.feature_importances_)

#sample_images = get_image_files(root = 'data/test*/**/*.jpg', classes = Constants.test_classes)
#clean_images = preprocess_dict(sample_images)
#show_images(clean_images)
#all_images = get_image_files()
#for key, value in all_images.items():
#    all_images[key] = value[0:min([400, len(value)])]
#y = get_classes(all_images)
#all_images = preprocess_dict(all_images)
    
x = get_features(all_images, [get_hog_descriptors])
result, importances = test_classifier(x,y)
print(result, importances)