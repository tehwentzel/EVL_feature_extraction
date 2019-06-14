# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:07:21 2019

@author: Andrew
"""

from images import *
from constants import Constants
from features import *

    
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
    
def test_classifier(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify = y)
    #pca = PCA(n_components = None)
    #x_train = pca.fit_transform(x_train)
    #x_test = pca.transform(x_test)
    
    model = RandomForestClassifier(n_estimators = 60,
                                   max_depth = 25,
                                   min_samples_split = 2,
                                   random_state = 0)
    model.fit(x_train,y_train)
    result = model.score(x_test, y_test)
    print(result)
    return (result, model.feature_importances_)

#sample_images = get_image_files(root = 'data/test*/**/*.jpg', classes = Constants.test_classes)
#clean_images = preprocess_dict(sample_images)
#show_images(clean_images)
all_images = get_image_files(classes = Constants.test_classes)
all_images = preprocess_dict(all_images)
y = get_classes(all_images)
x = get_color_histograms(all_images, bins = 5)
result, importances = test_classifier(x,y)