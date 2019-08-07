# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:08:55 2019

@author: Andrew Wentzel
"""

from keras.applications import VGG16, imagenet_utils
import cv2
import numpy as np

input_size = (244, 244, 3)
deep_net = VGG16(weights='imagenet', include_top=False, input_shape = input_size)

def deep_features(image):
    image = resize_and_crop(image, input_size[0:2])
    image = imagenet_utils.preprocess_input(image,mode='tf')
    features = deep_net.predict(image)
    return features.ravel()

def resize_and_crop(image, target_shape):
#    fx = target_shape[1]/image.shape[1]
#    fy = target_shape[0]/image.shape[0]
#    scale = max([fx, fy])
#    image = cv2.resize(image, None, fx = scale, fy = scale)
#    height_offset = max([int((image.shape[0] - target_shape[0])//2), 0])
#    width_offset = max([int((image.shape[1] - target_shape[1])//2), 0])
#    image = image[height_offset: target_shape[0] + height_offset,
#                  width_offset: target_shape[1] + width_offset, :]
    image = cv2.resize(image, target_shape)
    return np.expand_dims(image, axis = 0)
