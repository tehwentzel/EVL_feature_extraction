# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:39:55 2019

@author: Andrew Wentzel
"""
import numpy as np
import cv2
cimport numpy as np
cimport cython

ctypedef fused np_type:
    int
    double
    float
    char

ctypedef fused floating:
    double
    float

orgb_operator = np.array([[.114, .587, .299],
                      [-1, .5, .5],
                      [0, -.866, 866]], dtype = np.float32)

opp_operator = np.array([[ 0, -0.70710678,  0.70710678],
       [-0.81649658,  0.40824829,  0.40824829],
       [ 0.57735027,  0.57735027,  0.57735027]])

def opp_transform(np.ndarray pixel):
    return np.matmul(opp_operator, pixel)

def orgb_transform(np.ndarray pixel):
    pixel = np.matmul(orgb_operator, pixel)
    cdef float theta = np.arctan2(pixel[1], pixel[2])
    if theta < np.pi/3:
        theta = theta/2
    else:
        theta = np.pi/4 - theta/4
    return  rot_x(theta, pixel)

def rot_x(float theta, target):
    #applies a rotation matrix around the x axis
    #used for the roation in the oRGb color transform
    rotation = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ], dtype = np.float32)
    return np.matmul(rotation, target)

def bgr_to_orgb(np.ndarray image):
    image = image.astype('float32')/255.0
    orgb_image = np.apply_along_axis(orgb_transform, 2, image)
    scale = lambda x: (x-x.min())/(x.max() - x.min()) if x.min() < x.max() else x - x.min()
    for depth in range(3):
        orgb_image[:,:,depth] = scale(orgb_image[:,:,depth])
    return orgb_image

def bgr_to_opponent(np.ndarray image):
    if image.max() > 1:
        image = image.astype('float32')/255
    opp = np.apply_along_axis(opp_transform, 2, image)
    scale = lambda x: (x-x.min())/(x.max() - x.min()) if x.min() < x.max() else x - x.min()
    for depth in range(3):
        opp[:,:,depth] = scale(opp[:,:,depth])
    return opp

def get_hog(np.ndarray img, int n_bins = 50):
    if img.ndim >2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx,gy)
    magnitude = magnitude.ravel()
    angle = np.nan_to_num(angle % np.pi).ravel()
    histogram = np.zeros((n_bins,))
    cdef float bin_width = np.pi/n_bins
    for idx in range(len(magnitude)):
        histogram[ int(angle[idx]//bin_width) ] += magnitude[idx]
    if np.linalg.norm(histogram) <= 0:
        return np.zeros(histogram.shape)
    return histogram/np.linalg.norm(histogram)

def uniform_space_centers(float min_val, float max_val, int partitions):
    centers = np.zeros((partitions**3, 3), dtype = np.float32)
    cdef float[:,::1] centerview = centers
    cdef float bin_width = (max_val - min_val)/partitions
    cdef float x = bin_width / 2
    cdef int loc = 0
    while x < max_val:
        y = bin_width / 2
        while y < max_val:
            z = bin_width / 2
            while z < max_val:
                centerview[loc,0] = x
                centerview[loc, 1] = y
                centerview[loc, 2] = z
                z = z + bin_width
                loc = loc + 1

            y = y + bin_width
        x = x + bin_width
    return centers