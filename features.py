# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:59:27 2019

@author: Andrew
"""
import cv2
import numpy as np
from copy import copy
from skimage.feature import local_binary_pattern
from skimage.filters import gabor, meijering
from constants import Constants
from images import *
from skimage.transform import radon
import texture

class FeatureGenerator(ImageGenerator):

    def __init__(self, root = 'data\images_2\**\*.jpg', classes = None,
                 crop = True, scale = Constants.image_size,
                 remove_borders = True, denoise = True):
        super().__init__(root, classes, crop, scale, remove_borders, denoise)
        self.color_features = {
                'color_histograms': get_color_histogram,
#                'sobel_histograms': sobel_hist
                }
        self.gray_features = {
                'linear binary patterns': lbp,
                'HOG': multiscale_hog,
                'gray histogram': multiscale_histogram,
                'gabor sums': gabor_sums,
                'radon_histograms': radon_hists,
                'Chebyshev histograms': chebyshev2d,
                'Meijering sum': meijering_sum,
                'Hu Moments': lambda x: cv2.HuMoments(cv2.moments(x)).ravel(),
#                'Tamura Texture': tamura_features
                }
        self.fft_features = {
                'fft multiscale_histograms': multiscale_histogram,
                'fft radon_histogram': radon_hists,
                'fft chebyshev histogram': chebyshev2d
                }


    def get_features(self, num_images, classes = None):
        images, labels = self.get_images(num_images, classes)
        x = []
        for image in images:
            img_features = self.image_features(image)
            x.append(img_features)
        x = np.vstack(x)
        return x, labels, images

    def get_feature_positions(self):
        p = 0
        dummy_image = 10*np.random.random((100, 100, 3)).astype('float32')
        dummy_gray_image = dummy_image.mean(axis = 0)
        feature_inds = {}
        def add_names(im, fdict):
            nonlocal p
            for name, f in fdict.items():
                feature = f(im)
                frange = np.arange(p, p + len(feature))
                p = p + len(feature)
                feature_inds[name] = frange
        add_names(dummy_image, self.color_features)
        add_names(dummy_gray_image, self.gray_features)
        add_names(dummy_gray_image, self.fft_features)
        return feature_inds


    def image_features(self, image):
        features = []
        def add_features(i, funcs):
            for func in funcs.values():
                featureset = func(i).ravel()
                features.append(featureset)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fft_image = fft_magnitude(gray_img)
        add_features(image, self.color_features)
        add_features(gray_img, self.gray_features)
        add_features(fft_image, self.fft_features)
        return np.hstack(features).ravel()

def get_color_histogram(image, bins = 15,
                        additional_color_spaces = [
                             cv2.COLOR_BGR2YCrCb,
                             cv2.COLOR_BGR2HSV]):
    features = []
    features.append( get_colorspace_histogram(image, bins = bins) )
    for color_space in additional_color_spaces:
        converted_image = cv2.cvtColor(copy(image), color_space)
        features.append( get_colorspace_histogram(converted_image, bins = bins) )
    features = np.hstack(features)
    return features

def get_colorspace_histogram(image, bins = 15):
    if len(image.shape) > 2: #check if grayscale
        n_channels = image.shape[2]
    else:
        n_channels = 1
    hists = []
    for channel in range(n_channels):
        if n_channels > 1:
            img_slice = image[:,:,channel]
        else:
            img_slice = image
        color_histogram = multiscale_histogram(img_slice)
        hists.append(color_histogram.astype('float32'))
    return np.hstack(hists)

def tamura_features(image, kmax = 2, bins = 4):
    coarseness_matrix = texture.coarseness(image, kmax)
    #directionality is nan in some cases
#    directionality = texture.directionality(image)
    contrast = texture.contrast(image)
    sums = np.array([coarseness_matrix.mean(),  contrast])
    c_hist = np.histogram(coarseness_matrix.ravel(), bins = bins, range = (0,20))[0]
    return np.hstack([sums, c_hist])

def radon_hists(image, bins = 10):
    radon_image = radon(image,
          theta = [0, 45, 90, 135, 180],
          circle = False)
    return im2hist(radon_image, bins)

def lbp(img, n_bits = 4, radius = 2, n_scales = 2, max_bins = 15):
    lbp_fun = lambda i: local_binary_pattern(i, n_bits, radius, method = 'ror')
    bins = [min([max_bins, 2**n_bits - 1]) for dummy in range(n_scales)]
    return multiscale_histogram(img, transform = lbp_fun, bins = bins, range_ = (0, 2**n_bits))

def im2hist(img, bins, range_ = (0,255)):
    return np.histogram(img, bins = bins, density = True, range = range_)[0].astype('float32')

def gabor_sums(img, bins = 5):
    #gabor based features based on wndcharm
    vector = np.empty((bins,))
    integrate = lambda i: np.sum(i)/np.ones(i.shape).sum()
    gab = lambda f: integrate(gabor(img, f, theta = np.pi/2)[0])
    lower_bound = 0
    f = 0
    while lower_bound <= 0:
        f += .1
        lower_bound = gab(f + .1)
    for f in np.arange(1, bins + 1):
        vector[f-1] = gab(f)/lower_bound
    return vector.astype('float32')

def meijering_sum(img, n_scales = 2):
    mintegrate = lambda i, ridges: binary_integral( meijering(i, black_ridges = ridges))
    output = np.empty((2*n_scales,)).astype('float32')
    output[0] = mintegrate(img, True)
    output[1] = mintegrate(img, False)
    for scale in range(n_scales - 1):
        position = 2*(scale + 1)
        img = cv2.pyrDown(img)
        output[position] = mintegrate(img, True)
        output[position + 1] = mintegrate(img, False)
    return output

def fft_magnitude(img, scale = False):
    fft_img = np.fft.fft2(img)
    fft_img = np.conj(fft_img)*fft_img
    fft_img = np.real(fft_img)
    if scale:
        fft_img = img.max()*(fft_img - fft_img.min())/(fft_img.max() - fft_img.min())
    return fft_img


def sobel_hist(img, bins = 20):
    sobel = lambda x,y: (lambda i: cv2.Sobel(i, cv2.CV_32F, x, y))
    gx = multiscale_histogram(img, transform = sobel(1, 0), range_ = (-900,900) )
    gy = multiscale_histogram(img, transform = sobel(0, 1), range_ = (-900, 900) )
    hist = np.hstack([gx, gy])
    return hist

def multiscale_hog(img, n_scales = 3, n_bins = 50):
    bincounts = [int(n_bins/(i+1)) for i in range(n_scales)]
    hist = []
    for bincount in bincounts:
        if bincount != bincounts[0]:
            img = cv2.pyrDown(img)
        scale_hist = get_hog(img, bincount)
        hist.append(scale_hist)
    return np.hstack(hist)

def get_hog(img, n_bins = 50):
    if len(img.shape) > 2:
        img = cv2.cvtColor(copy(img), cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    magnitude, angle = cv2.cartToPolar(gx,gy)
    magnitude = magnitude.ravel()
    angle = np.nan_to_num(angle % np.pi).ravel()
    histogram = np.zeros((n_bins,))
    bin_width = np.pi/n_bins
    for idx in range(len(magnitude)):
        histogram[ int(angle[idx]//bin_width) ] += magnitude[idx]
    if np.linalg.norm(histogram) <= 0:
        return np.zeros(histogram.shape)
    return histogram/np.linalg.norm(histogram)

def binary_integral(gray_image):
    assert(len(gray_image.shape) == 2)
    bw_image = cv2.threshold(gray_image.astype('uint8'), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return np.sum(bw_image)/np.ones(bw_image.shape).sum()

def multiscale_histogram(img, bins = [7,5,3],
                         initial_size = None,
                         transform = None,
                         range_ = (0,255)):
    #computes a concatenation of histogram at progressively lower resolutions
    if initial_size is not None:
        img = cv2.resize(img, initial_size)
    hist = []
    for scale in range(len(bins)):
        if scale > 0:
            img = cv2.pyrDown(img)
        if transform is not None:
            img = transform(img)
        n_bins = bins[scale]
        hist.append( im2hist(img, n_bins, range_ = range_) )
    hist = np.hstack(hist)
    return hist

def chebyshev2d(image, degree = 20, bins = 10):
    assert(len(image.shape) == 2)
    def cheb1d(im):
        x = np.arange(im.shape[1])
        return np.polynomial.chebyshev.chebfit(x, im.T , degree)
    try:
        x_transform = cheb1d(image)
        y_transform = cheb1d(x_transform)
        hist = np.histogram(y_transform.ravel(), bins = bins, density = True, range = (0,.3))[0]
    except:
        hist = np.arange(0, bins)
    return hist


get_classes = lambda file_dict: np.hstack([k*np.ones((len(v), )) for k,v in file_dict.items()]).astype('int32')
