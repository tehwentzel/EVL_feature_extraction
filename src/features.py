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
from scipy.fftpack import rfft
from constants import Constants
import pickle
from images import *
from sklearn.decomposition import PCA
import cv_functions
from skimage.transform import radon
import texture
from skfuzzy import cmeans, cmeans_predict
from bag_of_words import *
from dl_features import deep_features

class FeatureGenerator(ImageGenerator):

    def __init__(self, root = 'data\images\**\*.jpg', class_roots = None,
                 crop = True, scale = Constants.image_size,
                 remove_borders = True, denoise = True, bovw_codebook = None):
        #inherits from ImageGnerator in IMages.py which reads and preprocesses inages
        super().__init__(root, class_roots, crop, scale, remove_borders, denoise)

        #precomputed codebook of visual words to use with sift_words
        if bovw_codebook is not None:
            file_path = 'data\\' + bovw_codebook + '.pickle'
            with open(file_path, 'rb') as f:
                self.codebook = pickle.load(f)
        else:
            self.codebook = None

        #various featurees and their names (for printing later) to extract from an image
        self.color_features = {
                'color_histograms': get_color_histogram,
                'fuzzy_color_histogram': fuzzy_color_histograms,
                'VGG16_features': deep_features,
                'SIFT BOW': lambda x: sift_words(x, self.codebook, dense = True),
#                'sobel_histograms': sobel_hist
                }
        self.gray_features = {
                'linear binary patterns': lbp,
                'HOG': multiscale_hog,
                'grayscale_histogram': multiscale_histogram,
#                'gabor sums': gabor_sums,
#                'radon_histograms': radon_hists,
#                'Chebyshev histograms': chebyshev2d,
#                'Meijering sum': meijering_sum,
#                'Hu Moments': lambda x: cv2.HuMoments(cv2.moments(x)).ravel(),
                'fft of Edges': fft_edges,
#                'Tamura Texture': tamura_features
                }
        self.fft_features = {
                'fft multiscale_histograms':lambda x: multiscale_histogram(x, range_ = (0, 800)),
#                'fft radon_histogram': radon_hists,
#                'fft chebyshev histogram': chebyshev2d
                }

        #dictionary of feature names of positions from get_features
        self.f_dict = self.get_feature_positions()

    def extract_features(self, images):
        #extracts features for a get of images
        x = []
        for image in images:
            img_features = self.image_features(image)
            x.append(img_features)
        x = np.vstack(x)
        return x

    def get_features(self, num_images, classes = None):
        #retreives images and extracted features for a given number of images
        images, labels = self.get_images(num_images, classes)
        x = self.extract_features(images)
        return x, labels, images

    def get_feature_positions(self):
        #makes a dummy image and calls image_features to figure out
        #the dimensionallity of each feature extracts from self.*_features
        #returns a dictionary of the function names and indexes of the features returned by image features
        p = 0
        dummy_image = 255*np.random.random((244, 244, 3)).astype('float32')
        dummy_gray_image = dummy_image.mean(axis = 0)
        feature_inds = {}
        def add_names(im, fdict):
            nonlocal p
            for name, f in fdict.items():
                feature = f(im)
                if len(feature) > 0:
                    frange = np.arange(p, p + len(feature))
                    p = p + len(feature)
                    feature_inds[name] = frange
        add_names(dummy_image, self.color_features)
        add_names(dummy_gray_image, self.gray_features)
        add_names(dummy_gray_image, self.fft_features)
        return feature_inds

    def image_features(self, image):
        #extracts features from dictionary of feature for  a single images
        #returns a 1d nd array of all the features in order
        features = []
        def add_features(i, funcs):
            for func in funcs.values():
                featureset = func(i)
                if featureset is not None:
                    featureset = np.nan_to_num(featureset)
                    features.append(featureset.ravel())
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fft_image = rfft(gray_img)
        add_features(image, self.color_features)
        add_features(gray_img, self.gray_features)
        add_features(fft_image, self.fft_features)
        return np.hstack(features).ravel()

    def featureset_pca(self, x, variance = .95):
        #does pca on each feature type of an image
        #mutates f_dict accordingly
        feature_names = self.get_feature_positions()
        new_features = []
        new_positions = {}
        pca = PCA()
        def pca_to_variance(x):
            x_pca = pca.fit_transform(x)
            n_features = 1
            while pca.explained_variance_ratio_[0:n_features].sum() < variance:
                n_features += 1
            return x_pca[:, 0:n_features]
        loc = 0
        for name, idxs in feature_names.items():
            x_set = x[:, idxs]
            x_new = pca_to_variance(x_set)
            new_features.append(x_new)
            new_positions[name] = np.arange(loc, loc + x_new.shape[1])
            loc += x_new.shape[1]
        self.f_dict = new_positions
        return np.hstack(new_features)

def get_color_spaces():
    #gets a list of functions that tranform a bgr image into different
    #colorspaces that are scaled to be between 0 and 1
    color_space_tranforms = [
            lambda i: i.astype('float32')/255,
            cv_functions.bgr_to_opponent,
            lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2YCrCb)/255.0,
            lambda i: cv2.cvtColor(i, cv2.COLOR_BGR2HSV)/255.0,
#            cv_functions.bgr_to_orgb,
            ]
    return color_space_tranforms

def get_color_histogram(image, bins = 20):
    #histogram of various colorspaces of an image at multiple resolutions
    features = []
    color_spaces = get_color_spaces()
    for color_space in color_spaces:
        converted_image = color_space(image)
        features.append( get_colorspace_histogram(converted_image, bins = bins) )
    features = np.hstack(features)
    return features

def get_colorspace_histogram(image, bins = 15, range_ = (0,1)):
    #gets a pyramid histogram for each channel in an image
    if image.ndim > 2: #check if grayscale
        n_channels = image.shape[2]
    else:
        n_channels = 1
    hists = []
    for channel in range(n_channels):
        if n_channels > 1:
            img_slice = image[:,:,channel]
        else:
            img_slice = image
        #some color spacs have different bounds
        if img_slice.max() > range_[1] or img_slice.min() < range_[0]:
            print('improper range used in colorspace histogram...')
        color_histogram = multiscale_histogram(img_slice, range_ = range_)
        hists.append(color_histogram.astype('float32'))
    return np.hstack(hists)

def tamura_features(image, kmax = 2, bins = 4):
    #tamura featues stolen from a library that I don't think realy works
    coarseness_matrix = texture.coarseness(image, kmax)
    #directionality is nan in some cases
#    directionality = texture.directionality(image)
    contrast = texture.contrast(image)
    sums = np.array([coarseness_matrix.mean(),  contrast])
    c_hist = np.histogram(coarseness_matrix.ravel(), bins = bins, range = (0,20))[0]
    return np.hstack([sums, c_hist])

def radon_hists(image, bins = 10):
    #histogram of the radon transform of a gryscale image
    radon_image = radon(image,
          theta = [0, 45, 90, 135, 180],
          circle = False)
    return im2hist(radon_image, bins)

def fft_edges(image, bins = 10):
    #fourier transform of a canny filder
    transform = lambda i: rfft(cv2.Canny(i.astype('uint8'),100,200))
    return multiscale_histogram(image, transform = transform, range_ = (0,800))

def lbp(img, n_bits = 6, radius = 2, n_scales = 2, max_bins = 15):
    #hisgram of linear-binary-pattern featues
    img = cv2.GaussianBlur(img, (5,5),0)
    lbp_fun = lambda i: local_binary_pattern(i, n_bits, radius, method = 'ror')
    bins = [2**n_bits - 1 for dummy in range(n_scales)]
    return multiscale_histogram(img, transform = lbp_fun, bins = bins, range_ = (0, 2**n_bits))

def im2hist(img, bins, range_ = (0,255)):
    #simplified function to get a density based histogram from numpy
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
    #computes the integral of he meijering transform (finds tubes and stuff) of an image
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
    #hlerp to get the mangitude of a comple furrier (sp) transform
    fft_img = np.fft.fft2(img)
    fft_img = np.conj(fft_img)*fft_img
    fft_img = np.real(fft_img)
    if scale:
        fft_img = img.max()*(fft_img - fft_img.min())/(fft_img.max() - fft_img.min())
    return np.sqrt(fft_img)


def sobel_hist(img, bins = 20):
    #histogram of edges from  a sobel edge detector
    sobel = lambda x,y: (lambda i: cv2.Sobel(i, cv2.CV_32F, x, y))
    gx = multiscale_histogram(img, transform = sobel(1, 0), range_ = (-900,900) )
    gy = multiscale_histogram(img, transform = sobel(0, 1), range_ = (-900, 900) )
    hist = np.hstack([gx, gy])
    return hist

def multiscale_hog(img, n_scales = 3, n_bins = 50):
    #computes a global histogram of oriented graidents at multiple scales
    bincounts = [int(n_bins/(i+1)) for i in range(n_scales)]
    hist = []
    for bincount in bincounts:
        if bincount != bincounts[0]:
            img = cv2.pyrDown(img)
        scale_hist = cv_functions.get_hog(img, bincount)
        hist.append(scale_hist)
    return np.hstack(hist)

def binary_integral(gray_image):
    #just helper function to binarize a grayscale image and comput the ratio
    #of images above a certain threshold
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
    #chebyshev 2d transform, I think
    #not really sure I did this right
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

def sift_words(image, codebook, dense = True):
    #extracts the visual works from sift features from a pre-copmuteed codebook
    #dense will use sift vectors from every 5-pixel window, rather than just keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    desc = root_descriptor(image.astype('uint8'), sift, dense)
    word_vector = extract_visual_words([desc], codebook).ravel()
    return word_vector/word_vector.sum()

def tfidf_weights(matrix):
    #takes an count ndarray and returns the tfidf wighitng of it
    idf = matrix.shape[0]/np.sum(matrix, axis = 0)
    idf = np.log(idf)
    tf = 1/np.sum(matrix, axis = 1)
    matrix = matrix*tf*idf
    return matrix

def fuzzy_color_histograms(image):
    #computes fuzzy histogram pyramids for multiple color spaces
    color_funcs = get_color_spaces()
    hists = []
    for color_space in color_funcs:
        i = color_space(image)
        hists.append( fuzzy_pyramid_histogram(i).ravel() )
    return np.hstack(hists)

def fuzzy_pyramid_histogram(image, range_ = (0,1), n_bins = [64, 27, 27]):
    #conputes fuzzy histograms for 3d colorspaces at multiple resolutions
    #uses uniformly spaced cluster centers over a cube with dims range[1] - range[0]
    #clusters found using fuzzy kmeans clustering over the images in the pixel
    hists = []
    def scaled_hist(bins, image):
        bin_edge_width = int(bins**(1/3))
        bin_centers = cv_functions.uniform_space_centers(range_[0],
                                                         range_[1],
                                                         bin_edge_width)
        fuzzy_centers = cmeans_predict(image.reshape(-1,3).T,
                                       bin_centers,
                                       2, .005, 1000)[0].T
        hist = fuzzy_centers.sum(axis = 0).ravel()
        return hist/np.sum(hist)
    for i in range(len(n_bins)):
        if i > 0:
            image = cv2.pyrDown(image)
        hist = scaled_hist(n_bins[i], image)
        hists.append(hist)
    return np.hstack(hists).astype('float32')


get_classes = lambda file_dict: np.hstack([k*np.ones((len(v), )) for k,v in file_dict.items()]).astype('int32')
