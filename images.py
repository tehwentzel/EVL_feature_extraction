import cv2
import numpy as np
import glob
from copy import copy
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern

class Constants():
    
    classes = ['Electron-Mix',
               'Transmission',
    #           'ReporterANDImmunoHistochem', 
               'Fluorescence-Mix',
               'Light-Mix', 
    #           'InSituHybridization'
               ]
    
    test_classes = [
            "Electron",
            "Fluorescence",
            "Light"
            ]

def get_image_files(root = 'data\images_2\**\*.jpg',
                    classes = None):
    if classes is None:
        classes = Constants.classes
    files = glob.glob(root, recursive = True)
    files = list(set(files))
    images = OrderedDict()
    for c in classes:
        images[classes.index(c)] = []
        pattern = re.compile(c)
        new_files = copy(files)
        for file in files:
            if pattern.search(file) is not None:
                if cv2.imread(file) is not None:
                    images[classes.index(c)].append(file)
                new_files.remove(file)
        files = new_files
    return images

def get_unsegmented_image_files(root = "data\samples*\*.bmp"):
    files = glob.glob(root)
    bad_files = []
    for file in files:
        if cv2.imread(file) is None:
            bad_files.append(file)
    for file in bad_files:
        files.remove(file)
    return files

def get_histogram(file, bins = 5, color_space = cv2.COLOR_BGR2YCrCb):
    image = cv2.imread(file)
    image = cv2.cvtColor(image, color_space)
    if len(image.shape) > 2:
        n_channels = image.shape[2]
    else:
        n_channels = 1
    hists = []
#    scale = lambda x: (x - x.min())/(x.max() - x.min())
    for channel in range(n_channels):
        if n_channels > 1:
            img_slice = image[:,:,channel]
        else:
            img_slice = image
#        image[:,:,channel] = scale(image[:,:,channel])
        color_histogram = np.histogram(img_slice, bins = bins, density = True)[0]
        hists.append(color_histogram.astype('float32'))
    return(hists)

def get_histograms(file_dict = None,
                     color_space = cv2.COLOR_BGR2YCrCb, 
                     bins = 15):
    if file_dict is None:
        file_dict = get_image_files()
    hist_distance_dict = {}
    for c in file_dict.keys():
        image_files  = file_dict[c]
        correlation = []
        position = 0
        for image_file in image_files:
            all_hists = []
            hists = get_histogram(image_file, color_space = color_space, bins = bins)
            for histogram in hists:
                all_hists.extend(histogram)
            position += 1
            correlation.append(np.hstack(all_hists).ravel())
        print(np.vstack(correlation).shape)
        hist_distance_dict[c] = np.vstack(correlation)
    return hist_distance_dict

def get_color_histogram_features(file_dict):
    features = []
    for color_space in [cv2.COLOR_BGR2GRAY, 
                        cv2.COLOR_BGR2RGB, 
                        cv2.COLOR_BGR2HSV
                        ]:
        color_hists = get_histograms(file_dict = file_dict, 
                                     color_space = color_space)
        color_matrix = np.vstack([v for v in color_hists.values()])
        features.append(color_matrix)
    return np.hstack(features)

def get_lbp_texture(file_dict, bin_size = 15):
    n_files = np.sum([len(fileset) for fileset in file_dict.values()])
    textures = np.zeros((n_files, bin_size))
    t_position = 0
    for c in range(max(file_dict.keys())):
        files = file_dict[c]
        matrix = np.zeros((len(files), bin_size))
        position = 0
        for file in files:
            img = cv2.imread(file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lbp = local_binary_pattern(img, 1, 3, method = 'ror')
            histogram = np.histogram(lbp, density = True, bins = bin_size)[0]
            matrix[position, :] = histogram
            position += 0
        end = t_position + len(files)
        textures[t_position:end, :] = matrix
        t_position = end
    return textures

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

def gaussian_blur_channels(image, kernel = (3,3), mean = 0):
    if len(image.shape) <= 2:
        return cv2.GaussianBlur(image, kernel, mean)
    else:
        n_channels = image.shape[2]
        for channel in range(n_channels):
            image[:,:,channel] = cv2.GaussianBlur(image[:,:,channel], kernel, mean)
    return image

def bilateral_blur(img, diameter = 20, sigmaColor = 20, sigmaSpace = 20):
    image = copy(img)
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)

def edge_extraction(img, d1 = 10, d2 = 10, t1 = 30, t2 = 30):
    image = denoise(img, d1, d2)
    edges = cv2.Canny(image,t1,t2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges

def word_extraction(img, max_area_fraction = .4):
    new_img = copy(img)
    if len(img.shape) > 2:
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(copy(new_img), 400, 400)
    kernel = np.ones((5,5))
    edges = cv2.dilate(edges, kernel)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, heirarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    good_contours = []
    max_area = max_area_fraction*img.shape[0]*img.shape[1]
    for contour in contours:
        if cv2.contourArea(contour) < max_area:
            good_contours.append(contour)
    cv2.drawContours(new_img, good_contours, -1, 0, cv2.FILLED)
    return crop_image(img, new_img)
    

def get_blobs(img):
    image = edge_extraction(img)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#    open_kernel =cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
#    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, open_kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, close_kernel)
    contours, heirarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(image.shape)
    cv2.drawContours(blank, contours,  -1, 255, cv2.FILLED)
    return blank

def lbp(img):
    image = denoise(img)
    lbp_image = local_binary_pattern(img, 1, 2, method = 'uniform')
    return lbp_image

def crop_image(img, reference_image = None, upper_bound = 220, lower_bound = 50):
    if reference_image is None:
        reference_image = copy(img)
#    reference_image = denoise(reference_image)
    if len(reference_image.shape) > 2:
        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    image_size = (img.shape[0], img.shape[1])
    valid_coords = np.argwhere( (reference_image > lower_bound)*(reference_image < upper_bound) )
    x0, y0 = valid_coords.min(axis = 0)
    x1, y1 = valid_coords.max(axis = 0) + 1
    if len(reference_image.shape) > 2:
        cropped_image = img[x0:x1, y0:y1]
    else:
        cropped_image = img[x0:x1, y0:y1, :]
    return cv2.resize(cropped_image, image_size)

def denoise(img, d1 = 8, d2 = 8):
    if len(img.shape) > 2:
        denoiser = lambda x: cv2.fastNlMeansDenoisingColored(x, None, d1, d2)
    else:
        denoiser = lambda x: cv2.fastNlMeansDenoising(x, None, d1, d2)
    return denoiser(img)

def show_images(image_set, compare_func = None):
    window_name = 'images'
    cv2.namedWindow(window_name)
    if compare_func is None:
        on_change = lambda x: cv2.imshow(window_name, image_set[x])
    else:
        on_change = lambda x: cv2.imshow(window_name, 
                                         np.hstack([image_set[x], compare_func(copy(image_set[x]))]))
    cv2.createTrackbar(window_name, window_name, 0, len(image_set) - 1, on_change)
    on_change(0)
    
def preprocess(images):
    cropped = [word_extraction(i) for i in images]
    cleaned = [denoise(i) for i in cropped]
    return cleaned

get_classes = lambda file_dict: np.hstack([k*np.ones((len(v), )) for k,v in file_dict.items()]).astype('int32')

imshow = lambda x: cv2.imshow('image', x)
imshow_big = lambda x: imshow(cv2.pyrUp(x))
imshow_func = lambda function, img: imshow( np.hstack([img, function(copy(img))]) )

sample_images = get_image_files(root = 'data/test*/**/*.jpg', classes = Constants.test_classes)
images = []
for files in sample_images.values():
    #aww yeah one line
    images.extend( [cv2.cvtColor(cv2.resize(cv2.imread(im), (300,300), interpolation = cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB) for im in files] )
grays = [cv2.cvtColor(copy(im), cv2.COLOR_BGR2GRAY) for im in images]

clean_images = preprocess(images)
#lbp = lambda x: local_binary_pattern(copy(x), 1, 2, method = 'ror')
#gaussian = lambda x: cv2.GaussianBlur(copy(x), (5,5), 0)
show_images(clean_images)
