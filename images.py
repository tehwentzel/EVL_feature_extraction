import cv2
import numpy as np
import glob
from copy import copy
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
from skimage.feature import local_binary_pattern
from constants import Constants
from crop import grabcut_crop, crop

class ImageGenerator():

    def __init__(self, root = 'data\images\**\*.jpg', class_roots = None,
                 crop = True, scale = Constants.image_size,
                 remove_borders = True, denoise = True, shuffle = True):
        classes = self.get_classes(class_roots)
        self.file_dict = self.get_image_files(root, classes)
        if shuffle:
            self.shuffle_files()
        class_count = [len(x) for x in self.file_dict.values()]
        self.class_ratios = np.array([x/np.sum(class_count) for x in class_count])
        self.inverse_class_positions = [x - 1 for x in class_count]
        self.crop_flag = crop
        self.scale = scale
        self.remove_borders_flag = remove_borders
        self.denoise_flag = denoise
        self.num_images = np.sum(class_count)

    def get_classes(self, class_root):
        if class_root is None:
            return list(Constants.test_classes)
        if isinstance(class_root, str):
            return Constants.class_hierarchy[class_root]
        classes = []
        for c in class_root:
            if c in Constants.class_hierarchy:
                classes.extend(Constants.class_hierarchy[c])
        return classes

    def shuffle_files(self):
        for key, file_list in self.file_dict.items():
            np.random.shuffle(file_list)

    def get_image_files(self, root, classes = None):
        if classes is None:
            classes = Constants.classes
        self.class_names = classes
        files = glob.glob(root, recursive = True)
        files = list(set(files))
        images = OrderedDict()
        for c in classes:
            images[classes.index(c)] = []
            pattern = re.compile(c)
            new_files = copy(files)
            for file in files:
                if pattern.search(file) is not None:
                    image = cv2.imread(file)
                    if image is not None and image.std() > 0.0001:
                        images[classes.index(c)].append(file)
                    new_files.remove(file)
            files = new_files
        return images

    def files_to_labels(self, files):
        file_lookup = {}
        for key, fileset in self.file_dict.items():
            for file in fileset:
                file_lookup[file] = key
        classes = np.empty((len(files))).astype('uint8')
        i = 0
        for file in files:
            classes[i] = file_lookup[file]
            i += 1
        return classes


    def get_images(self, num_images = 10, base_classes = None):
        if base_classes is None:
            base_classes = list(self.file_dict.keys())
        good_classes = np.argwhere(np.array(self.inverse_class_positions) >= 0).astype('int32')
        classes = np.array(base_classes)[good_classes.ravel()]
        images = []
        labels = []
        for idx in range(num_images):
            label = np.random.choice(classes)
            file_position = self.inverse_class_positions[label]
            if file_position <= 0:
                pos = np.argwhere(classes == label)
                classes = np.delete(classes, pos)
            self.inverse_class_positions[label] -= 1
            if self.reset_classes():
                classes = base_classes
            image_file = self.file_dict[label][file_position]
            images.append( self.process_image_file(image_file) )
            labels.append(image_file)
        return images, labels

    def reset_classes(self, count = 0):
        if np.sum(self.inverse_class_positions) <= count:
            self.inverse_class_positions = [len(x) - 1 for x in self.file_dict.values()]
            return True
        else:
            return False

    def process_image_file(self, image_file):
        image = cv2.imread(image_file)
        if self.crop_flag:
            cropped_image = crop(image)
            imsize = lambda i: i.shape[0]*i.shape[1]
            #skip if cropped image is super small
            if 4*imsize(cropped_image) > imsize(image):
                image = cropped_image
        if self.scale:
#            image = cv2.resize(image, self.scale)
            image = self.scale_and_crop(image, self.scale)
        if self.remove_borders_flag:
            image = self.word_extraction(image)
        if self.denoise_flag:
            image = self.denoise(image)
        return image

    def denoise(self, img, d1 = 5, d2 = 5):
        if len(img.shape) > 2:
            denoiser = lambda x: cv2.fastNlMeansDenoisingColored(x, None, d1, d2)
        else:
            denoiser = lambda x: cv2.fastNlMeansDenoising(x, None, d1, d2)
        return denoiser(img)


    def word_extraction(self, img, max_area_fraction = .4,
                        lower_canny_thresh = 200,
                        upper_canny_thresh = 300,
                        dilation_kernel = np.ones((5,5))):
        new_img = copy(img)
        if len(img.shape) > 2:
            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    #    new_img = cv2.bilateralFilter(new_img, 20,20, 20)
        edges = cv2.Canny(copy(new_img), lower_canny_thresh, upper_canny_thresh)
        edges = cv2.dilate(edges, dilation_kernel)
        _, contours, heirarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        good_contours = []
        max_area = max_area_fraction*img.shape[0]*img.shape[1]
        for contour in contours:
            if cv2.contourArea(contour) < max_area:
                good_contours.append(contour)
        cv2.drawContours(new_img, good_contours, -1, 0, cv2.FILLED)
        return self.crop_image(img, new_img)

    def crop_image(self, img, reference_image = None, upper_bound = 220, lower_bound = 40):
        if reference_image is None:
            reference_image = copy(img)
        if len(reference_image.shape) > 2:
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        image_size = (img.shape[0], img.shape[1])
        valid_coords = np.argwhere( (reference_image > lower_bound)*(reference_image < upper_bound) )
        if len(valid_coords) <= Constants.image_area/3:
            if upper_bound < 245 and lower_bound > 10:
                return self.crop_image(img, reference_image, upper_bound + 10, lower_bound - 10)
            else:
                print('error in cropping image')
                return img
        x0, y0 = valid_coords.min(axis = 0)
        x1, y1 = valid_coords.max(axis = 0) + 1
        if len(reference_image.shape) > 2:
            cropped_image = img[x0:x1, y0:y1]
        else:
            cropped_image = img[x0:x1, y0:y1, :]
        return cv2.resize(cropped_image, image_size)

    def scale_and_crop(self, image, target_shape):
        fx = target_shape[1]/image.shape[1]
        fy = target_shape[0]/image.shape[0]
        scale = min([fx, fy])
        image = cv2.resize(image, None, fx = scale, fy = scale)
#        height_offset = max([int((image.shape[0] - target_shape[0])//2), 0])
#        width_offset = max([int((image.shape[1] - target_shape[1])//2), 0])
#        image = image[height_offset: target_shape[0] + height_offset,
#                      width_offset: target_shape[1] + width_offset, :]
        return image


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
                image = cv2.imread(file)
                if image is not None and image.std() > 0.0001:
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

def edge_extraction(img, d1 = 10, d2 = 10, t1 = 20, t2 = 60):
    edges = cv2.Canny(image,t1,t2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges


def _crop_dict(image_file_dict):
    images = []
    for file_list in image_file_dict.values():
        for file in file_list:
            image = crop(file)
            image = cv2.resize(image, Constants.image_size)
            images.append(image)
    return images

imshow = lambda x: cv2.imshow('image', x)
imshow_big = lambda x: imshow(cv2.pyrUp(x))
imshow_func = lambda function, img: imshow( np.hstack([img, function(copy(img))]) )
