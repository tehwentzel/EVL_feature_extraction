import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
from constants import Constants
from sklearn.feature_selection import mutual_info_classif

class BagOfWords():

    def __init__(self):
        self.codebook = None
        self.args = None

    def fit(self, x, c1 = 100, c2 = None, dense = True):
        self.codebook, _ = bovw_codebook(x, c1, c2, dense)

    def transform(self, x, y = None,  info_threshold = .05):
        bow = extract_visual_words(x, self.codebook)
        if y is not None:
            assert(bow.shape[0] == y.shape[0])
            info = mutual_info_classif(bow, y)
            args = np.argwhere(info > info_threshold)
            bow = bow[:, args]
        return bow

    def fit_transform(self, x, y = None,
                      info_threshold = .05,
                      c1 = 100, c2 = None,
                      dense = True):
        self.fit(x, c1, c2, dense)
        bow = self.transform(x, y, info_threshold)
        return bow

def root_descriptor(img, descriptor, dense = True):
    desc = get_descriptors(img, descriptor, dense)
    if desc is None:
        return desc
    desc = desc/(desc.sum(axis = 1).reshape(-1,1) + 1e-6)
    desc = np.sqrt(desc)
    return desc

def get_descriptors(img, detector, dense = True):
    if dense:
        step_size = 5
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0,img.shape[0], step_size) for x in range(0, img.shape[1], step_size)]
    else:
        kp = detector.detect(img,None)
    return detector.compute(img, kp)[1]


def bovw_codebook(images, n_img_clusters = 50, n_total_clusters = None, dense = True):
    if n_total_clusters is None:
        n_total_clusters = Constants.n_bovw_groups
    clusters = []
    image_sifts = []
    kmeans = KMeans(n_clusters = n_img_clusters)
    sift = cv2.xfeatures2d.SIFT_create()
    i = 0
    for image in images:
        print(i/len(images))
        i+=1
        keypoints = root_descriptor(image, sift, dense)
        if keypoints is None:
            image_sifts.append(False)
            continue
        image_sifts.append(keypoints)
        if dense is False and keypoints.shape[0] > n_img_clusters:
            kmeans.fit(keypoints)
            clusters.append(kmeans.cluster_centers_)
        else:
            clusters.append(keypoints)
    print('loading done, clustering...')
    clusters = np.vstack(clusters)
    codebook = KMeans(n_clusters = n_total_clusters, n_init = 5).fit(clusters)
    codebook = KDTree(codebook.cluster_centers_)
    print('clustering complete...')
    return codebook, image_sifts

def extract_visual_words(descriptors, codebook):
    if codebook is None:
        return None
    bow = np.zeros((len(descriptors), descriptors[0].shape[1]))
    for i in range(len(descriptors)):
        keypoints = descriptors[i]
        if keypoints:
            words = codebook.query(keypoints)[1]
            for word in words:
                bow[i,word] += 1
    return bow

def bovw(image, n_img_clusters = 50, n_total_clusters = None, dense = True):
    codebook, descriptors = bovw_codebook(image, n_img_clusters, n_total_clusters, dense)
    bow = extract_visual_words(descriptors, codebook)
    return bow