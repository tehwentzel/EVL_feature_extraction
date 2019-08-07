import numpy as np
import cv2
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import KDTree
from constants import Constants
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from lshash.lshash import LSHash

class PCAKDTree():

    def __init__(self, points, n_features = 15):
        self.n_features = len(points)
        self.pca = PCA(n_features).fit(points)
        self.points = self.pca.transform(points)
        self.kd_tree = KDTree(self.points)

    def get_nearest(self, keypoints):
        transformed = self.pca.transform(keypoints)
        return self.kd_tree.query(transformed)[1]

class LSHAnn():

    def __init__(self, points, hash_size = 10, n_hashtables = 25):
        self.lsh = LSHash(hash_size, points.shape[1], n_hashtables)
        self.n_points = points.shape[0]
        for n in range(self.n_points):
            self.lsh.index(points[n], extra_data = n)

    def get_words(self, keypoints):
        word_count = np.zeros((self.n_points,))
        for k in range(keypoints.shape[0]):
            out = self.lsh.query(keypoints[k], 1)
            if len(out) > 0:
                if len(out[0][0]) == 2:
                    index = out[0][0][1]
                    if index == 0:
                        print('nope')
                else: #I think if I add in 0 as the extra data it evaluates to False?
                    index = 0
                word_count[index] += 1
            else:
                print('ANN missed')
        return word_count

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
    codebook = MiniBatchKMeans(n_clusters = n_total_clusters).fit(clusters)
    codebook = LSHAnn(codebook.cluster_centers_)
    print('clustering complete...')
    return codebook, image_sifts


def extract_visual_words(descriptors, codebook):
    if codebook is None:
        return None

    if isinstance(codebook, PCAKDTree):
        num_words = codebook.kd_tree.data.shape[0]
        def get_words(k):
            words = np.zeros((num_words,))
            nearest = codebook.get_nearest(k)
            for word in nearest:
                words[word] += 1
            return words
    elif isinstance(codebook, LSHAnn):
        num_words = codebook.n_points
        def get_words(k):
            return codebook.get_words(k)

    bow = np.zeros((len(descriptors),  num_words))
    for i in range(len(descriptors)):
        keypoints = descriptors[i]
        if keypoints is not False:
            bow[i] = get_words(keypoints)
    return bow


def bovw(image, n_img_clusters = 50, n_total_clusters = None, dense = True):
    codebook, descriptors = bovw_codebook(image, n_img_clusters, n_total_clusters, dense)
    bow = extract_visual_words(descriptors, codebook)
    return bow