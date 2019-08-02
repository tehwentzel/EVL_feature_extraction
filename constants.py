# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:09:10 2019

@author: Andrew
"""
import numpy as np

class Constants():

    default_image_pickle = 'processes_images_all'
    default_file_pickle = 'image_files_all'
    default_feature_pickle = 'image_features_all'
    classes = ['Electron-Mix',
               'Transmission',
               'ReporterANDImmunoHistochem',
               'Fluorescence-Mix',
               'Light-Mix',
               'InSituHybridization'
               ]
    top_level_classes = ['Experimental', 'Graphics', 'MolecularStructure', 'Organs']
    class_hierarchy = {
            'Experimental': ['Microscopy', 'Plate', 'Gel'],
            'Microscopy': ['Electron', 'Light', 'Fluorescence'],
            'Organs': ['MRI', 'Xray'],
            'Graphics': ['BarChart', 'Diagram', 'LineChart', 'ScatterPlot', 'Table', 'Other'],
            'MolecularStructure': ['3d', 'Chemical', 'MacromoleculeSequence'],
            'MacromoleculeSequence': ['DNA', 'Protein']
            }

    test_classes = []
    for item in class_hierarchy.values():
        test_classes.extend(item)

    image_root = 'data\images_2\\'
    test_image_root = 'data\test\\'


    top_edge_kernel = np.array([
                [1,1,1],
                [0,0,0],
                [-1,-1,-1]
            ])
    bottom_edge_kernel = -1*top_edge_kernel
    left_edge_kernel = np.array([
                [1,1,-1],
                [1,0,-1],
                [1,0,-1]
            ])
    right_edge_kernel = -1*left_edge_kernel
    all_edges = [top_edge_kernel, bottom_edge_kernel, left_edge_kernel, right_edge_kernel]
    image_size = (400,400)
    image_area = 1600

    bgr2lcc_operator = np.array([[.114, .587, .299],
                                   [-1, .5, .5],
                                   [0, -.866, 866]])

    n_bovw_groups = 500
    codebook_file = 'sift_word_codebook'
    #I think this is reversed but you need to transpose for skfuzzy so it should for with skfuzzy.cmean_predict
    orgb_fuzzy_cluster_centers  = np.array([[ 6.70249862e-01, -6.95485846e-04,  6.04555859e+02],
       [ 4.21342001e-01, -2.26813048e-02,  3.52587668e+02],
       [ 9.58630618e-02,  5.03485379e-03,  4.26474121e+01],
       [ 5.74999276e-01, -1.49060309e-02,  5.04311618e+02],
       [ 5.74141427e-02, -4.99449947e-03,  1.83976411e+01],
       [ 4.81927539e-01,  3.64892526e-03,  4.28126164e+02],
       [ 4.72010024e-02, -3.75556805e-03,  2.23619378e+00],
       [ 8.74603819e-01,  1.30162570e-02,  7.84101283e+02],
       [ 2.56089829e-01,  4.72003428e-03,  2.33343294e+02],
       [ 3.51679852e-01, -3.83774681e-02,  2.97186123e+02],
       [ 7.55759001e-01,  3.92551386e-03,  6.88698425e+02],
       [ 1.52162455e-01, -8.61111952e-03,  1.03265138e+02],
       [ 5.93191040e-01,  6.38585549e-03,  5.31314052e+02],
       [ 4.67033887e-01,  4.18713762e-02,  4.48289050e+02],
       [ 2.94047827e-01, -9.74868092e-03,  2.64787115e+02],
       [ 1.97651184e-01,  4.24523482e-03,  1.61147324e+02],
       [ 3.97668140e-01,  2.17476638e-02,  3.90028659e+02],
       [ 5.14497011e-01,  9.06084873e-03,  4.74277487e+02],
       [ 1.20126680e-01,  2.91625049e-03,  6.98300290e+01],
       [ 4.15847061e-01, -1.55867178e-02,  3.72262841e+02],
       [ 8.59938866e-01, -1.22969257e-03,  7.52463330e+02],
       [ 6.52636946e-01, -3.24046597e-03,  5.82152754e+02],
       [ 9.75093442e-01,  1.63765297e-02,  8.59038949e+02],
       [ 2.27811987e-01,  3.79994035e-03,  1.97233875e+02],
       [ 3.46402912e-01,  7.92054756e-03,  3.27114494e+02],
       [ 7.03487837e-01,  8.74088623e-04,  6.31747707e+02],
       [ 9.15452678e-01,  2.40729582e-02,  8.25232054e+02],
       [ 4.83374474e-01, -1.60115119e-02,  4.11469430e+02],
       [ 6.28288384e-01,  3.03702824e-03,  5.57958145e+02],
       [ 7.02413852e-01,  2.57408942e-02,  6.58540815e+02],
       [ 1.62429788e-01,  6.75528096e-03,  1.34739657e+02],
       [ 7.91632982e-01,  6.64394888e-03,  7.20270806e+02]])