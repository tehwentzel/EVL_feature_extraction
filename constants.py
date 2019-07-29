# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:09:10 2019

@author: Andrew
"""
import numpy as np

class Constants():

    classes = ['Electron-Mix',
               'Transmission',
               'ReporterANDImmunoHistochem',
               'Fluorescence-Mix',
               'Light-Mix',
               'InSituHybridization'
               ]

    class_hierarchy = {
            'Experimental': ['Microscopy', 'Plate', 'Gel'],
            'Microscopy': ['Electron', 'Light', 'Fluorescence'],
            'Organs': ['MRI', 'Xray'],
            'Graphics': ['BarChart', 'Diagram', 'LineChart', 'ScatterPlot', 'Table', 'Other'],
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

    n_bovw_groups = 1000
    codebook_file = 'sift_word_codebook'