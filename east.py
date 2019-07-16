# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:32:18 2019

@author: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/
"""
import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def fill_characters(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10)
    characters = pytesseract.image_to_boxes(img,
                                            output_type = pytesseract.Output.DICT)
    n_characters = len(characters['page'])
    possible_characters = set(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k'])
    bbox_keys = ['left', 'bottom', 'right', 'top']
    boxes = []
    for char_num in range(n_characters):
        char = characters['char'][char_num]
        if char.lower() not in possible_characters:
            continue
        edges = [characters[key][char_num] for key in bbox_keys]
        bbox = [(edges[0], img.shape[0] - edges[3]), (edges[2], img.shape[0] - edges[1])]
        boxes.append(bbox)
    for bbox in boxes:
        img = cv2.rectangle(img, bbox[1], bbox[0], (0,0,0), 3)
    return(img)