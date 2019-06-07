import cv2
import numpy as np
import glob
from copy import copy
import re

root = 'data\images[_2]*\**\*.jpg'
files = glob.glob(root, recursive = True)
files = list(set(files))
print(len(files))
classes = ['Electron', 'Light', 'Fluorescence'] #there are technically more sub types but they're not really labeled?!?!?!?!?
images = dict()
for c in classes:
    images[c] = []
    pattern = re.compile( c)
    new_files = copy(files)
    for file in files:
        if pattern.search(file) is not None:
            if cv2.imread(file) is not None:
                images[c].append(file)
            new_files.remove(file)
    files = new_files
    print(len(images[c]))
