"""
Import required libraries
"""
import os
import numpy as np
from PIL import Image
import regex as re
import tensorflow as tf
import random as rng


"""""""""""""""""""""
Global Parameters:

"""""""""""""""""""""

# Resized source input image dimensions:
Height = 64;
Width = 64;
# images_per_class represent the batch size of tensorflow as images_per_class * num_of_classes
images_per_class = 5


"""""""""""""""""""""
Pre-Processing

"""""""""""""""""""""

# define file path of the folder of images
path_init = '/Users/nonborn/[MSc] Business Analytics/3rd Semester/[SQ-PT-2017] Social Network Analysis and Social Media Analytics/Assignment (tensorflow)/rendered_256x256/256x256/sketch/tx_000000000000'


def exclude_os_files(files_path):
    # Regex expression for hidden mac os files
    reg = re.compile("^\.")
    filenames = [x for x in os.listdir(files_path) if not reg.match(x)]
    return filenames


def get_numpy(fpath):
    im = Image.open(fpath)
    # im.show(); # for debugging purposes
    im = im.resize((Height, Width), Image.ANTIALIAS) # resize the image to 64 x 64
    # im.show();
    im = im.convert('L')  # to convert an image to grayscale
    im = np.asarray(im, dtype=np.float32)
    return im


def one_hot_function(word):
    # Vocabulary & 1 hot vectors
    text_idx = range(0, num_of_classes)
    print(text_idx)
    text_length = len(text_idx)
    one_hot = np.zeros(([num_of_classes, text_length]))
    one_hot[text_idx, np.arange(text_length)] = 1
    one_hot = one_hot.astype(int)
    return one_hot[path.index(word)]


def random_batch():
    class_path = exclude_os_files(path_init)
    print(class_path)
    random_images = [];
    for f in range (0, num_of_classes):
        current_path = path_init + '/' + class_path[f]
        files = os.listdir(current_path)     # list of files in current path
        tmp = exclude_os_files(current_path)  # exclude system files
        index = rng.sample(range(0, len(tmp)), images_per_class)  # get a number of images per class - indexing
        random_images_features = random_images + [current_path + '/' + files[s] for s in index]  # create paths of random images per class
        random_images_labels = one_hot_function(files[s])
    #print random_images
    return random_images_features, random_images_labels



# Variables Initialization
class_path = exclude_os_files(path_init)
num_of_classes = len(class_path);


print (class_path);
print (num_of_classes)
print (images_per_class*num_of_classes)

random_batch()


