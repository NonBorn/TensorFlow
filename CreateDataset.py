import regex as re
import random as rng
import os, errno
import shutil

# define file path of the folder of images
path_init = '/Users/nonborn/Desktop/Test Dataset'
#target = '/Users/nonborn/Desktop/Test Dataset'
target_train = '/Users/nonborn/Desktop/whole_test'

def exclude_os_files(files_path):
    # Regex expression for hidden mac os files
    reg = re.compile("^\.")
    filenames = [x for x in os.listdir(files_path) if not reg.match(x)]
    return filenames


class_path = exclude_os_files(path_init)
num_of_classes = len(class_path)

# debugging - shows the list of folders within the parent folder
print(class_path)

tmp_img_list = []

i = 0
"""num_of_classes"""
for f in range(0, num_of_classes    ):
    #print(class_path[f])
    source_path = path_init + '/' + class_path[f]
    target_path = target_train
    #print(source_path)
    #print(target_path)
    i=0
    for j in os.listdir(source_path):
        i=i+1
        sp2 = source_path + '/' + j
        newname = class_path[f] + '_' + str(i) + '.png'
        shutil.copy(sp2, target_path + '/' +newname)