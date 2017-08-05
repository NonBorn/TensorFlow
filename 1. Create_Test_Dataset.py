import regex as re
import random as rng
import os, errno
import shutil

# define file path of the folder of images
path_init = '/Users/nonborn/Desktop/tx_000100000000/Train Dataset'
target = '/Users/nonborn/Desktop/tx_000100000000/Test Dataset'

def exclude_os_files(files_path):
    # Regex expression for hidden mac os files
    reg = re.compile("^\.")
    filenames = [x for x in os.listdir(files_path) if not reg.match(x)]
    return filenames


class_path = exclude_os_files(path_init)
num_of_classes = len(class_path)

print(class_path) # debugging - shows the list of folders within the parent folder

tmp_img_list = []

for f in range(0, num_of_classes):
    source_path = path_init + '/' + class_path[f]
    target_path = target + '/' + class_path[f]
    files = os.listdir(source_path)  # list of files in current path
    tmp = exclude_os_files(source_path)  # exclude system files
    #print(len(tmp), int(0.1*len(tmp)))
    index = rng.sample(range(0, len(tmp)), int(0.1*len(tmp)))  # get a number of images per class - indexing
    #print(index)
    # print (class_path[f])
    # create paths of random images per class
    source_img_list = tmp_img_list + [source_path + '/' + files[s] for s in index]

    #print(source_img_list)
    #
    try:
        os.makedirs(target_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    for f in source_img_list:
        shutil.move(f, target_path)