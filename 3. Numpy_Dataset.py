import os
import regex as re
import numpy as np
from PIL import Image



def exclude_os_files(files_path):
    # Regex expression for hidden mac os files
    reg = re.compile("^\.")
    filenames = [x for x in os.listdir(files_path) if not reg.match(x)]
    return filenames

def get_numpy(fpath, H, W):
    im = Image.open(fpath)
    #im.show(); # for debugging purposes
    im = im.resize((H, W), Image.ANTIALIAS) # resize the image to 64 x 64
    #im.show();
    im = im.convert('L')  # to convert an image to grayscale
    im = np.asarray(im, dtype=np.float32)
    return im

def one_hot_function(word):
    # Vocabulary & 1 hot vectors
    text_idx = range(0, num_of_classes)
    #print(text_idx)
    vocab_size = len(class_path1)
    text_length = len(text_idx)
    one_hot = np.zeros(([vocab_size, text_length]))
    one_hot[text_idx, np.arange(text_length)] = 1
    one_hot = one_hot.astype(int)
    return one_hot[class_path1.index(word)]

def random_batch (dir, index, batch_size):
    if index + batch_size <= len(dir):
        tmp_list = [x for x in dir[index:index + batch_size]]
        classes = [x.rsplit('_', 1)[0].rsplit('/', 1)[1] for x in tmp_list]

        batch_xx = np.asarray([get_numpy(x) for x in dir[index:index + batch_size]])
        batch_yy = np.asarray([one_hot_function(y) for y in classes])
        t_index = index + batch_size + 1
    else:
        tmp_list = [x for x in dir[index:index + len(dir)]]
        classes = [x.rsplit('_', 1)[0].rsplit('/', 1)[1] for x in tmp_list]

        batch_xx = np.asarray([get_numpy(x) for x in dir[index:index + len(dir)]])
        batch_yy = np.asarray([one_hot_function(y) for y in classes])
        t_index = 0
    return batch_xx, batch_yy, t_index


path = '/Users/nonborn/Desktop/tx_000100000000/Numpy/Train'
train_path = '/Users/nonborn/Desktop/tx_000100000000/Train Dataset'
test_path = '/Users/nonborn/Desktop/tx_000100000000/Test Dataset'

target_path = '/Users/nonborn/Desktop/tx_000100000000/Numpy/Train'

#for i in range(0:len(os.listdir(path)))

print(len(os.listdir(train_path)))
print(len(os.listdir(test_path)))


tr_files = exclude_os_files(train_path)
sketch_size = 128

for i in range(0, len(os.listdir(train_path))-1):
    tpath = train_path + '/' + tr_files[i]
    img = get_numpy(tpath, sketch_size, sketch_size)
    np.save(target_path + '/' + tr_files[i], img)