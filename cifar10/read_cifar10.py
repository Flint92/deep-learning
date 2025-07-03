def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


labels = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

import glob
import numpy as np
import os
import cv2

# train_list = glob.glob('./data_batch_*')
# folder_name = 'train'

train_list = glob.glob('./test_batch')
folder_name = 'test'

for l in train_list:
    l_dict = unpickle(l)

    for im_idx, im_data in enumerate(l_dict[b'data']):
        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]

        im_label_name = labels[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])
        im_data = np.transpose(im_data, (1, 2, 0))

        # import cv2m
        # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        if not os.path.exists('./{}/{}'.format(folder_name, im_label_name)):
            os.makedirs('./{}/{}'.format(folder_name, im_label_name))

        cv2.imwrite('./{}/{}/{}'.format(folder_name,
                                        im_label_name,
                                        im_name.decode('utf-8')),
                    im_data)
