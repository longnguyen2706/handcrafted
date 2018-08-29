import os
# /tmp/bottleneck/Nucleus/r06aug97.gidap.12--1---2.dat.jpg_inception_resnet_v2.txt
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
import scipy.io as sio
import collections



MATLAB_DIR = '/mnt/6B7855B538947C4E/Stage_1_To_Long'
IMAGE_BASE_DIR = os.path.join(MATLAB_DIR, 'image/JPEG_data')
CNN_FEATURE_BASE_DIR = os.path.join(MATLAB_DIR, 'generated_features/off_the_shelf')
# HANDCRAFTED_FEATURE_BASE_DIR = os.path.join(MATLAB_DIR, 'generated_features/handcrafted')
DATASET = 'Hela_JPEG'

IMAGE_DIR = os.path.join(IMAGE_BASE_DIR, DATASET)
CNN_FEATURE_DIR = os.path.join(CNN_FEATURE_BASE_DIR, 'Hela_JPEG')
# HANDCRAFTED_FEATURE_DIR = os.path.join(HANDCRAFTED_FEATURE_BASE_DIR, DATASET)
SAVE_MAT_DIR = '/home/long/Desktop/cnn_mat'

class MyDataset():
    def __init__(self, directory, test_size, val_size):
        self.directory = directory
        self.filenames = None
        self.labels = None
        self.label_names = None
        self.class_names = None
        self.categories = None
        self.test_size = test_size
        self.val_size = val_size

    def list_images(self):
        self.labels = os.listdir(self.directory)
        self.labels.sort()

        files_and_labels = []
        for label in self.labels:
            for f in os.listdir(os.path.join(self.directory, label)):
                files_and_labels.append((os.path.join(label, f), label))

        self.filenames, self.labels = zip(*files_and_labels)
        self.filenames = list(self.filenames)
        self.labels = list(self.labels)
        self.label_names = copy.copy(self.labels)
        unique_labels = list(set(self.labels))
        unique_labels.sort()

        label_to_int = {}
        for i, label in enumerate(unique_labels):
            label_to_int[label] = i

        self.labels = [label_to_int[l] for l in self.labels]
        self.class_names = unique_labels
        self.categories = list(set(self.labels))
        return


    def get_data(self):
        self.list_images()  # get image list
        return self.filenames, self.labels, self.label_names, self.class_names


def read_feature_file(filepath):
    with open(filepath, 'r') as feature_file:
        feature_string = feature_file.read()
    try:
        feature_values = [float(x) for x in feature_string.split(',')]
    except ValueError:
        print('Invalid float found')
    return np.asarray(feature_values)


def save_feature_to_mat(list_images, label_names, feature_dir, save_dir, feature_type='concat'):
    for i, image_path in enumerate(list_images):
        label_name = label_names[i]
        image_name = image_path.split('/')[-1]
        prefix = os.path.join(feature_dir, label_name, image_name)

        inception_features = read_feature_file(prefix+"_inception_v3.txt")
        resnet_features = read_feature_file(prefix+"_resnet_v2.txt")
        inception_resnet_features = read_feature_file(prefix+"_inception_resnet_v2.txt")
        if feature_type == 'concat':
            features= np.concatenate((inception_features, resnet_features, inception_resnet_features))
            assert (features.shape  == ((2048 *2 + 1536),)) # if no exception -> correct
            # print (features.shape)
        elif feature_type == 'inception_v3':
            features = inception_features
        elif feature_type == 'resnet_v2':
            features = resnet_features
        elif feature_type == 'inception_resnet_v2':
            features = inception_resnet_features
        else:
            raise Exception
        mat_category_dir = os.path.join(save_dir, label_name)
        if not os.path.exists(mat_category_dir):
            os.makedirs(mat_category_dir)

        mat_file = image_name.split('.jpg')[0] + '.mat'
        sio.savemat(os.path.join(mat_category_dir, mat_file), {'features': features})

    return


if __name__ == '__main__':
    dataset = MyDataset(directory=IMAGE_DIR, test_size=0.0, val_size=0.0)
    file_names, labels, label_names, class_names = dataset.get_data()

    print('train files shape: ', len(file_names))

    if not os.path.exists(SAVE_MAT_DIR):
        os.makedirs(SAVE_MAT_DIR)
    save_feature_to_mat(file_names, label_names, CNN_FEATURE_DIR, SAVE_MAT_DIR)

