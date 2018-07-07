from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import copy

from sklearn.utils import Bunch

import load_CNN_features
import seaborn as sns

from surf.surf_bow import SURF_BOW
from svm_classifier import SVM_CLASSIFIER

sns.set()

IMAGE_DIR = '/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG'
FEATURE_DIR = '/mnt/6B7855B538947C4E/Dataset/features/off_the_shelf'
OUT_MODEL1 = '/mnt/6B7855B538947C4E/handcraft_models/stage1.pkl'
OUT_MODEL2 = '/mnt/6B7855B538947C4E/handcraft_models/stage2.pkl'
# PARAM_GRID = {'linearsvc__C': [1, 5, 10, 50]}
HYPER_PARAMS = {
    'pow_min': -15,
    'pow_max': 15,
    'base': 2,
    'pow_step':1,
    'type': 'linearsvc__C'
}

CLASSIFIER = svm.LinearSVC()
NUM_OF_WORDS = 1000
T = [0.3, 0.4, 0.5, 0.6, 0.7]
class MyDataset():
    def __init__(self, directory, test_size, val_size):
        self.directory = directory
        self.filenames = None
        self.labels = None
        self.label_names = None
        self.categories = None
        self.test_size = test_size
        self.val_size = val_size

    def list_images(self):
        self.labels = os.listdir(self.directory)
        self.labels.sort()

        files_and_labels = []
        for label in self.labels:
            for f in os.listdir(os.path.join(self.directory, label)):
                files_and_labels.append((os.path.join(self.directory, label, f), label))

        self.filenames, self.labels = zip(*files_and_labels)
        self.filenames = list(self.filenames)
        self.labels = list(self.labels)
        self.label_names = copy.copy(self.labels)
        unique_labels = list(set(self.labels))

        label_to_int = {}
        for i, label in enumerate(unique_labels):
            label_to_int[label] = i

        self.labels = [label_to_int[l] for l in self.labels]
        self.categories = list(set(self.labels))
        return

    def get_data(self):
        self.list_images()  # get image list

        dataset = Bunch(
            data=np.asarray(self.filenames),
            label_names=np.asarray(self.label_names), labels=np.asarray(self.labels),
            DESCR="Dataset"
        )
        print(dataset.data.shape)
        print(dataset.label_names)
        train_files, test_files, train_labels, test_labels, train_label_names, test_label_names \
            = train_test_split(dataset.data, dataset.labels, dataset.label_names, test_size=self.test_size)
        train_files, val_files, train_labels, val_labels, train_label_names, val_label_names \
            = train_test_split(train_files, train_labels, train_label_names, test_size=self.val_size)
        print('train size: ', train_labels.shape)
        return train_files, train_labels, train_label_names, \
               val_files, val_labels, val_label_names, \
               test_files, test_labels, test_label_names


def gen_grid(hyper_params):
    grid_params = []
    for i in range(hyper_params['pow_max']-hyper_params['pow_min']+1):
        if (i % hyper_params['pow_step'] == 0):
            grid_params.append(pow(hyper_params['base'],hyper_params['pow_min'] + i))
    params_grid = {hyper_params['type']: grid_params}
    print('param grids for HYPER PARAMS: ', hyper_params, params_grid)
    return params_grid

def get_CNN_features(train_files, train_labels, train_label_names,
                  val_files, val_labels, val_label_names,
                  test_files, test_labels, test_label_names):
    train_CNN_features = load_CNN_features.get_features(train_files, train_label_names, FEATURE_DIR)
    val_CNN_features = load_CNN_features.get_features(val_files, val_label_names, FEATURE_DIR)
    test_CNN_features = load_CNN_features.get_features(test_files, test_label_names, FEATURE_DIR)
    return train_CNN_features, val_CNN_features, test_CNN_features

def get_BOW_features(train_files, train_labels, train_label_names,
                  val_files, val_labels, val_label_names,
                  test_files, test_labels, test_label_names):
    surf_bow = SURF_BOW(num_of_words=NUM_OF_WORDS)
    surf_bow.build_vocab(train_files)
    train_surf_features = surf_bow.extract_bow_hists(train_files)
    val_surf_features = surf_bow.extract_bow_hists(val_files)
    test_surf_features = surf_bow.extract_bow_hists(test_files)
    return train_surf_features, val_surf_features, test_surf_features

def get_2_stage_performance(cls1, cls2, dataset, CNN_features, surf_features, labels, label_names):
    for t in T:
        Y = []
        for i, features in enumerate(CNN_features):
            y1 = cls1.trained_model.predict([features])[0]
            cs = cls1.cal_CS(features, y1, dataset.categories)
            if (cs < 1 - t):
                print("*** Stage 1 reject with t, cs = ", t, cs, " ***")
                features_bow = surf_features[i]
                y2 = cls2.trained_model.predict([features_bow])[0]
                print("*** y1, y2: ", y1, y2, " ***")
                Y.append(y2)
            else:
                print("*** Stage 1 accept with t, cs = ", t, cs, " ***")
                Y.append(y1)
        print("Classification report with t = ", t)
        print(classification_report(Y, labels,
                                    target_names=label_names))
        print("----------------------------")


def main():
    dataset = MyDataset(directory=IMAGE_DIR, test_size=0.2, val_size=0.25)
    train_files, train_labels, train_label_names, \
    val_files, val_labels, val_label_names, \
    test_files, test_labels, test_label_names = dataset.get_data()

    params_grid = gen_grid(HYPER_PARAMS)

    train_CNN_features, val_CNN_features, test_CNN_features = get_CNN_features(
        train_files, train_labels, train_label_names,
        val_files, val_labels, val_label_names,
    test_files, test_labels, test_label_names)

    train_surf_features, val_surf_features, test_surf_features = get_BOW_features(
        train_files, train_labels, train_label_names,
        val_files, val_labels, val_label_names,
        test_files, test_labels, test_label_names
    )

    # now train stage 1
    cls1 = SVM_CLASSIFIER(params_grid, CLASSIFIER, OUT_MODEL1)
    cls1.prepare_model()
    cls1.train(train_CNN_features, train_labels)
    print ("Finish train stage 1")
    print ("Now eval stage 1 on val set")
    cls1.test(val_CNN_features, val_labels, val_label_names)
    print("Now eval stage 1 on test set")
    cls1.test(test_CNN_features, test_labels, test_label_names)
    print ("---------------------")

    # now train stage 2
    cls2 = SVM_CLASSIFIER(params_grid, CLASSIFIER, OUT_MODEL2)
    cls2.prepare_model()
    cls2.train(train_surf_features, train_labels)
    print("Finish train stage 2")
    print("Now eval stage 2 on val set")
    cls2.test(val_surf_features, val_labels, val_label_names)
    print("Now eval stage 2 on test set")
    cls2.test(test_surf_features, test_labels, test_label_names)
    print("---------------------")

    # now train rejection rate
    cls1.get_centroids(train_CNN_features, train_labels, dataset.categories)
    print ("Now eval 2 stages on val set: ")
    get_2_stage_performance(cls1, cls2, dataset, val_CNN_features, val_surf_features, val_labels, val_label_names)
    print ("Now eval 2 stages on test set: ")
    get_2_stage_performance(cls1, cls2, dataset, test_CNN_features, test_surf_features, test_labels, test_label_names)

if __name__ == '__main__':
    main()
