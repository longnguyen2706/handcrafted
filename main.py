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

from traitlets import Bunch

import load_CNN_features
import seaborn as sns

from surf.surf_bow import SURF_BOW
from svm_classifier import SVM_CLASSIFIER

sns.set()


FEATURE_DIR = '/mnt/6B7855B538947C4E/Dataset/features/off_the_shelf'
OUT_MODEL = '/mnt/6B7855B538947C4E/handcraft_models/filename.pkl'


def list_images(directory):
    labels = os.listdir(directory)
    labels.sort()

    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    label_names = copy.copy(labels)
    unique_labels = list(set(labels))

    label_to_int = {}
    for i, label in enumerate(unique_labels):
        label_to_int[label] = i


    labels = [label_to_int[l] for l in labels]
    return filenames, labels, label_names

def get_dataset(filenames, labels, label_names):
    return Bunch(
        data=np.asarray(filenames),
        label_names=np.asarray(label_names), labels=np.asarray(labels),
        DESCR="Dataset"
    )


def main():
    # dataset = load_CNN_features.get_dataset(FEATURE_DIR)
    filenames, labels, label_names = list_images('/mnt/6B7855B538947C4E/Dataset/JPEG_data/Hela_JPEG')
    dataset =  get_dataset(filenames, labels, label_names)
    # print(dataset.data.shape)
    # print(dataset.label_names)

    train_files, test_files, train_labels, test_labels, train_label_names, test_label_names \
        = train_test_split(dataset.data, dataset.labels, dataset.label_names,  test_size=0.2)
    train_files, val_files, train_labels, val_labels, train_label_names, val_label_names \
        = train_test_split(train_files, train_labels, train_label_names, test_size=0.25)

    print ('train size: ', train_labels.shape)

    # train_CNN_features = load_CNN_features.get_features(train_files, train_label_names, FEATURE_DIR)
    # val_CNN_features = load_CNN_features.get_features(val_files, val_label_names, FEATURE_DIR)
    # test_CNN_features = load_CNN_features.get_features(test_files, test_label_names, FEATURE_DIR)

    surf_bow = SURF_BOW(num_of_words=100)
    surf_bow.build_vocab(train_files)
    train_surf_features = surf_bow.extract_bow_hists(train_files)
    val_surf_features = surf_bow.extract_bow_hists(val_files)
    test_surf_features = surf_bow.extract_bow_hists(test_files)

    print (train_surf_features.shape)


    PARAM_GRID = {'linearsvc__C': [1, 5, 10, 50]}
    CLASSIFIER = svm.LinearSVC()

    cls1 = SVM_CLASSIFIER(PARAM_GRID, CLASSIFIER, OUT_MODEL)
    cls1.prepare_model()
    cls1.train(train_surf_features, train_labels)
    cls1.test(val_surf_features, val_labels, val_label_names)
    cls1.test(test_surf_features, test_labels, test_label_names)


    # model = get_model(CLASSIFIER)
    # grid = grid_search(model, PARAM_GRID)

    # trained_model = train(grid, Xtrain, ytrain)
    # save(trained_model)
    # test(trained_model, dataset, Xtest, ytest)
    # trained_model = load()
    # test_x = Xtest[0]
    # test_y = ytest[0]
    # print (test_x, test_y)
    # ds = cal_DS(trained_model._final_estimator, Xtest, 0)


if __name__ == '__main__':
    main()
