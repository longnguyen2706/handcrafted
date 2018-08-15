import argparse
import shutil
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import copy
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils import Bunch
import collections
import load_CNN_features
import seaborn as sns

import load_handcrafted_features
from sift.sift_bow import SIFT_BOW
from surf.surf_bow import SURF_BOW
from svm_classifier import SVM_CLASSIFIER
import matlab.engine
import datetime

# sns.set()

MATLAB_DIR = '/mnt/6B7855B538947C4E/Stage_1_To_Long'
IMAGE_BASE_DIR = os.path.join(MATLAB_DIR, 'image/JPEG_data')
CNN_FEATURE_BASE_DIR = os.path.join(MATLAB_DIR, 'generated_features/off_the_shelf')
HANDCRAFTED_FEATURE_BASE_DIR = os.path.join(MATLAB_DIR, 'generated_features/handcrafted')
DATASET = 'PAP_JPEG'

IMAGE_DIR = os.path.join(IMAGE_BASE_DIR, DATASET)
CNN_FEATURE_DIR = os.path.join(CNN_FEATURE_BASE_DIR, DATASET)
HANDCRAFTED_FEATURE_DIR = os.path.join(HANDCRAFTED_FEATURE_BASE_DIR, DATASET)
OUT_MODEL1 = '/mnt/6B7855B538947C4E/handcraft_models/stage1.pkl'
OUT_MODEL2 = '/mnt/6B7855B538947C4E/handcraft_models/stage2.pkl'
# PARAM_GRID = {'linearsvc__C': [1, 5, 10, 50]}

HYPER_PARAMS_1 = [
    {
        'pow_min': -15,
        'pow_max': 15,
        'base': 2,
        'pow_step': 1,
        'type': 'linearsvc__C',
    },
]
HYPER_PARAMS_2 = [
    {
        'pow_min': -15,
        'pow_max': 15,
        'base': 2,
        'pow_step': 1,
        'type': 'svc__C',
    },
    {
        'pow_min': -5,
        'pow_max': 5,
        'base': 2,
        'pow_step': 1,
        'type': 'svc__gamma'
    }
]

CLASSIFIER_1 = svm.LinearSVC()
# CLASSIFIER_2 = svm.SVC(kernel='rbf', class_weight='balanced')
CLASSIFIER_2 = svm.LinearSVC()
DIM_REDUCER = PCA(n_components=300, whiten=True, random_state=42,svd_solver='randomized')
NUM_OF_WORDS = 1000
T_MIN,T_MAX, T_STEP = 0.1, 0.9, 0.01

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

        dataset = Bunch(
            data=np.asarray(self.filenames),
            label_names=np.asarray(self.label_names), labels=np.asarray(self.labels),
            DESCR="Dataset"
        )
        print(dataset.data.shape)
        # print(dataset.label_names)
        train_files, test_files, train_labels, test_labels, train_label_names, test_label_names \
            = train_test_split(dataset.data, dataset.labels, dataset.label_names, test_size=self.test_size,  stratify=dataset.labels)
        train_files, val_files, train_labels, val_labels, train_label_names, val_label_names \
            = train_test_split(train_files, train_labels, train_label_names, test_size=self.val_size, stratify=train_labels)

        print('train size: ', train_labels.shape)
        self.data_split_report(train_label_names, 'train')
        self.data_split_report(val_label_names,'val' )
        self.data_split_report(test_label_names, 'test')

        return train_files, train_labels, train_label_names, \
               val_files, val_labels, val_label_names, \
               test_files, test_labels, test_label_names, self.class_names

    def data_split_report(self, label_names, set_name):
        class_freq = collections.Counter(label_names)
        print ("class freq for set %s "% set_name)
        print('*********')
        for key in sorted(class_freq):
            print( "%s: %s" % (key, class_freq[key]))
        print("-----------------------------------")

def gen_grid(hyper_params):
    params_grid ={}
    for hyper_param in hyper_params:
        grid_params = []
        for i in range(hyper_param['pow_max'] - hyper_param['pow_min'] + 1):
            if (i % hyper_param['pow_step'] == 0):
                grid_params.append(pow(hyper_param['base'], hyper_param['pow_min'] + i))
        params_grid[str(hyper_param['type'])]=grid_params
    print('param grids for HYPER PARAMS: ', hyper_params, params_grid)
    return params_grid

def gen_threshold(min, max, step):
    T = []
    for i in range (0, int ((max-min)/step)+1):
        t = min + i*step
        t = float("{0:.2f}".format(t)) # take only 2 decimal place
        T.append(t)
    print('generated T for min, max, step', min, max, step, T)
    return T

def get_CNN_features(train_files, train_labels, train_label_names,
                     val_files, val_labels, val_label_names,
                     test_files, test_labels, test_label_names, feature_type='concat'):
    train_CNN_features = load_CNN_features.get_features(train_files, train_label_names, CNN_FEATURE_DIR, feature_type)
    val_CNN_features = load_CNN_features.get_features(val_files, val_label_names, CNN_FEATURE_DIR, feature_type)
    test_CNN_features = load_CNN_features.get_features(test_files, test_label_names, CNN_FEATURE_DIR, feature_type)
    return train_CNN_features, val_CNN_features, test_CNN_features


# def get_BOW_features(train_files, train_labels, train_label_names,
#                      val_files, val_labels, val_label_names,
#                      test_files, test_labels, test_label_names):
#     surf_bow = SURF_BOW(num_of_words=NUM_OF_WORDS)
#     surf_bow.build_vocab(train_files)
#     train_surf_features = surf_bow.extract_bow_hists(train_files)
#     val_surf_features = surf_bow.extract_bow_hists(val_files)
#     test_surf_features = surf_bow.extract_bow_hists(test_files)
#     return train_surf_features, val_surf_features, test_surf_features

def get_handcrafted_features(train_files, train_labels, train_label_names,
                     val_files, val_labels, val_label_names,
                     test_files, test_labels, test_label_names):
    train_features = load_handcrafted_features.get_features(train_files, train_label_names, HANDCRAFTED_FEATURE_DIR)
    val_features = load_handcrafted_features.get_features(val_files, val_label_names, HANDCRAFTED_FEATURE_DIR)
    test_features = load_handcrafted_features.get_features(test_files, test_label_names, HANDCRAFTED_FEATURE_DIR)
    return train_features, val_features, test_features

def find_best_t(cls1, cls2, dataset, s1_features, s2_features, labels, class_names, T):
    accuracies = []
    for t in T:
       result= get_2_stage_performance(cls1, cls2, dataset, s1_features, s2_features, labels, class_names, t)
       acc = result['accuracy']
       accuracies.append(acc)
    best_acc =  max(accuracies)
    best_t = T[np.argmax(accuracies)]
    return best_t, best_acc # TODO: return best recall

def get_2_stage_performance(cls1, cls2, dataset, s1_features, s2_features, labels, class_names, t):
    Y = []
    for i, features in enumerate(s1_features):
        y1 = cls1.trained_model.predict([features])[0]
        cs = cls1.cal_CS(features, y1, dataset.categories)
        if (cs < 1 - t):
            # print("*** Stage 1 reject with t, cs = ", t, cs, " ***")
            features_s2 = s2_features[i]
            y2 = cls2.trained_model.predict([features_s2])[0]
            # print("*** y1, y2: ", y1, y2, " ***")
            Y.append(y2)
        else:
            # print("*** Stage 1 accept with t, cs = ", t, cs, " ***")
            Y.append(y1)
    print("Classification report with t = ", t)
    print(classification_report(labels,Y,
                                target_names=class_names))
    # now call precision
    precision, recall, fscore, support = score(labels, Y)
    accuracy = accuracy_score(labels, Y)
    print('accuracy: ', float("{0:.4f}".format(accuracy)))
    print("----------------------------")
    # print('precision: {}'.format(precision))
    # print('recall: {}'.format(recall))
    # print('fscore: {}'.format(fscore))
    # print('support: {}'.format(support))

    average_precision = 0
    for p in precision:
        average_precision = average_precision + p / len(precision)
    # print('average precision: ', average_precision)
    return {'accuracy': accuracy, 'average_precision': average_precision, 'precision': precision, 'recall': recall, 'fscore': fscore,
            'support': support}

def cal_mean_and_std(result_arr, name):
    mean = sum(result_arr) / float(len(result_arr))
    std = np.std(result_arr, dtype=np.float32, ddof=1)
    print("average  %s result" % str(name), mean)
    print("standard dev of %s" % str(name), std)
    print ("_________________________________________")
    return mean, std

def eval_model(model, features, labels, class_names, acc_arr=None):
    score = model.test(features, labels, class_names)
    acc = score['accuracy']
    acc = float("{0:.4f}".format(acc)) # take only 4 decimal place
    print('accuracy', acc)
    if acc_arr is not None:
        acc_arr.append(acc)
    return acc_arr

def get_train_val(train_features, train_labels, val_features, val_labels):
    return np.concatenate((train_features, val_features)), np.concatenate((train_labels, val_labels))

def format_best_param(best_param):
    for key, val in best_param.items():
        best_param[key] = np.asarray([val])
    print ('best param formatted: ', best_param)
    return best_param

def run_matlab_feature_extractor(knn, pyramid):
    if len(os.listdir(HANDCRAFTED_FEATURE_DIR) ) !=0: # if not empty, delete all file first
        shutil.rmtree(HANDCRAFTED_FEATURE_DIR, ignore_errors=True)
        if not os.path.exists(HANDCRAFTED_FEATURE_DIR): # recreate the folder if needed
            os.makedirs(HANDCRAFTED_FEATURE_DIR)

    eng = matlab.engine.start_matlab()
    eng.cd(MATLAB_DIR) # cd to dir
    eng.addpath(eng.genpath(MATLAB_DIR)) # add all dir and subdir to path
    if (DATASET == 'PAP_JPEG'):
        eng.extract_PAP(knn, pyramid, HANDCRAFTED_FEATURE_DIR, nargout=0)
    elif (DATASET == 'Hela_JPEG'):
        eng.extract_Hela(knn, pyramid, HANDCRAFTED_FEATURE_DIR, nargout=0)
    elif (DATASET == 'Hep_JPEG'):
        eng.extract_Hep(knn, pyramid, HANDCRAFTED_FEATURE_DIR, nargout=0)
    else:
        print("Dataset not match. cannot find matlab script")
    eng.exit() # exit when done
    return

def main(args):
    all_acc_train_s1 = []
    all_acc_train_s2 = []
    all_acc_train_2s = []
    all_acc_val_s1 = [] # all accuracy CNN
    all_acc_val_s2 = []
    all_acc_val_2s = []
    all_acc_test_s1 = []
    all_acc_test_s2= []
    all_acc_test_2s = []

    all_acc_train_val_s1 = []
    all_acc_train_val_s2 = []
    all_acc_train_val_2s = []
    all_acc_final_test_s1 = []
    all_acc_final_test_s2 = []
    all_acc_final_test_2s = []

    T = gen_threshold(T_MIN, T_MAX, T_STEP)
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    print('args: ', args)
    print(args.knn, [float(item) for item in args.pyramid.split(',')])
    run_matlab_feature_extractor(args.knn, [float(item) for item in args.pyramid.split(',')])

    for i in range (30):
        print ("Train model ith = %s/" % str(i+1), str(30))
        dataset = MyDataset(directory=IMAGE_DIR, test_size=0.2, val_size=0.25) #0.2 0.25
        train_files, train_labels, train_label_names, \
        val_files, val_labels, val_label_names, \
        test_files, test_labels, test_label_names, class_names = dataset.get_data()

        params_grid_1 = gen_grid(HYPER_PARAMS_1)
        params_grid_2 = gen_grid(HYPER_PARAMS_2)

        train_s1_features, val_s1_features, test_s1_features = get_handcrafted_features(
            train_files, train_labels, train_label_names,
            val_files, val_labels, val_label_names,
            test_files, test_labels, test_label_names)

        train_s2_features, val_s2_features, test_s2_features =get_CNN_features(
            train_files, train_labels, train_label_names,
            val_files, val_labels, val_label_names,
            test_files, test_labels, test_label_names)

        # now train stage 1
        cls1 = SVM_CLASSIFIER(params_grid_1, CLASSIFIER_1, OUT_MODEL1)
        cls1.prepare_model()
        best_params_cls1= cls1.train(train_s1_features, train_labels)
        best_params_cls1 = format_best_param(best_params_cls1)

        print("Finish train stage 1")

        print ("Now eval stage 1 on train set")
        all_acc_train_s1 = eval_model(cls1, train_s1_features, train_labels, class_names, all_acc_train_s1)
        print("Now eval stage 1 on val set")
        all_acc_val_s1= eval_model(cls1, val_s1_features, val_labels, class_names, all_acc_val_s1)
        print("Now eval stage 1 on test set")
        all_acc_test_s1= eval_model(cls1, test_s1_features, test_labels, class_names, all_acc_test_s1)
        print("---------------------")

        # now train stage 2
        cls2 = SVM_CLASSIFIER(params_grid_1 , CLASSIFIER_2, OUT_MODEL2)
        cls2.prepare_model()
        best_params_cls2 = cls2.train(train_s2_features, train_labels)
        best_params_cls2 = format_best_param(best_params_cls2)
        print("Finish train stage 2")


        print("Now eval stage 2 on train set")
        all_acc_train_s2 = eval_model(cls2, train_s2_features, train_labels, class_names, all_acc_train_s2)
        print("Now eval stage 2 on val set")
        all_acc_val_s2 = eval_model(cls2, val_s2_features, val_labels, class_names, all_acc_val_s2)
        print("Now eval stage 2 on test set")
        all_acc_test_s2= eval_model(cls2, test_s2_features, test_labels, class_names, all_acc_test_s2)
        print("---------------------")

        ###################################################
        # now train rejection rate
        cls1.get_centroids(train_s1_features, train_labels, dataset.categories)
        print("Now eval 2 stages on val set: ")
        t, acc_val_2_stage = find_best_t(cls1, cls2, dataset, val_s1_features, val_s2_features, val_labels, class_names, T)
        print ('The best t, val acc is ', t, acc_val_2_stage)
        all_acc_val_2s.append(acc_val_2_stage)

        print("Now eval 2 stages on train set: ")
        train_2_stage = get_2_stage_performance(cls1, cls2, dataset, train_s1_features,
                                               train_s2_features, train_labels, class_names, t)
        acc_train_2_stage = float("{0:.4f}".format(train_2_stage['accuracy'])) # take only 4 decimal place
        all_acc_train_2s.append(acc_train_2_stage)

        print("Now eval 2 stages on test set: ")
        test_2_stage = get_2_stage_performance(cls1, cls2, dataset, test_s1_features,
                                               test_s2_features, test_labels, class_names, t)
        acc_test_2_stage = float("{0:.4f}".format(test_2_stage['accuracy']))
        all_acc_test_2s.append(acc_test_2_stage)

        ##################################################
        # now retrain the model with train+val
        print("________________________________________")
        print("Now retrain both stages on train+val")
        train_val_s1_features, train_val_s1_labels = get_train_val(train_s1_features, train_labels, val_s1_features, val_labels)
        train_val_s2_features, train_val_s2_labels = get_train_val(train_s2_features, train_labels, val_s2_features, val_labels)
        print('train_val s1_features shape: ', train_val_s1_features.shape)
        print('train_val s2_features shape: ', train_val_s2_features.shape)
        print('train_val_labels shape: ', train_val_s2_labels.shape)

        # prepare models
        cls1 = SVM_CLASSIFIER(best_params_cls1, CLASSIFIER_1, OUT_MODEL1)
        cls1.prepare_model()
        cls2 = SVM_CLASSIFIER(best_params_cls2, CLASSIFIER_2, OUT_MODEL2)
        cls2.prepare_model()

        # train models
        _ = cls1.train(train_val_s1_features, train_val_s1_labels)
        print('Finish retraining stage 1 with train+val')
        _ = cls2.train(train_val_s2_features, train_val_s2_labels)
        print('Finish retraining stage 2 with train+val')
        cls1.get_centroids(train_val_s1_features, train_val_s1_labels, dataset.categories)

        # eval on train+val
        print("---------------------")
        print('Now eval stage 1 on train+val set')
        all_acc_train_val_s1 = eval_model(cls1, train_val_s1_features, train_val_s1_labels, class_names, all_acc_train_val_s1)
        print('Now eval stage 2 on train+val set')
        all_acc_train_val_s2 = eval_model(cls2, train_val_s2_features, train_val_s2_labels, class_names, all_acc_train_val_s2)
        print('Now eval 2 stage on train+val set')
        acc_train_val_2s = get_2_stage_performance(cls1, cls2, dataset, train_val_s1_features,
                                               train_val_s2_features, train_val_s1_labels, class_names, t)
        all_acc_train_val_2s.append(float("{0:.4f}".format(acc_train_val_2s['accuracy'])))

        print("---------------------")
        print('Now eval stage 1 on test set')
        all_acc_final_test_s1= eval_model(cls1, test_s1_features, test_labels, class_names,
                                          all_acc_final_test_s1)
        print('Now eval stage 2 on test set')
        all_acc_final_test_s2= eval_model(cls2, test_s2_features, test_labels, class_names,
                                          all_acc_final_test_s2)
        print('Now eval 2 stage on test set')
        acc_final_test_2s = get_2_stage_performance(cls1, cls2, dataset, test_s1_features,
                                                   test_s2_features, test_labels, class_names, t)
        all_acc_final_test_2s.append(float("{0:.4f}".format(acc_final_test_2s['accuracy'])))


    cal_mean_and_std(all_acc_train_s1, "train_stage_1")
    cal_mean_and_std(all_acc_train_s2, "train_stage_2")
    cal_mean_and_std(all_acc_train_2s, "train_2_stage")

    cal_mean_and_std(all_acc_val_s1, "val_stage_1")
    cal_mean_and_std(all_acc_val_s2, "val_stage_2")
    cal_mean_and_std(all_acc_val_2s, "val_2_stage")

    cal_mean_and_std(all_acc_test_s1, "test_stage_1")
    cal_mean_and_std(all_acc_test_s2, "test_stage_2")
    cal_mean_and_std(all_acc_test_2s, "test_2_stage")

    cal_mean_and_std(all_acc_train_val_s1, "train_val_stage_1")
    cal_mean_and_std(all_acc_train_val_s2, "train_val_stage_2")
    cal_mean_and_std(all_acc_train_val_2s, "train_val_2_stage")

    cal_mean_and_std(all_acc_final_test_s1, "final_test_stage_1")
    cal_mean_and_std(all_acc_final_test_s2,"final_test_stage_2")
    cal_mean_and_std(all_acc_final_test_2s, "final_test_2_stage")


def test():
    # dataset = MyDataset(directory=IMAGE_DIR, test_size=0.2, val_size=0.25)  # 0.2 0.25
    # train_files, train_labels, train_label_names, \
    # val_files, val_labels, val_label_names, \
    # test_files, test_labels, test_label_names, class_names = dataset.get_data()
    #
    # train_s1_features, val_s1_features, test_s1_features = get_CNN_features(
    #     train_files, train_labels, train_label_names,
    #     val_files, val_labels, val_label_names,
    #     test_files, test_labels, test_label_names)
    #
    # train_s2_features, val_s2_features, test_s2_features = get_handcrafted_features(
    #     train_files, train_labels, train_label_names,
    #     val_files, val_labels, val_label_names,
    #     test_files, test_labels, test_label_names)
    # train_val_s1_features, train_val_s1_labels = get_train_val(train_s1_features, train_labels, val_s1_features,
    #                                                            val_labels)
    # train_val_s2_features, train_val_s2_labels = get_train_val(train_s2_features, train_labels, val_s2_features,
    #                                                            val_labels)
    # print('train_val s1_features shape: ', train_val_s1_features.shape)
    # print('train_val s2_features shape: ', train_val_s2_features.shape)
    # print('train_val_labels shape: ', train_val_s2_labels.shape)

    # gen_threshold(T_MIN, T_MAX, T_STEP)
    run_matlab_feature_extractor(9, [1.0,2.0,4.0])
    # print(feature_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--knn', type=int)
    parser.add_argument('--pyramid', type=str)
    args = parser.parse_args()
    main(args)