from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

import load_CNN_features
import seaborn as sns
sns.set()

# PARAM_GRID = {'svc__C': [1, 5, 10, 50],
#               'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
# CLASSIFIER  = svm.SVC(kernel='rbf', class_weight='balanced')

PARAM_GRID = {'linearsvc__C': [1,5,10,50]}
CLASSIFIER = svm.LinearSVC()

FEATURE_DIR = '/mnt/6B7855B538947C4E/Dataset/features/off_the_shelf'
OUT_MODEL = '/mnt/6B7855B538947C4E/handcraft_models/filename.pkl'

def prepare_train_test(dataset, test_size):
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.data, dataset.labels,
                                                    random_state=42, test_size=test_size)
    print(Xtrain.shape)
    return Xtrain, Xtest, ytrain, ytest

def get_model(classifier):
    model = make_pipeline(classifier)
    return model


def grid_search(model, param_grid):
    grid = GridSearchCV(model, param_grid)
    return grid

def cal_DS(trained_model, x, category):
    y = trained_model.decision_function(x)[category]
    w_norm = np.linalg.norm(trained_model.coef_[category])
    dist = y / w_norm
    ds = 1/(1 + np.math.exp(-10 * dist))
    return ds

def train(grid, Xtrain, ytrain):

    grid.fit(Xtrain, ytrain)
    print(grid.best_params_)

    trained_model = grid.best_estimator_

    return trained_model

def test(trained_model, dataset, Xtest, ytest):
    yfit = trained_model.predict(Xtest)
    print(classification_report(ytest, yfit,
                                target_names=dataset.label_names))


    # mat = confusion_matrix(ytest, yfit)
    # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
    #             xticklabels=dataset.label_names,
    #             yticklabels=dataset.label_names)
    # plt.xlabel('true label')
    # plt.ylabel('predicted label')
    # plt.show()

def save(trained_model):
    joblib.dump(trained_model, OUT_MODEL)

def load():
    return joblib.load(OUT_MODEL)

def main():
    dataset = load_CNN_features.get_dataset(FEATURE_DIR)
    print(dataset.data.shape)
    print(dataset.label_names)

    Xtrain, Xtest, ytrain, ytest = prepare_train_test(dataset, test_size=0.2)
    # model = get_model(CLASSIFIER)
    # grid = grid_search(model, PARAM_GRID)

    # trained_model = train(grid, Xtrain, ytrain)
    # save(trained_model)
    # test(trained_model, dataset, Xtest, ytest)
    trained_model = load()
    test_x = Xtest[0]
    test_y = ytest[0]
    print (test_x, test_y)
    ds = cal_DS(trained_model._final_estimator, Xtest, 0)


if __name__ == '__main__':
    main()







