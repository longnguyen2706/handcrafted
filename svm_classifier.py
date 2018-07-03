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


class SVM_CLASSIFIER:
    # PARAM_GRID = {'svc__C': [1, 5, 10, 50],
    #               'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
    # CLASSIFIER  = svm.SVC(kernel='rbf', class_weight='balanced')

    # PARAM_GRID = {'linearsvc__C': [1, 5, 10, 50]}
    # CLASSIFIER = svm.LinearSVC()
    def __init__(self, param_grid, classifier, out_model):
        self.param_grid = param_grid
        self.classifier = classifier
        self.out_model = out_model
        self.grid = None
        self.model = None
        self.trained_model = None

    def get_model(self):
        self.model = make_pipeline(self.classifier)
        return

    def grid_search(self):
        self.grid = GridSearchCV(self.model, self.param_grid)
        return

    def cal_DS(trained_model, x, category):
        y = trained_model.decision_function(x)[category]
        w_norm = np.linalg.norm(trained_model.coef_[category])
        dist = y / w_norm
        ds = 1 / (1 + np.math.exp(-10 * dist))
        return ds

    def prepare_model(self):
        self.get_model()
        self.grid_search()

    def train(self, Xtrain, ytrain):
        self.grid.fit(Xtrain, ytrain)
        print(self.grid.best_params_)
        self.trained_model = self.grid.best_estimator_
        return

    def test(self, Xtest, ytest, label_names):
        yfit = self.trained_model.predict(Xtest)
        print(classification_report(ytest, yfit,
                                    target_names=label_names))


        # mat = confusion_matrix(ytest, yfit)
        # sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
        #             xticklabels=dataset.label_names,
        #             yticklabels=dataset.label_names)
        # plt.xlabel('true label')
        # plt.ylabel('predicted label')
        # plt.show()

    def save(self):
        joblib.dump(self.trained_model, self.out_model)
        return

    def load(self):
        return joblib.load(self.out_model)










