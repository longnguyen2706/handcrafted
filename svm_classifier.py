from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
        self.centroids = None

    def get_model(self):
        self.model = make_pipeline(self.classifier)
        return

    def grid_search(self):
        self.grid = GridSearchCV(self.model, self.param_grid)
        return

    def cal_DS_m(self, x, cat_m, a=10):
        y = self.trained_model.decision_function(x)[0]
        y= y[cat_m]
        w_norm = np.linalg.norm(self.trained_model._final_estimator.coef_[cat_m])
        dist = y / w_norm
        ds = 1 / (1 + np.math.exp(-a * dist))
        return ds

    def cal_Ds_norm(self, x, cat_m, cats):
        sum_Ds = 0
        Ds_m = None
        for cat in cats:
            Ds_i = self.cal_DS_m(x, cat)
            sum_Ds = sum_Ds +  Ds_i
            if (cat == cat_m):
                Ds_m = Ds_i
        return Ds_m/sum_Ds

    def cal_Dc_m(self, X, centroid):
        return np.sqrt(np.sum((X - centroid) ** 2, axis=1))[0] # sum row by row

    def cal_PS_m(self,x, cat_m, cats):
        sum_Dc=0
        sum_Dc_ex_m = 0
        for cat in cats:
            Dc = self.cal_Dc_m(x, self.centroids[cat])
            sum_Dc = sum_Dc + Dc
            if (cat is not cat_m):
                sum_Dc_ex_m = sum_Dc_ex_m + Dc
        return sum_Dc_ex_m/sum_Dc

    def cal_PS_norm(self, x, cat_m, cats):
        sum_Ps = 0
        Ps_m = None
        for cat in cats:
            Ps_i = self.cal_PS_m(x, cat, cats)
            sum_Ps = sum_Ps + Ps_i
            if (cat == cat_m):
                Ps_m = Ps_i
        return Ps_m/sum_Ps

    def cal_CS(self, x, cat_m, cats):
        Ds_norm = self.cal_Ds_norm(x, cat_m, cats)
        Ps_norm = self.cal_PS_norm(x, cat_m, cats)
        return (Ds_norm+Ps_norm)/2

    def get_cat_m(self, X_data, y_data, cat_m):
        X_cat = []
        y_cat = []
        for i, y in enumerate(y_data):
            if (y == cat_m):
                X_cat.append(X_data[i])
                y_cat.append(y_data[i])
        return np.asarray(X_cat), np.asarray(y_cat)

    def get_centroid_m(self, X, y, cat_m):
        X_cat, y_cat = self.get_cat_m(X, y, cat_m)
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(X_cat, y_cat)
        centroid = kmeans.cluster_centers_[0] # [0] to get the correct shape
        return centroid

    def get_centroids(self, X, y, cats):
        centroids = []
        for cat in cats:
            centroid = self.get_centroid_m(X, y, cat)
            centroids.append(centroid)
        self.centroids = centroids
        return

    #TODO: cal_DS_norm, cal_PS_m, cal_PS_norm

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










