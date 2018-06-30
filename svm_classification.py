from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


import load_CNN_features
import seaborn as sns
sns.set()

param_grid = {'svc__C': [1, 5, 10, 50],
              'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
# param_grid = {'svc__C': [1,5,10,50]}

def prepare_train_test(dataset, test_size):
    Xtrain, Xtest, ytrain, ytest = train_test_split(dataset.data, dataset.labels,
                                                    random_state=42, test_size=test_size)
    print(Xtrain.shape)
    return Xtrain, Xtest, ytrain, ytest

def get_model():
    svc = svm.SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(svc)
    return model


def grid_search(model, param_grid):
    grid = GridSearchCV(model, param_grid)
    return grid

def train(grid, Xtrain, ytrain):

    grid.fit(Xtrain, ytrain)
    print(grid.best_params_)

    trained_model = grid.best_estimator_

    return trained_model

def test(trained_model, dataset, Xtest, ytest):
    yfit = trained_model.predict(Xtest)
    print(classification_report(ytest, yfit,
                                target_names=dataset.label_names))


    mat = confusion_matrix(ytest, yfit)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=dataset.label_names,
                yticklabels=dataset.label_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()


def main():
    dataset = load_CNN_features.get_dataset('/tmp/bottleneck')
    print(dataset.data.shape)
    print(dataset.label_names)

    Xtrain, Xtest, ytrain, ytest = prepare_train_test(dataset, test_size=0.2)
    model = get_model()
    grid = grid_search(model, param_grid)

    trained_model = train(grid, Xtrain, ytrain)
    test(trained_model, dataset, Xtest, ytest)


if __name__ == '__main__':
    main()







