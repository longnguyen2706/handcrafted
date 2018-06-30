from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print("xx: ", xx, "yy: ", yy)
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target


X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1) #create grid (x,y) from boundary of original data

model = svm.LinearSVC()
clf = model.fit(X, y)
ax = plt.gca() # create a figure to plot

plot_contours(ax, clf, xx, yy,
              cmap=plt.cm.coolwarm, alpha=0.8) #plot class separation line and coloring class region
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k') # plot each class with dif color (c=class_label)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
ax.set_xlabel('Sepal length')
ax.set_ylabel('Sepal width')
ax.set_xticks(())
ax.set_yticks(())

plt.show()


def find_distace(X, model):
    print(X)
    return model.decision_function(X)

D= find_distace(X, clf)
print('distance to hyperplane: ', D)



