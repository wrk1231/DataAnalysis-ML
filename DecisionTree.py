from helpers_05_08 import visualize_tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs


import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from ipywidgets import interact


        
fig, ax = plt.subplots(1, 4, figsize=(16, 3))
fig.subplots_adjust(left=0.02, right=0.98, wspace=0.1)

X, y = make_blobs(n_samples=300, centers=4,
                  random_state=0, cluster_std=1.0)

for axi, depth in zip(ax, range(1, 5)):
    model = DecisionTreeClassifier(max_depth=depth)
    visualize_tree(model, X, y, ax=axi)
    axi.set_title('depth = {0}'.format(depth))



class DecisionTreeVisualization(object):

    def __init__(self, max_depth, X, y, boundaries=True):
        self.if_boundaries = boundaries ## if plot boundary
        self.max_depth =max_depth ## depth of the decision tree

        self.X = X
        self.y = y
        self.model = DecisionTreeClassifier(max_depth=self.max_depth)


    def visualize_tree(self,  X, y, boundaries=True,  ax=None):
        ax = ax or plt.gca()
        
        # Plot the training points
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=30, cmap='rainbow', clim=(self.y.min(), self.y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim() 

        self.model.fit(self.X, self.y)

        xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                            np.linspace(*ylim, num=200))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        n_classes = len(np.unique(y))
        Z = Z.reshape(xx.shape)
        contours = ax.contourf(xx, yy, Z, alpha=0.3,
                            levels=np.arange(n_classes + 1) - 0.5,
                            cmap='rainbow', clim=(self.y.min(), self.y.max()),
                            zorder=1)

        ax.set(xlim=xlim, ylim=ylim)
        
        # Plot the decision boundaries
        def plot_boundaries(i, xlim, ylim):
            if i >= 0:
                tree = self.model.tree_
            
                if tree.feature[i] == 0:
                    ax.plot([tree.threshold[i], tree.threshold[i]], ylim, '-k', zorder=2)
                    plot_boundaries(tree.children_left[i],
                                    [xlim[0], tree.threshold[i]], ylim)
                    plot_boundaries(tree.children_right[i],
                                    [tree.threshold[i], xlim[1]], ylim)
            
                elif tree.feature[i] == 1:
                    ax.plot(xlim, [tree.threshold[i], tree.threshold[i]], '-k', zorder=2)
                    plot_boundaries(tree.children_left[i], xlim,
                                    [ylim[0], tree.threshold[i]])
                    plot_boundaries(tree.children_right[i], xlim,
                                    [tree.threshold[i], ylim[1]])
                
        if boundaries:
            plot_boundaries(0, xlim, ylim)


    @staticmethod
    def plot_tree_interactive(X, y):
        def interactive_tree(depth=5):
            clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
            visualize_tree(clf, X, y)

        return interact(interactive_tree, depth=[1, 5])

    @staticmethod
    def randomized_tree_interactive(X, y):
        N = int(0.75 * X.shape[0])
        
        xlim = (X[:, 0].min(), X[:, 0].max())
        ylim = (X[:, 1].min(), X[:, 1].max())
        
        def fit_randomized_tree(random_state=0):
            clf = DecisionTreeClassifier(max_depth=15)
            i = np.arange(len(y))
            rng = np.random.RandomState(random_state)
            rng.shuffle(i)
            visualize_tree(clf, X[i[:N]], y[i[:N]], boundaries=False,
                        xlim=xlim, ylim=ylim)
        
        interact(fit_randomized_tree, random_state=[0, 100]);