import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class SVMClf(object):

    def __init__(self, X, y, C = 1, kernel='linear'):
        
        self.X = X ## X as data
        self.y = y ## y as target
        ## kernal and C are model specific in SVM
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel= self.kernel, C=self.C)

        self.lowerBound = np.min(X) - np.std(X)
        self.upperBound = np.min(X) + np.std(X)
        self.new_X = np.linspace(self.lowerBound, self.upperBound, 500)


        self.trained = False

    def train_model(self):
        self.model.fit(X,y)
        self.trained = True

    def get_parameters(self):
        if self.trained == True:
            self.support_vectors = self.model.support_vectors_
            self.w = self.model.coef_[0]
            self.b = self.model.intercept_[0]
            
            self.decision_boundary = -self.w[0]/self.w[1] * self.new_X - self.b/self.w[1]

            self.margin = 1/self.w[1]
            self.gutter_up   = self.decision_boundary + self.margin
            self.gutter_down = self.decision_boundary - self.margin

        else:
            pass;

    def plot_model(self):
        self.fig = plt.figure(figsize = (12,8))
        self.ax = plt.axes()

        if self.trained == True:
            self.ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', alpha=0.5)
            self.ax.scatter(self.support_vectors[:,0], self.support_vectors[:,1], s = 100, facecolors='k')