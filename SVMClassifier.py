import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC

class SVMClf(object):

    def __init__(self, X, y, linear=False,**kwargs):
        
        self.X = X ## X as data
        self.y = y ## y as target
        ## kernal and C are model specific in SVM
        self.linear = linear
        if self.linear == True:
            self.model = LinearSVC(**kwargs)
        else:
            self.model = SVC(**kwargs)

        self.lowerBound = np.min(X) - 0.2*np.std(X)
        self.upperBound = np.max(X) + 0.2*np.std(X)
        self.new_X = np.linspace(self.lowerBound, self.upperBound, 500)




    def train_model(self):
        self.model.fit(self.X, self.y)
        self.trained = True

    def get_parameters(self, model=None):



    
        self.support_vectors = self.model.support_vectors_
        self.w = self.model.coef_[0]
        self.b = self.model.intercept_[0]
        
        self.decision_boundary = -self.w[0]/self.w[1] * self.new_X - self.b/self.w[1]

        self.margin = 1/self.w[1]
        self.gutter_up   = self.decision_boundary + self.margin
        self.gutter_down = self.decision_boundary - self.margin



    def model_predict(self, new_data):

        return self.model.predict(new_data)

    def plot_model(self):
        self.fig = plt.figure(figsize = (12,8))
        self.ax = plt.axes()

        if self.trained == True:
            self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='rainbow', alpha=0.5)
            self.ax.scatter(self.support_vectors[:,0], self.support_vectors[:,1], s = 100, facecolors='k')

    def plot_model_boundary(self):
        self.fig = plt.figure(figsize = (12,8))
        self.ax = plt.axes()

        if self.trained == True:
            self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='rainbow', alpha=0.5)
            self.ax.scatter(self.support_vectors[:,0], self.support_vectors[:,1], s = 100, facecolors='k')
            self.ax.plot(self.new_X, self.decision_boundary, "r-.", linewidth = 2)
            self.ax.plot(self.new_X, self.gutter_up, "g--", linewidth = 2)
            self.ax.plot(self.new_X, self.gutter_down, "g--", linewidth = 2)


    @staticmethod
    def plot_given_model(input_model, xmin, xmax):
        w = input_model.coef_[0]
        b = input_model.intercept_[0]

        # At the decision boundary, w0*x0 + w1*x1 + b = 0
        # => x1 = -w0/w1 * x0 - b/w1
        x0 = np.linspace(xmin, xmax, 200)
        decision_boundary = -w[0]/w[1] * x0 - b/w[1]
        
        ## how to calculate SVM Margin?
        margin = 1/w[1]
        gutter_up = decision_boundary + margin
        gutter_down = decision_boundary - margin

        svs = input_model.support_vectors_
        plt.scatter(svs[:, 0], svs[:, 1], s=250, facecolors='#FFAAAA')
        plt.plot(x0, decision_boundary, "r-.", linewidth=2)
        plt.plot(x0, gutter_up, "k--", linewidth=2)
        plt.plot(x0, gutter_down, "k--", linewidth=2)