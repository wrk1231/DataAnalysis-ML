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
            self.ax.scatter(self.support_vectors[:,0], self.support_vectors[:,1], s = 100, facecolors='b')

    def plot_model_boundary(self):
        self.fig = plt.figure(figsize = (12,8))
        self.ax = plt.axes()

        if self.trained == True:
            self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='rainbow', alpha=0.5)
            self.ax.scatter(self.support_vectors[:,0], self.support_vectors[:,1], s = 100, facecolors='b')
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


    @staticmethod
    def plot_dataset(X, y , limits=None):
      
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "rs")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "k^")
        plt.xlabel(r"$x_1$", fontsize = 20)
        plt.ylabel(r"$x_2$", fontsize = 20, rotation=0)
        if limits!=None:
            plt.axis(limits)

    @staticmethod
    def plot_predictions(model, limits):

        x0s = np.linspace(limits[0], limits[1], 100)
        x1s = np.linspace(limits[2], limits[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X = np.c_[x0.ravel(), x1.ravel()]
        y_pred = model.predict(X).reshape(x0.shape)
        y_decision = model.decision_function(X).reshape(x0.shape)
        
        plt.contourf(x0, x1, y_pred, cmap="rainbow", alpha=0.2)
        plt.contourf(x0, x1, y_decision, cmap="rainbow", alpha=0.1)

    @staticmethod
    def plot_svc_decision_function(model, ax=None, plot_support=True):
        """Plot the decision function for a 2D SVC"""
        if ax is None:
            ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # create grid to evaluate model
        x = np.linspace(xlim[0], xlim[1], 30)
        y = np.linspace(ylim[0], ylim[1], 30)
        Y, X = np.meshgrid(y, x)
        xy = np.vstack([X.ravel(), Y.ravel()]).T
        P = model.decision_function(xy).reshape(X.shape)
        
        # plot decision boundary and margins
        ax.contour(X, Y, P, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
        
        # plot support vectors
        if plot_support:
            ax.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, linewidth=1, facecolors='none');
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


from sklearn.svm import LinearSVR

class   SVMRegression(object):

    def __init__(self, X, y, epsilon, **kwargs):

        self.X = X
        self.y = y
        self.epsilon = epsilon
        self.model = LinearSVR(epsilon=epsilon, **kwargs) 

    def train_model(self):
        self.model.fit(self.X, self.y)
        self.epsilon = self.model.epsilon
        self.y_pred = self.model.predict(self.X)

    def get_support_vectors(self):
        """
        Get the index of points which is off the street
        """
        self.if_off_margin = (np.abs(self.y - self.y_pred) >= self.epsilon)
        self.idx_support_ = np.argwhere(self.if_off_margin)
        return self.idx_support_

    def model_predict(self, x_new):
        return self.model.predict(x_new)

    def plot_svm_regression(self, axes):
        """
        Plot SVM Regression
        """
        x_new = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
        y_estimate = self.model.predict(x_new)

        plt.plot(x_new, y_estimate, "k-", linewidth=2, label="Prediction of y")
        plt.plot(x_new, y_estimate + self.epsilon, "r--", label="Upper Bound")
        plt.plot(x_new, y_estimate - self.epsilon, "g--", label="Lower Bound")
        
        plt.scatter(self.X[self.idx_support_], self.y[self.idx_support_], s=180, facecolors='#FFAAAA')
        plt.plot(self.X, self.y, "bo")
        plt.xlabel(r"$x_1$", fontsize=18)
        plt.ylabel(r"$y$", fontsize=18, rotation=0)
        plt.legend(loc="best", fontsize=18)
        plt.axis(axes)

