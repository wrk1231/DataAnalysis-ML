import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

class SVMClf(object):

    def __init__(X, y, C = 1, kernal='linear'):
        
        self.X = X ## X as data
        self.y = y ## y as target
        ## kernal and C are model specific in SVM
        self.kernal = kernal
        self.C = C
        self.model = SVC(kernel= self.kernal, C=self.C)

        self.lowerBound = np.min(X) - np.std(X)
        self.upperBound = np.min(X) + np.std(X)
        self.new_X = np.linspace(self.lowerBound, self.upperBound, 500)


    def train_model():
        self.model.fit(X,y)
    
    def get_parameters():
            self.w = self.model.coef_[0]
            self.b = self.model.intercept_[0]
            
            self.decision_boundary = -self.w[0]/self.w[1] * self.new_X - self.b/self.w[1]

            self.margin = 1/self.w[1]
            self.gutter_up   = self.decision_boundary + self.margin
            self.gutter_down = self.decision_boundary - self.margin