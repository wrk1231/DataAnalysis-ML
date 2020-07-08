class LR(object):
        
    def __init__(X, y, learning_rate=0.03, iterations=1500, epsilon = 1e-5):
        ## Training Data
        self.X = X
        self.y = y

        ## Hyperparameters
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.iterations = iterations

        ## initial parameters
        self.m = len(y)

        self.X = np.hstack((np.ones((m,1)), self.X))
        self.n = np.size(self.X,1)
        self.params = np.zeros((self.n,1))
        

    def sigmoid(self, t):
        return 1 / (1 + np.exp(-t))

    def loss_function(self, theta):

        m = len(self.y)
        h = self.sigmoid(self.X @ theta)

        ## Cost function based on our mathematical expression
        cost = (1/m)*(((-y).T @ np.log(h + self.epsilon))-((1-y).T @ np.log(1-h + self.epsilon)))
        
        return cost

    def gradient_descent(self):
        m = len(y)
        cost_history = np.zeros((self.iterations,1))

        for i in range(self.iterations):
            self.params = self.params - (self.learning_rate/m) * (X.T @ (self.sigmoid(self.X @ self.params) - self.y)) 
            cost_history[i] = loss_function(self.X, self.y, self.params)

        return (cost_history, params)

    def predict(self, new_X):
        return np.round(self.sigmoid(new_X @ self.params))