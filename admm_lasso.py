import numpy as np
from numpy.linalg import inv 
import pandas as pd
from sklearn import linear_model



class Admm_for_the_lasso(object):
    def __init__(self, Lambda=1.0, rho=1.0, mu=0, max_iter=100):
        # initial parameters
        self.Lambda = Lambda
        self.rho = rho
        self.mu = 0

        # train iter
        self.max_iter = max_iter

    def soft_threshold(self, x):
        threshold = self.Lambda / self.rho

        ones = np.ones(x.shape)
        positive = (x >= threshold) * x - threshold * ones * (x >= threshold)
        negative = (x <= -threshold) * x + threshold * ones * (x <= -threshold)
        return positive + negative

    def train(self, X, y):
        self.N, self.dim = X.shape
        beta = np.random.rand(self.dim)
        theta = beta.copy()
        
        for t in range(self.max_iter):
            beta = np.dot(
                            inv(np.dot(X.T, X) + self.rho * np.eye(self.dim)),
                            np.dot(X.T, y) + self.rho * theta - self.mu
                   )
            theta = self.soft_threshold(beta + self.mu/self.rho)
            self.mu += self.rho * (beta - theta)

        self.coef_ = beta
        
        

def main():
    df = pd.read_csv("boston.csv", index_col=0)
    y = df.iloc[:, 13].values
    df = (df - df.mean())/df.std()

    # using scikit-learn
    skl_model = linear_model.Lasso(alpha=1.0, max_iter=100)
    skl_model.fit(df, y)
    print("lasso by sklearn")
    print(skl_model.coef_)
    print()
    
    # using my ADMM class
    my_model = Admm_for_the_lasso(max_iter=100)
    my_model.train(df, y)
    print("lasso by my ADMM class")
    print(my_model.coef_)


if __name__ == '__main__':
    main()
