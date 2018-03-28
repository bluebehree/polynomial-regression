"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
"""

# This code was adapted from course material by Jenna Wiens (UMichigan).

# python libraries
import os

# numpy libraries
import numpy as np

# matplotlib libraries
import matplotlib.pyplot as plt

# Added these libraries
import time

######################################################################
# classes
######################################################################

class Data :
    
    def __init__(self, X=None, y=None) :
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
    
    def load(self, filename) :
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname(__file__)
        f = os.path.join(dir, '..', 'data', filename)
        
        # load data
        with open(f, 'r') as fid :
            data = np.loadtxt(fid, delimiter=",")
        
        # separate features and labels
        self.X = data[:,:-1]
        self.y = data[:,-1]
    
    def plot(self, **kwargs) :
        """Plot data."""
        
        if 'color' not in kwargs :
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()

# wrapper functions around Data class
def load_data(filename) :
    data = Data()
    data.load(filename)
    return data

def plot_data(X, y, **kwargs) :
    data = Data(X, y)
    data.plot(**kwargs)


class PolynomialRegression() :
    
    def __init__(self, m=1, reg_param=0) :
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param
    
    
    def generate_polynomial_features(self, X) :
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """
        
        n,d = X.shape
        
        ### ========== TODO : START ========== ###
        # part b: modify to create matrix for simple linear model
        m = self.m_

        # Use Python's list comprehension, map, and lambda
        # We pass in the range iterable into our lambda function,
        # so it will calculate x^0, x^1,..., x^m for all the values in X
        # Map will return a list for each value in X, and combined it will
        # make a numpy array with dimensions (n, m+1)

        # part g: modify to create matrix for polynomial model
        # Need the x[0] because we are grabbing an element from a numpy array
        Phi = np.array([list(map(lambda n: pow(x[0], n), range(0, m+1))) for x in X])

        ### ========== TODO : END ========== ###
        return Phi
    
    
    def fit_GD(self, X, y, eta=None,
                eps=0, tmax=10000, verbose=False) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
            t+1     -- the total number of iterations that took palce
            time.time() - start_time -- the total amount of time it took for this method to complete
        """
        start_time = time.time()

        if self.lambda_ != 0 :
            raise Exception("GD with regularization not implemented")
        
        if verbose :
            plt.subplot(1, 2, 2)
            plt.xlabel('iteration')
            plt.ylabel(r'$J(\theta)$')
            plt.ion()
            plt.show()
        
        X = self.generate_polynomial_features(X) # map features
        n,d = X.shape
        self.coef_ = np.zeros(d)                 # coefficients
        err_list  = np.zeros((tmax,1))           # errors per iteration
        
        # GD loop
        for t in range(tmax) :
            ### ========== TODO : START ========== ###
            # part f: update step size
            # change the default eta in the function signature to 'eta=None'
            # and update the line below to your learning rate function
            if eta is None :
                eta = 1 / float(1+t+1)

            ### ========== TODO : END ========== ###
                
            ### ========== TODO : START ========== ###
            # part d: update theta (self.coef_) using one step of GD
            # hint: you can write simultaneously update all theta using vector math
            self.coef_ = self.coef_ - eta * (np.dot(np.dot(X.T, X), self.coef_ ) - np.dot(X.T, y))

            # track error
            # hint: you cannot use self.predict(...) to make the predictions
            y_pred = np.dot(self.coef_, X.T)
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)                
            ### ========== TODO : END ========== ###
            
            # stop?
            if t > 0 and abs(err_list[t] - err_list[t-1]) <= eps :
                break
            
            # debugging
            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                plt.subplot(1, 2, 1)
                plt.cla()
                plot_data(x, y)
                self.plot_regression()
                plt.subplot(1, 2, 2)
                plt.plot([t+1], [cost], 'bo')
                plt.suptitle('iteration: %d, cost: %f' % (t+1, cost))
                plt.draw()
                plt.pause(0.05) # pause for 0.05 sec

        print('number of iterations: %d' % (t+1))

        # t+1 is the number of iterations
        # time.time() - start_time is the total amount of time it took to complete this method
        return self, t+1, time.time() - start_time
    
    
    def fit(self, X, y, l2regularize=None) :
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization
                
        Returns
        --------------------        
            self    -- an instance of self
            time.time() - start_time -- the total amount of time it took for the computation to complete
        """
        start_time = time.time()
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part e: implement closed-form solution
        # hint: use np.dot(...) and np.linalg.pinv(...)
        #       be sure to update self.coef_ with your solution

        self.coef_ = (np.linalg.pinv(np.dot(X.T, X))).dot(X.T).dot(y)
        return self, time.time() - start_time
        ### ========== TODO : END ========== ###
    
    
    def predict(self, X) :
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None :
            raise Exception("Model not initialized. Perform a fit first.")
        
        X = self.generate_polynomial_features(X) # map features
        
        ### ========== TODO : START ========== ###
        # part c: predict y
        y = np.dot(self.coef_, X.T)

        ### ========== TODO : END ========== ###
        
        return y
    
    
    def cost(self, X, y) :
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(theta)
        """
        ### ========== TODO : START ========== ###
        # part d: compute J(theta)

        # Cost function of linear regression is:
        # J(theta) = sum( (y_n - h_theta(x_n))^2 )
        cost = np.sum((y - self.predict(X))**2)

        ### ========== TODO : END ========== ###    

        return cost
    
    def rms_error(self, X, y) :
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        ### ========== TODO : START ========== ###
        # part h: compute RMSE
        n = float(X.shape[0])

        # Error is E_rms = sqrt(J(theta) / N)
        error = np.sqrt(self.cost(X, y) / n)
        ### ========== TODO : END ========== ###
        return error
    
    
    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs) :
        """Plot regression line."""
        if 'color' not in kwargs :
            kwargs['color'] = 'r'
        if 'linestyle' not in kwargs :
            kwargs['linestyle'] = '-'
        
        X = np.reshape(np.linspace(0,1,n), (n,1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


######################################################################
# main
######################################################################

def main() :
    # load data
    train_data = load_data('regression_train.csv')
    test_data = load_data('regression_test.csv')
    
    ### ========== TODO : START ========== ###
    # part a: main code for visualizations
    # print('Visualizing data...')

    # Grab the X,y data from train_data
    train_X = train_data.X
    train_y = train_data.y

    # Grab the X,y data from test_data
    test_X = test_data.X
    test_y = test_data.y

    # Plots the training and testing data
    # plot_data(train_X, train_y)
    # plot_data(test_X, test_y)

    # print('Finished visualizing data!')
    
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts b-f: main code for linear regression

    print('Investigating linear regression...')

	# Part di.    
    model = PolynomialRegression()
    model.coef_ = np.zeros((1,2))
    print(model.cost(train_X, train_y))

    # Part d ii. and iii.
    print('Fitting with Gradient Descent...')
    etas = [0.0001, 0.001, 0.01, 0.0407]
    info = []

    for eta in etas:
    	model, num_iters, time = model.fit_GD(X=train_X, y=train_y, eta=eta, verbose=False)
    	info.append({'eta': eta, 'coefficient': model.coef_, 'num_iters': num_iters, 'cost': model.cost(train_X, train_y), 'time': time})

    print('')

    for item in info:
    	print('For eta ' + str(item['eta']))
    	print('Coefficient:')
    	print(item['coefficient'])
    	print('Number of iterations: ' + str(item['num_iters']))
    	print('Final value of objective function:')
    	print(item['cost'])
    	print('Total time elapsed:')
    	print(item['time'])
    	print('')

    # Part e

    # model, time = model.fit(X=train_X, y=train_y)
    # print('')
    # print('Closed-form linear regression information:')
    # print('Coefficients:')
    # print(model.coef_)
    # print('Final value of objective function:')
    # print(model.cost(train_X, train_y))
    # print('Total time elapsed:')
    # print(time)

    # print('')

    # Part f
    # print('Investigating part f learning rate for GD...')
    # model, num_iters, time = model.fit_GD(X=train_X, y=train_y)

    # print('Coefficients:')
    # print(model.coef_)
    # print('Number of iterations for part f GD: ' + str(num_iters))
    # print('Final value of objective function:')
    # print(model.cost(train_X, train_y))
    # print('Total time elapsed:')
    # print(time)

    # print('')

    # print('Finished investigating linear regression!')

    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # parts g-i: main code for polynomial regression
    # print('Investigating polynomial regression...')

    # train_rmse = {}
    # test_rmse = {}

    # for m in range(0, 11):
    # 	model = PolynomialRegression(m=m)
    # 	model.fit(train_X, train_y)
    # 	train_rmse[m] = model.rms_error(train_X, train_y)
    # 	test_rmse[m] = model.rms_error(test_X, test_y)

    # print('Plotting RMSE...')
    # plt.plot(train_rmse.keys(), train_rmse.values(), 'b', label='Training Data')
    # plt.plot(test_rmse.keys(), test_rmse.values(), 'r', label='Testing Data')
    # plt.xlabel('Model Complexity')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.show()

    # # Get minimum RMSE for testing data
    # best_m = min(test_rmse, key=test_rmse.get)
    # print('')
    # print('Polynomial with least RMSE: ' + str(best_m))
    # print('RMSE: ')
    # print(test_rmse[best_m])
    # print('')

    # # m = 8 has comparable RMSE to m =5
    # print('Polynomial m = 8 RMSE:')
    # print(test_rmse[8])
    # print('')

    # print('Finished investigating polynomial regression!')
    ### ========== TODO : END ========== ###

if __name__ == "__main__" :
    main()
