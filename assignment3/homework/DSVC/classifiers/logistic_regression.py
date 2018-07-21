# coding=utf-8
import numpy as np
import random
from pprint import pprint


class LogisticRegression(object):

    def sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def __init__(self):
        self.w = None

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        # me-----------my error: reshape y,w to (m,1) (n,1),not understand yet--------------------------------
        # 用矩阵的方法-------------------------
        # m, n = X_batch.shape
        # X = X_batch
        # y = y_batch.reshape(-1, 1)
        # weights = self.w.reshape
        # x_dot_w = np.dot(X, weights)  # (m,1)
        # exp_wx = np.exp(x_dot_w)
        # loss = y * x_dot_w - np.log(1 + exp_wx)
        # loss = (-1 / m) * loss.sum()
        # y_predict = exp_wx / (1.0 + exp_wx)  # (64,1)
        # gradient = X.T.dot(y_predict - y)
        # gradient = gradient / m

        # 用向量的方法-------------------------
        w = self.w
        x_dot_w = np.dot(X_batch,w)
        exp_wx = np.exp(x_dot_w)
        y_predict = exp_wx / (1 + exp_wx)
        loss = np.sum(y_batch * np.log(y_predict) + (1 - y_batch)* np.log(1 - y_predict)) / len(y_batch)
        loss = -loss
        gradient = X_batch.T.dot(y_predict - y_batch) / len(X_batch)

        return loss, gradient  # (weights.shape[1],1)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
              batch_size=200, verbose=True, decay_rate=0.95):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        v_dw = None
        for it in range(num_iters):
            X_batch = None
            y_batch = None
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            random_index = np.random.choice(num_train, batch_size)
            X_batch = X[random_index]
            y_batch = y[random_index]
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)
            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################

            # learning_rate decay linearly
            # learning_rate = 1 / ((1 + decay_rate * it)) * learning_rate
            # learning_rate decay exponentially
            # learning_rate = (0.9 ** it) * learning_rate

            # momentumn
            beta = 0.9
            if v_dw is None:
                v_dw = grad

            if (it < num_iters / 10):  # fix v_dw bias in early estimation
                v_dw = (beta * v_dw + (1 - beta) * grad) / (1 - beta ** it)
            else:
                v_dw = beta * v_dw + (1 - beta) * grad

            # update w
            # self.w = self.w - learning_rate * v_dw        # momentum
            self.w = self.w - learning_rate * grad


            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))


        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        x_dot_w = np.dot(X, self.w)
        exp_wx = np.exp(x_dot_w)
        y_pred = exp_wx / (1 + exp_wx)

        y_pred = np.where(y_pred > 0.5, 1, 0)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
