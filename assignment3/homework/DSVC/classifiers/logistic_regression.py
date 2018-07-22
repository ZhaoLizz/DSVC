# coding=utf-8
import numpy as np
import random
from pprint import pprint


class LogisticRegression(object):

    def sigmoid(self, t):
        return 1. / (1. + np.exp(-t))

    def __init__(self):
        self.w = None
        self.ws = None

    def loss(self, X_batch, y_batch, classify_i):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - classify_object: the number you want to classify by one-vs-all

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        if (classify_i is None):
            w = self.w
            x_dot_w = np.dot(X_batch, w)
            exp_wx = np.exp(x_dot_w)
            y_predict = exp_wx / (1 + exp_wx)
            loss = np.sum(y_batch * np.log(y_predict) + (1 - y_batch) * np.log(1 - y_predict))
            loss = -loss
            gradient = X_batch.T.dot(y_predict - y_batch) / len(X_batch)
        else:
            ws_i = self.ws[classify_i]  # (n, )
            x_dot_w = np.dot(X_batch, ws_i)  # (m,1)
            exp_wx = np.exp(x_dot_w)
            y_predict = exp_wx / (1 + exp_wx)
            loss = np.sum(y_batch * np.log(y_predict) + (1 - y_batch) * np.log(1 - y_predict))
            loss = -loss
            gradient = X_batch.T.dot(y_predict - y_batch) / len(X_batch)  # (n, )

        return loss, gradient
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

    def predict(self, X,one_vs_all=False):
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
        if (not one_vs_all):
            y_pred = np.zeros(X.shape[1])
            ###########################################################################
            # TODO:                                                                   #
            # Implement this method. Store the predicted labels in y_pred.            #
            ###########################################################################
            x_dot_w = np.dot(X, self.w)
            exp_wx = np.exp(x_dot_w)
            y_pred = exp_wx / (1 + exp_wx)
            y_pred = np.where(y_pred > 0.5, 1, 0)
            return y_pred
        else:
            # one vs all !!!(false dont konw why)!!!
            # x_dot_w = np.dot(X, self.ws.T)  # (m,10)
            # exp_wx = np.exp(x_dot_w)
            # y_pred = exp_wx / (1 + exp_wx)  # (m,10),10列代表对每个数值的预测概率
            # print("row y_pred", y_pred.shape)
            # y_pred = np.argmax(y_pred, axis=1)

            y_pred1 = []
            print('ws', self.ws.shape)
            for theta in self.ws:  # ws (10,n) ,w(1,n)
                logits = 1 / (1 + np.exp(X.dot(theta.T)))  # (m,1)
                y_pred1.append(logits)
            y_pred1 = np.array(y_pred1)  # (10,m)
            y_pred1 = np.argmax(y_pred1 , axis=0)
            print('y_pred1',y_pred1)
            return y_pred1


    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
                   batch_size=200, verbose=True, decay_rate=0.5):
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
        num_train, dim = X.shape
        if self.ws is None:
            self.ws = 0.001 * np.random.randn(10, dim)

        loss_historys = []

        for i in range(10):
            loss_history_i = []
            print("classify ", i, '-----------------------')
            y_trans = (i != y).astype(int)  # 正样本为0
            for it in range(num_iters):
                random_index = np.random.choice(num_train, batch_size)
                X_batch = X[random_index]
                y_batch = y_trans[random_index]

                # evaluate loss and gradient
                loss, grad = self.loss(X_batch, y_batch, classify_i=i)
                loss_history_i.append(loss)

                # learning_rate decay linearly
                # learning_rate = 1 / ((1 + decay_rate * it)) * learning_rate
                # learning_rate decay exponentially
                # learning_rate = (0.9 ** it) * learning_rate

                # update w
                self.ws[i] = self.ws[i] - learning_rate * grad

                if verbose and it % 100 == 0:
                    print('iteration %d / %d: loss %f' % (it, num_iters, loss))
            loss_historys.append(loss_history_i)
        return loss_historys

    # def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
    #                batch_size=200, verbose=True):
    #     """
    #     Train this linear classifier using stochastic gradient descent.
    #     Inputs:
    #     - X: A numpy array of shape (N, D) containing training data; there are N
    #      training samples each of dimension D.
    #     - y: A numpy array of shape (N,) containing training labels;
    #     - learning_rate: (float) learning rate for optimization.
    #     - num_iters: (integer) number of steps to take when optimizing
    #     - batch_size: (integer) number of training examples to use at each step.
    #     - verbose: (boolean) If true, print progress during optimization.
    #
    #     """
    #     num_train, dim = X.shape
    #
    #     loss_history = []
    #     thetas = np.zeros((10, dim))
    #     for i in range(10):
    #         self.w = 0.001 * np.random.randn(dim)
    #         self.RATE = learning_rate
    #         loss_list = []
    #         t = 0
    #         for it in range(num_iters):
    #             t += 1
    #             index = np.random.choice(num_train, batch_size)
    #             # 正类为0
    #             X_batch = X[index]
    #             y_batch = np.where(y == i, 0, 1)
    #             y_batch = y_batch[index]
    #             loss, grad = self.loss(X_batch, y_batch, i)
    #             loss_list.append(loss)
    #             rate = float(self.RATE / (self.RATE * t + 1))
    #             self.w = self.w - rate * grad.T
    #         thetas[i] = self.w
    #         loss_history.append(loss_list)
    #     self.ws = thetas
    #     self.w = None
    #     self.RATE = None
