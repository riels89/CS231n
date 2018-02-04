import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
# Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

#############################################################################
# TODO: Compute the softmax loss and its gradient using explicit loops.     #
# Store the loss in loss and the gradient in dW. If you are not careful     #
# here, it is easy to run into numeric instability. Don't forget the        #
# regularization!                                                           #
#############################################################################

    for i in xrange(num_train):
        scores = X[i].dot(W)
        f = scores - np.max(scores)

        p = np.exp(f) / np.sum(np.exp(f))
        loss += -np.log(p[y[i]])

        for j in range(num_classes):
            dW[:, j] += X[i] * (p[j] - (j == y[i]))


        # Right now the loss is a sum over all training examples, but we want it
        # to be an average instead so we divide by num_train.

    loss /= num_train
    dW /= num_train
    dW += reg * W
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
#############################################################################
#                          END OF YOUR CODE                                 #
#############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    f = scores - np.max(scores)
    correct_class_scores = [np.arange(num_train), y]

    p = np.exp(f) / np.sum(np.exp(f), axis=1)[:, np.newaxis]
    loss += -np.log(p[correct_class_scores])
    loss = np.sum(loss)

    j = np.zeros_like(scores)
    j[np.arange(num_train), y] = 1

    dW = np.dot(X.T, (p - j))
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    loss /= num_train
    dW /= num_train
    dW += reg * W
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    return loss, dW

