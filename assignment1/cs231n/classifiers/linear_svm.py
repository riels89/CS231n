import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        counter = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                counter += 1

        dW[:, y[i]] += counter * -X[i]

            # Right now the loss is a sum over all training examples, but we want it
            # to be an average instead so we divide by num_train.

    loss /= num_train
    dW /= num_train
    dW += reg * W
# Add regularization to the loss.
    loss += reg * np.sum(W * W)

#############################################################################
# TODO:                                                                     #
# Compute the gradient of the loss function and store it dW.                #
# Rather that first computing the loss and then computing the derivative,   #
# it may be simpler to compute the derivative at the same time that the     #
# loss is being computed. As a result you may need to modify some of the    #
# code above to compute the gradient.                                       #
#############################################################################

    return loss, dW

def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # shape of scores is (500, 10)\
    num_train = X.shape[0]
    dW = np.zeros(W.shape)

    scores = X.dot(W)

    correct_class_scores = scores[np.arange(num_train), y]
    margins = np.maximum(0, scores.T - correct_class_scores.T + 1.0)
    margins[y, np.arange(num_train)] = 0
    loss = np.sum(margins)

    loss /= num_train  # get mean
    loss += 0.5 * reg * np.sum(W * W)
    #############################################################################
    #                             END OF YOUR CODE                              #
    # #############################################################################

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    #  to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    # X = (500, 3073)
    # margins = (10, 500)
    # dW = (3073, 10)
    # incorrect counts = (500,)
    # y = (500,)

    # Semi-vectorized version. It's not any faster than the loop version.
    # num_classes = W.shape[1]
    # incorrect_counts = np.sum(margins > 0, axis=1)
    # for k in xrange(num_classes):
    #   # use the indices of the margin array as a mask.
    #   wj = np.sum(X[margins[:, k] > 0], axis=0)
    #   wy = np.sum(-incorrect_counts[y == k][:, np.newaxis] * X[y == k], axis=0)
    #   dW[:, k] = wj + wy

    # first attempt which got halfway and I gave up on this method
    # dW = np.zeros((W.shape[0], W.shape[1]))  # initialize the gradient as zero
    # num_classes = W.shape[1]
    #
    # incorrect_counts = np.sum(margins > 0, axis=0)
    # dwy = np.zeros((num_train, num_classes))
    # dwy[np.arange(500), y] = -incorrect_counts
    # dwy = dwy.sum(axis=0) * X.sum(axis=0)[:, np.newaxis]
    #
    # dwj = margins.T[np.arange(500), np.delete(np.arange(10), y, axis=1)]
    # print(dwj.shape)

    dL = (margins > 0).T * np.ones(margins.T.shape)
    incorrect_counts = np.sum(dL, axis=1)
    dL[np.arange(num_train), y] = -incorrect_counts
    dW = X.T.dot(dL)

    dW /= num_train
    dW += reg * W
    return loss, dW
