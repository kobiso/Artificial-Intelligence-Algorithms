# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 2016

@author: jphong
"""
import classificationMethod
import numpy as np
import util
import copy

class LogisticRegressionClassifier(classificationMethod.ClassificationMethod):
  def __init__(self, legalLabels, type, seed):
    self.legalLabels = legalLabels
    self.type = type
    self.learningRate = [0.01, 0.001, 0.0001]
    self.l2Regularize = [1.0, 0.1, 0.0]
    self.numpRng = np.random.RandomState(seed)
    self.initialWeightBound = None
    self.posteriors = []
    self.costs = []
    self.epoch = 1000
    self.correct = 0

    self.bestParam = None # You must fill in this variable in validateWeight

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method.
    Iterates several learning rates and regularization parameter to select the best parameters.

    Do not modify this method.
    """
    for lRate in self.learningRate:
      #curCosts = []
      for l2Reg in self.l2Regularize:
        curCosts = []
        self.initializeWeight(trainingData.shape[1], len(self.legalLabels))
        for i in xrange(self.epoch):
          cost, grad = self.calculateCostAndGradient(trainingData, trainingLabels)
          self.updateWeight(grad, lRate, l2Reg)
          curCosts.append(cost)
        #print 'curCost', curCosts
        self.validateWeight(validationData, validationLabels)
        self.costs.append(curCosts)

  def initializeWeight(self, featureCount, labelCount):
    """
    Initialize weights and bias with randomness.

    Do not modify this method.
    """
    if self.initialWeightBound is None:
      initBound = 1.0
    else:
      initBound = self.initialWeightBound
    self.W = self.numpRng.uniform(-initBound, initBound, (featureCount, labelCount))
    self.b = self.numpRng.uniform(-initBound, initBound, (labelCount, ))

  def calculateCostAndGradient(self, trainingData, trainingLabels):
    """
    Fill in this function!

    trainingData : (N x D)-sized numpy array
    trainingLabels : N-sized list
    - N : the number of training instances
    - D : the number of features (PCA was used for feature extraction)
    RETURN : (cost, grad) python tuple
    - cost: python float, negative log likelihood of training data
    - grad: gradient which will be used to update weights and bias (in updateWeight)

    Evaluate the negative log likelihood and its gradient based on training data.
    Gradient evaluted here will be used on updateWeight method.
    Note the type of weight matrix and bias vector:
    self.W : (D x C)-sized numpy array
    self.b : C-sized numpy array
    - D : the number of features (PCA was used for feature extraction)
    - C : the number of legal labels
    """

    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    """ Calculate u (Gradient descent version) (Softmax_alternated) """
    wxb = np.dot(trainingData, self.W) + self.b
    amax = [np.amax(wxb, axis=1)] * len(self.legalLabels)
    wxbm = wxb - np.array(amax).T
    numerator = np.exp(wxbm)
    denominator = [np.sum(numerator, axis=1)] * len(self.legalLabels)
    self.u = numerator / np.array(denominator).T

    """ Calculate negative log-likelihood (Log-softmax_alternated) """
    N, D = trainingData.shape
    yic = (np.arange(N), trainingLabels)
    cost = (-1) * np.sum(wxbm[yic] - np.log(np.sum(numerator, axis=1)))

    """ Calculate gradient of the parameters"""
    uy = copy.deepcopy(self.u)
    for i in range(len(trainingData)):
      uy[i][trainingLabels[i]] = uy[i][trainingLabels[i]] - 1

    self.bcE = [1] * len(trainingData)
    self.bcE = np.dot(self.bcE, uy)
    grad = np.dot(trainingData.T, uy)

    return cost, grad

  def updateWeight(self, grad, learningRate, l2Reg):
    """
    Fill in this function!
    grad : gradient which was evaluated in calculateCostAndGradient
    learningRate : python float, learning rate for gradient descent
    l2Reg: python float, L2 regularization parameter

    Update the logistic regression parameters using gradient descent.
    Update must include L2 regularization.
    Please note that bias parameter must not be regularized.
    """

    "*** YOUR CODE HERE ***"

    wcE = grad + l2Reg * self.W
    self.W = self.W - np.multiply(learningRate, wcE)
    self.b = self.b - np.multiply(learningRate, self.bcE)
    #util.raiseNotDefined()

  def validateWeight(self, validationData, validationLabels):
    """
    Fill in this function!

    validationData : (M x D)-sized numpy array
    validationLabels : M-sized list
    - M : the number of validation instances
    - D : the number of features (PCA was used for feature extraction)

    Choose the best parameters of logistic regression.
    Calculates the accuracy of the validation set to select the best parameters.
    """

    "*** YOUR CODE HERE ***"
    self.bestParam = (self.W, self.b)
    guesses = self.classify(validationData)
    cur_correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    #print 'cur_correct', cur_correct
    if self.correct < cur_correct:
      self.correct = cur_correct
      (self.bestWa, self.bestBa) = (self.W, self.b)
      self.bestParam = (self.bestWa, self.bestBa)
    else:
      self.bestParam = (self.bestWa, self.bestBa)

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    Do not modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      logposterior = self.calculateConditionalProbability(datum)
      guesses.append(np.argmax(logposterior))
      self.posteriors.append(logposterior)

    return guesses
    
  def calculateConditionalProbability(self, datum):
    """
    datum : D-sized numpy array
    - D : the number of features (PCA was used for feature extraction)
    RETURN : C-sized numpy array
    - C : the number of legal labels

    Returns the conditional probability p(y|x) to predict labels for the datum.
    Return value is NOT the log of probability, which means 
    sum of your calculation should be 1. (sum_y p(y|x) = 1)
    """
    
    bestW, bestb = self.bestParam # These are parameters used for calculating conditional probabilities

    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()

    """ Calculate Conditional Probability """
    wxb = np.dot(datum, bestW) + bestb
    numerator = np.exp(wxb - np.amax(wxb, axis=0))
    denominator = np.sum(numerator, axis=0)
    prob = numerator / denominator

    return prob
