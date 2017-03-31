# -*- coding: utf-8 -*-

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    self.features = trainingData[0].keys() # this could be useful for your code later...
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"    

    self.counter = util.Counter()
    self.N = len(trainingData)

    "initialize counter"
    for label in self.legalLabels:
        self.counter[label] = 0 # corresponds to P(C)
        for feature in self.features:
            self.counter[(label, feature)] = [0, 0] # corresponds to [P(x_i = 0 | C), P(x_i = 1 | C)]

    "Start counting"
    for n in range(len(trainingData)):
        label = trainingLabels[n]
        self.counter[label] += 1
        for feature in self.features:
            binary_value = trainingData[n][feature]
            self.counter[(label, feature)][binary_value] += 1

    "Choose the best k"
    bestCorrect = 0
    bestK = kgrid[0]
    for k in kgrid:
        self.k = k # this value will be used during self.classify
        guesses = self.classify(validationData)
        correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
        if correct > bestCorrect:
            bestCorrect = correct
            bestK = k

    self.k = bestK

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.

    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses

  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    logJoint = util.Counter()

    "*** YOUR CODE HERE ***"
    for label in self.legalLabels:
        logJoint[label] = math.log(float(self.counter[label]) / self.N) # log P(label)
        for feature in self.features:
            numerator   = self.counter[(label, feature)][datum[feature]] + self.k # add smoothing parameter k here
            denominator = self.counter[(label, feature)][0] + self.counter[(label, feature)][1] + 2 * self.k # we assumed binary feature {0,1}

            logJoint[label] += math.log(float(numerator) / denominator) # log P(x_i | label)
    
    return logJoint
