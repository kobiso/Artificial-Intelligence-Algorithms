import sys
import classificationMethod
import numpy as np
import util
import math


class GaussianDiscriminantAnalysisClassifier(classificationMethod.ClassificationMethod):
  def __init__(self, legalLabels, type):
    self.legalLabels = legalLabels
    self.type = type
    self.model=''

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method
    """
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels)

  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels):
    
    """
    Fill in this function!
    trainingData : (N x D)-sized numpy array
    validationData : (M x D)-sized numpy array
    trainingLabels : N-sized list
    validationLabels : M-sized list
    - N : the number of training instances
    - M : the number of validation instances
    - D : the number of features (PCA was used for feature extraction)
    
    Train the classifier by estimating MLEs.
    Evaluate LDA and QDA respectively and select the model that gives
    higher accuracy on the validationData.
    """
    
    #print 'legalLabels', self.legalLabels #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #print 'trainingData', trainingData.shape
    #print 'trainingLabels', len(trainingLabels)
    #print 'validationData', validationData.shape
    #print 'validationLabels', len(validationLabels)

    #Calculate prior probabilities
    #np.zeros(): Return a new array of given shape and type, filled with zeros.    
    #[[0. 0.] [0. 0.].....[0. 0.]]   
    #Calculate the number of trainingLabels for each legalLabels
    prior = np.zeros((len(self.legalLabels),2))
    for label in trainingLabels:
        prior[label][0] += 1
    self.prior = prior
    #Calculate 'current trainingLabels/total num of Labels'
    for label in self.legalLabels: 
        prior[label][1] = float(prior[label][0])/len(trainingLabels)
    #print 'prior', prior
    #print 'prior shape', prior.shape
    
    #Calculate mean for each labels
    mean = np.zeros((len(self.legalLabels),len(trainingData[0])))
    for data in range(len(trainingData)):
        mean[trainingLabels[data]] += trainingData[data]        
    for label in self.legalLabels:
        mean[label] = mean[label]/prior[label,0]
    self.mean = mean
    #print 'mean', mean[0]
        
    #Calculate covariance for each label
    cov = np.zeros((len(self.legalLabels),len(trainingData[0]),len(trainingData[0])))
    for i in range(len(trainingData)):
        deviation = np.zeros(len(trainingData[:,0]))      
        deviation = trainingData[i,:]
        cov[trainingLabels[i]] += np.outer(deviation - mean[trainingLabels[i]],deviation - mean[trainingLabels[i]])
    self.cov = cov
    for label in self.legalLabels:
        self.cov[label] /= prior[label][0]
    #print 'cov', cov[0]

    #Calculate LDA covariance
    lda_cov = np.zeros((len(trainingData[0]),len(trainingData[0])))
    for i in range(len(trainingData)):
        lda_cov += np.outer(trainingData[i]-mean[trainingLabels[i]], trainingData[i]-mean[trainingLabels[i]]) 
    lda_cov = lda_cov/len(trainingData)
    self.lda_cov = lda_cov
    #self.lda_cov = np.cov(trainingData.T)
    #print 'lda_cov', lda_cov    
    
    # Validate QDA accuracy
    self.model='QDA'
    guesses = self.classify(validationData)
    correct = 0  
    for i, prediction in enumerate(guesses):
        if validationLabels[i] == prediction: correct += 1
    qda_accuracy = float(correct) / len(guesses)  
    print "QDA accuracy: %d correct out of 100 (%2.1f%%)" % (correct, 100*qda_accuracy)  
    
    # Validate LDA accuracy
    self.model='LDA'
    guesses = self.classify(validationData)
    correct = 0    
    for i, prediction in enumerate(guesses):
        if validationLabels[i] == prediction: correct += 1
    lda_accuracy = float(correct) / len(guesses) 
    print "LDA accuracy: %d correct out of 100 (%2.1f%%)" % (correct, 100*lda_accuracy)
    
    # Compare QDA accuracy and LDA accuracy, put higher model to analyze
    if qda_accuracy > lda_accuracy:
        self.model='QDA'
    else:
        self.model='LDA'
    
    #util.raiseNotDefined()

  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      logposterior = self.calculateLogJointProbabilities(datum)
      guesses.append(np.argmax(logposterior))
      self.posteriors.append(logposterior)

    return guesses
    
  def calculateLogJointProbabilities(self, datum):
    """
    datum: D-sized numpy array
    - D : the number of features (PCA was used for feature extraction)
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the list, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    """
    
    logJoint = [0 for c in self.legalLabels]
    
    # If the selected model is QDA, calculate log posteriors with QDA model
    if self.model=='QDA':
        for label in self.legalLabels:
            logJoint[label] += np.log(self.prior[label][1])
            logJoint[label] += np.inner(datum-self.mean[label], np.dot(datum - self.mean[label], np.linalg.inv(self.cov[label]))) * -0.5
            logJoint[label] += np.log(np.linalg.det(self.cov[label])) * -0.5
    
    # If the selected model is LDA, calculate log posteriors with LDA model
    elif self.model=='LDA':
        beta = np.zeros(len(datum))
        gamma = 0.0                
        for label in range(len(self.legalLabels)):
            beta = np.dot(self.mean[label], np.linalg.inv(self.lda_cov))
            gamma = -(0.5)*np.inner(self.mean[label], np.dot(self.mean[label],np.linalg.inv(self.lda_cov)))+np.log(self.prior[label][1])
            prob = np.dot(datum, beta.T) + gamma
            logJoint[label] += prob
            
    #util.raiseNotDefined()
    
    return logJoint