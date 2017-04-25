# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 2016

@author: bjlee
"""
import classificationMethod
import numpy as np
import scipy as sp
from scipy import linalg
import sys
import util

def softmax(X):
    e = np.exp(X - np.max(X))
    det = np.sum(e, axis=1)
    return (e.T / det).T

class gaussianProcessClassifier(classificationMethod.ClassificationMethod):
    def __init__(self, legalLabels, type, seed, data):
        self.legalLabels = legalLabels
        self.numpRng = np.random.RandomState(seed)
        self.numberofsamples = 1000  
        self.data = data
        
    def initializeHyp(self):
        """
        Initialize hyper-parameter appropriately
    
        Do not modify this method.
        """
        self.trainingShape = np.shape(self.trainingData)
        [n,d] = self.trainingShape
        c = len(self.legalLabels)
        if self.data == 'faces':
            noise = 250.0
        else:
            noise = 20.0
        self.hypSize = [c, 2]
        self.hyp = np.zeros(self.hypSize)
        for i in range(c):
            self.hyp[i,:] = np.array([np.log(noise)/2, np.log(noise/4)/2])
    

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        You may commentize or decommentize few lines here to change the behavior of the program
        """
        
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
        self.initializeHyp()
        
        """
        Decommentize the line below to check whether the implementation is correct:
        
        This method compares the gradient computed by method 'derivative_of_marginalLikelihood'
        
        with the finite difference of method 'marginalLikelihood'.
        
        If the final output of 'checkGradient' method is very small (less than 1e-4),
        
        your implementation on method 'marginalLikelihood' may be right.
        """
        #self.checkGradient(1e-4)
        
        """
        The three lines below optimizes the marginal likelihood with conjugate gradient algorithm. 
        
        Since I initialized hyper-parameters with appropriate values, the algorithm would do well without optimization;       
               
        decommentize theses lines if you are curious.
        
        Note: optimization with digit dataset may not work due to the numerical issue
        """
        #print "initial hyp: ", self.hyp.ravel()
        #print "initial negative approx. marginal likelihood: ", self.marginalLikelihood(self.hyp)
        #res = sp.optimize.minimize(self.marginalLikelihood, self.hyp, method='CG'\
        #    , jac=self.derivative_of_marginalLikelihood, options = {'maxiter':25})    
        #self.hyp = np.reshape(res.x, self.hypSize)
        #print res
        
    def calculateIntermediateValues(self, t, a, Kcs):
        """
        You should implement this method:
        
        Read README file.
        """
        [n,d] = self.trainingShape
        cLen = len(self.legalLabels)
        ############ Implement here

        # Compute 'pi' from a with eq(2)
        a_matrix = np.reshape(a, (n,cLen), order='F') # Be careful with the order of 'np.reshape' !
        amax = [np.amax(a_matrix, axis=1)] * cLen
        aMinusMax = a_matrix - np.array(amax).T
        numerator = np.exp(aMinusMax)
        denominator = [np.sum(numerator, axis=1)] * cLen
        pi = numerator / np.array(denominator).T
        pi_l = np.reshape(pi, n*cLen, order = 'F') # for np.diag(pi_l)
        pi_linear = np.reshape(pi, (n*cLen, 1), order = 'F')

        # Calculate big pi
        bigPi = np.diag(pi.T[0])
        for c in range(cLen-1):
          bigPi = np.vstack((bigPi, np.diag(pi.T[c+1])))

        K = self.block_diag(Kcs)
        D = np.diag(pi_l)
        logdet = 0.0
        Ec_sum=np.zeros((n,n))
        Ecs=[]
        for c in range(cLen):
          DcSqrt = np.sqrt(np.diag((pi.T)[c]))
          L = np.linalg.cholesky(np.identity(n) + np.dot(DcSqrt, Kcs[c]).dot(DcSqrt))

          #Ec = DcSqrt.dot(np.linalg.inv(L.T)).dot(np.linalg.inv(L).dot(DcSqrt))
          """
          MRE = np.linalg.solve(L, DcSqrt)
          MMRE = np.linalg.solve(L.T, MRE)
          Ec = DcSqrt.dot(MMRE)
          """
          Ec = DcSqrt.dot(np.linalg.solve(L.T, np.linalg.solve(L, DcSqrt)))

          Ecs.append(Ec)
          Ec_sum += Ec
          logdet = logdet + np.sum(np.log(np.diag(L)))

        M = np.linalg.cholesky(Ec_sum)
        E = self.block_diag(Ecs)
        logdet = logdet + np.sum(np.log(np.diag(M)))
        W = D - np.dot(bigPi, bigPi.T)
        c_now = np.dot(W, a) + t - pi_linear
        d = np.dot(E,K).dot(c_now)

        R = np.dot(np.linalg.inv(D), bigPi)
        #b = c_now - d + E.dot(R).dot(np.linalg.inv(M.T)).dot(np.linalg.inv(M).dot((R.T).dot(d)))
        """
        MRE = np.linalg.solve(M, R.T.dot(d))
        MMRE = np.linalg.solve(M.T, MRE)
        b = c_now - d + E.dot(R.dot(MMRE))
        """
        b = c_now - d + E.dot(R.dot(np.linalg.solve(M.T, np.linalg.solve(M, R.T.dot(d)))))

        pi = pi_linear
        
        ############ Implement here
        valuesForModes = [W, b, logdet, K]        
        valuesForDerivatives = [E, M, R, b, pi, K]
        valuesForPrediction = [pi, Ecs, M, R, K]
        return valuesForModes, valuesForDerivatives, valuesForPrediction
      
    def findMode(self, trainingData, trainingLabels, hyp):
        [n,d] = self.trainingShape
        c = len(self.legalLabels)
          
        Kcs = self.calculateCovariance(trainingData, hyp) # Compute Kc using covariance function, with X and hyp
        [t,_] = self.trainingLabels2t(trainingLabels)
        """
        You should implement this method:
        
        Read README file.
        """
        ############ Implement here
        a = np.zeros((n*c,1))

        oldObj = 0.0
        obj = 0.0
        while True:
          oldObj=obj
          [[_,b,logdet,K],_,_] = self.calculateIntermediateValues(t, a, Kcs)
          a = K.dot(b)
          a_matrix = np.reshape(a, (n,c), order='F')
          rSide = np.sum(np.log(np.exp(a_matrix).dot(np.ones([c,1]))))
          obj = -(0.5)*(b.T).dot(a) + (t.T).dot(a) - rSide
          if abs(obj - oldObj) < 1e-10 :
            break
        Z = obj - logdet
        
        ############ Implement here
        
        return a, Z
      
    def calculatePredictiveDistribution(self, datum, pi, Ecs, M, R, tc):
        """
        You should implement this method:
        
        Read README file.
        """
        ############ Implement here
        [n, d] = self.trainingShape
        cLen = len(self.legalLabels)
        datum = np.reshape(datum, (1, len(datum)))
        pi = np.reshape(pi, (n, cLen), order='F')

        k1=self.covARD(self.hyp[0, :], datum, self.trainingData)
        k2=self.covARD(self.hyp[0, :], datum, datum)
        for c in range(cLen):
          if c==0:
            continue
          k1 = np.vstack((k1, self.covARD(self.hyp[c, :], datum, self.trainingData)))
          k2 = np.vstack((k2, self.covARD(self.hyp[c, :], datum, datum)))
          np.vstack(self.covARD(self.hyp[c, :], datum, self.trainingData))

        Rcs = np.vsplit(R, cLen)

        mu = np.zeros((1,cLen))
        sigma = np.zeros((cLen,c+1))
        sig_stack=[]
        for c in range(cLen):
          mu[(0,c)]=((tc.T[c] - pi.T[c]).dot(k1[c]))
          f = Ecs[c].dot(k1[c])
          """
          MRE = np.linalg.solve(M, Rcs[c].T.dot(f))
          MMRE = np.linalg.solve(M.T, MRE)
          g = Ecs[c].dot(Rcs[c].dot(MMRE))
          """
          g = Ecs[c].dot(Rcs[c].dot(np.linalg.solve(M.T, np.linalg.solve(M, Rcs[c].T.dot(f)))))

          sig_stack.append(g)
          """
          for i in range(len(self.legalLabels)):
            sigma[(c,i)] = (g.T).dot(k1[i])
          """
          sigma[(c,c)] = sigma[(c,c)] + k2[c] - (f.T).dot(k1[c])
        sig_stack = np.array(sig_stack).dot(k1.T)
        sigma = sigma + sig_stack
        
        ############ Implement here
        samples = self.numpRng.multivariate_normal(mu.ravel(), sigma, self.numberofsamples)
        predict = softmax(samples)
        return np.mean(predict,0)
        
    def derivative_of_marginalLikelihood(self, hyp):
        """
        This method calculates the derivative of marginal likelihood.
        
        You may refer to this code to see what methods in numpy is useful
        
        while you are implementing other functions.
        
        Do not modify this method.
        """
        trainingData = self.trainingData
        trainingLabels = self.trainingLabels
        c = len(self.legalLabels)
        [n,d] = self.trainingShape
        hyp = np.reshape(hyp, self.hypSize)
        
        [mode,_] = self.findMode(trainingData, trainingLabels, hyp)
        [t,_] = self.trainingLabels2t(trainingLabels)
          
        Ks = self.calculateCovariance(trainingData, hyp)
        [_,[E, M, R, b, totpi, K],_] = self.calculateIntermediateValues(t, mode, Ks)
          
        MRE = np.linalg.solve(M,R.T.dot(E))
        MMRE = np.linalg.solve(M.T,MRE)
        KWinvinv = E-E.dot(R.dot(MMRE))
          
        KinvWinv = K-K.dot(KWinvinv.dot(K))
        partitioned_KinvWinv = np.transpose(np.array(np.split(np.array(np.split(KinvWinv, c)),c,2)),[2,3,1,0])
          
        s2 = np.zeros([n,c])
        for i in range(n):
            pi_n = softmax(np.reshape(mode,[c,n])[:,i:i+1].T).T
            pipj = pi_n.dot(pi_n.T)
            pi_3d = np.zeros([c,c,c])
            pi_3d[np.diag_indices(c,3)] = pi_n.ravel()
            pipjpk = np.tensordot(pi_n,np.reshape(pipj,(1,c,c)),(1,0))
            pipj_3d = np.zeros([c,c,c])
            pipj_3d[np.diag_indices(c)] = pipj
            W_3d = pi_3d + 2 * pipjpk - pipj_3d - np.transpose(pipj_3d,[2,1,0]) - np.transpose(pipj_3d,[1,2,0])
            s2[i,:] = -0.5*np.trace(partitioned_KinvWinv[i,i].dot(W_3d))
              
        b_rs = np.reshape(b, [c,n])
        dZ = np.zeros(hyp.shape)
        for j in range(2):
            cs = []
            zeroCs = [np.zeros([n,n]) for i in range(c)]
            for i in range(c):
                C = self.covARD(hyp[i,:],trainingData,None,j)
                dZ[i,j] = 0.5*b_rs[i,:].T.dot(C.dot(b_rs[i,:]))
                zeroCs[i] = C
                cs.append(self.block_diag(zeroCs))
                zeroCs[i] = np.zeros([n,n])
                
            for i in range(c):
                dd = cs[i].dot(t-totpi)
                s3 = dd - K.dot(KWinvinv.dot(dd))
                dZ[i,j] +=  - 0.5 * np.trace(KWinvinv.dot(cs[i])) + s2.T.ravel().dot(s3) # 
                  
        return -dZ.ravel()
      
    def marginalLikelihood(self, hyp):
        """
        Wrapper function for scipy.optimize:
                
        Do not modify this method.
        """
        trainingData = self.trainingData
        trainingLabels = self.trainingLabels
        hyp = np.reshape(hyp, self.hypSize)
          
        [_, Z] = self.findMode(trainingData, trainingLabels, hyp)
        return -Z
        
    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
    
        Do not modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        
        [mode,_] = self.findMode(self.trainingData, self.trainingLabels, self.hyp)
        Kcs = self.calculateCovariance(self.trainingData, self.hyp)
        [t,tc] = self.trainingLabels2t(self.trainingLabels)
        [_,_,[pi, Ecs, M, R, K]] = self.calculateIntermediateValues(t, mode, Kcs)
        
        for datum in testData:
          logposterior = self.calculatePredictiveDistribution(datum, pi, Ecs, M, R, tc)
          guesses.append(np.argmax(logposterior))
          self.posteriors.append(logposterior)
    
        print guesses
        return guesses
    
        
    def checkGradient(self, error):
        """
        Method to check whether the gradient is right by comparing with finite difference.
        
        Since I give you the right gradient function, you may use this to check
        
        whether the marginal likelihood implementation is right.
        """
        hyp = self.hyp
        c = len(self.legalLabels)
        [n,d] = self.trainingShape
        dh = np.zeros(hyp.shape)
        dZ = np.reshape(self.derivative_of_marginalLikelihood(hyp),self.hypSize)
        for i in range(c):
            for j in range(2):
                print (i,j)
                ehyp = np.copy(hyp)
                ehyp[i,j] += error
                Z2 = self.marginalLikelihood(ehyp)
                ehyp[i,j] -= error * 2
                Z3 = self.marginalLikelihood(ehyp)
                dh[i,j] = (Z2-Z3) / (2 * error)
        print 'dZ', dZ
        print 'dh', dh
        print 'inte.', (dh-dZ)/(dh+dZ)


    def covARD(self, hyp, x, z = None, i = None):
        """
         Squared Exponential covariance function with isotropic distance measure. The
         covariance function is parameterized as:
        
         k(x^p,x^q) = sf^2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2) 
        
         where the P matrix is ell^2 times the unit matrix and sf^2 is the signal
         variance. The hyperparameters are:
         hyp = [ log(ell)
                 log(sf)  ]

        """
        def sq_dist(a, b = None):
            [D, n] = np.shape(a)
            if b is None:
                mu = np.mean(a,1)
                a = (a.T - mu).T
                b = a
                m = n
            else:
                [d, m] = np.shape(b)
                if d != D:
                    print 'Error: column lengths must agree.'
                    sys.exit(1)
                mu = (m/(n+m))*np.mean(b,1) + (n/(n+m))*np.mean(a,1)
                a = (a.T - mu).T
                b = (b.T - mu).T
            return np.tile(np.sum(a*a, 0), [m, 1]).T + np.tile(np.sum(b*b, 0), [n, 1]) - 2 * a.T.dot(b)
            
        xeqz = z is None
        ell = np.exp(hyp[0])
        sf2 = np.exp(2 * hyp[1])
        if xeqz:
            K = sq_dist(x.T/ell)
        else:
            K = sq_dist(x.T/ell,z.T/ell)
        if i is not None:
            if i == 0:
                K = sf2 * np.exp(-K/2) * K
            elif i == 1:
                K = 2 * sf2 * np.exp(-K/2)
            else:
                print 'Unkown parameter!'
                sys.exit(1)
        else:
            K = sf2 * np.exp(-K/2)
        return K      
  
  

    def calculateCovariance(self, trainingData, hyp):
        Ks = []
        c = len(self.legalLabels)
        for i in range(c):
            Ks.append(self.covARD(hyp[i,:],trainingData))
        return Ks
  
    def trainingLabels2t(self, trainingLabels):
        t = []
        n = np.shape(trainingLabels)[0]
        c = len(self.legalLabels)
        for i in range(n):
            temp = np.zeros([c,1])
            temp[trainingLabels[i]] = 1
            t.append(temp)
        ttot = np.concatenate(t)
        tc = np.reshape(ttot,[n,c])
        ttot = np.reshape(tc.T,[n*c,1])
          
        return ttot, tc
      
    def block_diag(self, args):
        return sp.linalg.block_diag(*args)