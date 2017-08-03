# -*- coding: utf-8 -*-
import cPickle
from neuralNetwork import NeuralNetworkClassifier as TestClassifier
import numpy as np

EPSILON = 0.0001

def test(testFlag, scoreName = ""):
	ptStr = ""
	if testFlag:
		ptStr += "[ OK ] "
	else:
		ptStr += "[FAIL] "
	ptStr += scoreName
	print ptStr


def main():
	"""
	0: trainingData
	1: solution forward output | 0
	2: trainingLabel
	3: initialW
	4: initialb
	5: solutionW
	6: solutionb
	"""
	f = open('testData.np', 'rb')
	testData = cPickle.load(f)
	f.close()

	hwInst = TestClassifier(range(10), 'hwCls', 123)
	hwInst.initializeWeight(testData[0][0].shape[0], 10)
	hwInst.W = testData[0][3]
	hwInst.b = testData[0][4]
	netOut = hwInst.forwardPropagation(testData[0][0])
	if type(netOut) != np.ndarray:
		test(False, "hand-written, 450, forward, invalid return type")
	flag1 = (np.max(np.fabs(netOut - testData[0][1])) < EPSILON)
	test(flag1, "hand-written, 450, forward")
	hwInst.backwardPropagation(testData[0][1], testData[0][2], 0.02 / 450.)
	if type(hwInst.W) != list:
		test(False, "hand-written, 450, backward, invalid return type")
	if type(hwInst.b) != list:
		test(False, "hand-written, 450, backward, invalid return type")
	for i, w in enumerate(hwInst.W):
		if type(w) != np.ndarray:
			test(False, "hand-written, 450, backward, invalid return type")
		flag1 = (np.max(np.fabs(w - testData[0][5][i])) < EPSILON)
		test(flag1, "hand-written, 450, backward, W, " + str(i))
	for i, b in enumerate(hwInst.b):
		if type(b) != np.ndarray:
			test(False, "hand-written, 450, backward, invalid return type")
		flag1 = (np.max(np.fabs(b - testData[0][6][i])) < EPSILON)
		test(flag1, "hand-written, 450, backward, b, " + str(i))

	hwInst = TestClassifier(range(10), 'hwCls', 123)
	hwInst.initializeWeight(testData[1][0].shape[0], 10)
	hwInst.W = testData[1][3]
	hwInst.b = testData[1][4]
	hwInst.nLayer = len(hwInst.W)
	netOut = hwInst.forwardPropagation(testData[1][0])
	if type(netOut) != np.ndarray:
		test(False, "hand-written, 1500, forward, invalid return type")
	flag1 = (np.max(np.fabs(netOut - testData[1][1])) < EPSILON)
	test(flag1, "hand-written, 1500, forward")
	hwInst.backwardPropagation(testData[1][1], testData[1][2], 0.02 / 1500.)
	if type(hwInst.W) != list:
		test(False, "hand-written, 1500, backward, invalid return type")
	if type(hwInst.b) != list:
		test(False, "hand-written, 1500, backward, invalid return type")
	for i, w in enumerate(hwInst.W):
		if type(w) != np.ndarray:
			test(False, "hand-written, 1500, backward, invalid return type")
		flag1 = (np.max(np.fabs(w - testData[1][5][i])) < EPSILON)
		test(flag1, "hand-written, 1500, backward, W, " + str(i))
	for i, b in enumerate(hwInst.b):
		if type(b) != np.ndarray:
			test(False, "hand-written, 1500, backward, invalid return type")
		flag1 = (np.max(np.fabs(b - testData[1][6][i])) < EPSILON)
		test(flag1, "hand-written, 1500, backward, b, " + str(i))

	hwInst = TestClassifier(range(2), 'hwCls', 123)
	hwInst.initializeWeight(testData[2][0].shape[0], 2)
	hwInst.W = testData[2][3]
	hwInst.b = testData[2][4]
	hwInst.nLayer = len(hwInst.W)
	netOut = hwInst.forwardPropagation(testData[2][0])
	if type(netOut) != np.ndarray:
		test(False, "faces, 450, forward, invalid return type")
	flag1 = (np.max(np.fabs(netOut - testData[2][1])) < EPSILON)
	test(flag1, "faces, 450, forward")
	hwInst.backwardPropagation(testData[2][1], testData[2][2], 0.02 / 450.)
	if type(hwInst.W) != list:
		test(False, "faces, 450, backward, invalid return type")
	if type(hwInst.b) != list:
		test(False, "faces, 450, backward, invalid return type")
	for i, w in enumerate(hwInst.W):
		if type(w) != np.ndarray:
			test(False, "faces, 450, backward, invalid return type")
		flag1 = (np.max(np.fabs(w - testData[2][5][i])) < EPSILON)
		test(flag1, "faces, 450, backward, W, " + str(i))
	for i, b in enumerate(hwInst.b):
		if type(b) != np.ndarray:
			test(False, "faces, 450, backward, invalid return type")
		flag1 = (np.max(np.fabs(b - testData[2][6][i])) < EPSILON)
		test(flag1, "faces, 450, backward, b, " + str(i))

if __name__ == '__main__':
	main()