"""ANNtf_algorithmFBANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf_main.py

# Usage:
see ANNtf_main.py

# Description:
ANNtf algorithm FBANN - define feedback artificial neural network

"""

import tensorflow as tf
import numpy as np
from ANNtf_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf_operations
import ANNtf_globalDefs
import math



feedbackConnections = True #add feedback connections (disable for benchmarking)
supportSkipLayers = True #fully connected skip layer network

if(feedbackConnections):
	numberOfFeedBackwardsIterations = 1	#5	#number of feed forward/back iterations	#requires calibration (dynamic?)
	additiveZmethodRecordBackwardFeedDelta = True	#this ensures that the backwards pass signal is always normalised (not cumulated across multiple iterations)	#TESTHIS
else:
	numberOfFeedBackwardsIterations = 0
	additiveZmethodRecordBackwardFeedDelta = False

normaliseFirstLayer = False
equaliseNumberExamplesPerClass = False

Wf = {}
Wb = {}
B = {}
Ztrace = {}
Atrace = {}
if(additiveZmethodRecordBackwardFeedDelta):
	ZtraceBackwardFeedDelta = {}


#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

learningRate = 0.0
batchSize = 0

highestLayer = 0	#numberOfLayers
if(feedbackConnections):
	lowestLayerWithIncomingBackwardsConnections = 1	#eg 0 or 1
	highestLayerWithIncomingBackwardsConnections = 0	#eg highestLayer-1 or highestLayer-2


def getNoisySampleGenerationNumSamples():
	return False, 0, 0
	
def defineTrainingParameters(dataset):

	global learningRate
	global forgetRate
	global batchSize
	
	learningRate = 0.001
	trainingSteps = 1000
	batchSize = 100		#1	#10	#100	#1000	#temporarily reduce batch size for visual debugging (array length) purposes)
	numEpochs = 10
	
	displayStep = 100
				
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global highestLayer
	global highestLayerWithIncomingBackwardsConnections
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet)

	highestLayer = numberOfLayers
	highestLayerWithIncomingBackwardsConnections = highestLayer-1

	print("highestLayer = ", highestLayer)
	
	return numberOfLayers


def defineNeuralNetworkParameters():

	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
	
		for l1 in range(1, highestLayer+1):

			if(supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))
						#print("Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, \"Wf\")].shape = ", Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")].shape)
				if(feedbackConnections):
					if((l1 <= highestLayerWithIncomingBackwardsConnections) and (l1 >= lowestLayerWithIncomingBackwardsConnections)):
						for l2 in range(l1+1, highestLayer+1):
							if(l2 > l1):
								Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")] = tf.Variable(randomNormal([n_h[l2], n_h[l1]]))
			else:
				Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")] = tf.Variable(randomNormal([n_h[l1-1], n_h[l1]]))
				if(feedbackConnections):
					if((l1 <= highestLayerWithIncomingBackwardsConnections) and (l1 >= lowestLayerWithIncomingBackwardsConnections)):
						Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")] = tf.Variable(randomNormal([n_h[l1+1], n_h[l1]]))
			
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(tf.zeros(n_h[l1]))
			#print("B[generateParameterNameNetwork(networkIndex, l1, \"B\")].shape = ", B[generateParameterNameNetwork(networkIndex, l1, "B")].shape)

			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			
			if(additiveZmethodRecordBackwardFeedDelta):
				ZtraceBackwardFeedDelta[generateParameterNameNetwork(networkIndex, l1, "ZtraceBackwardFeedDelta")] = tf.Variable(tf.zeros(n_h[l1], dtype=tf.dtypes.float32))


def resetTraceNeuralNetworkParametersFBANN():
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l1 in range(1, highestLayer+1):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
			if(additiveZmethodRecordBackwardFeedDelta):
				ZtraceBackwardFeedDelta[generateParameterNameNetwork(networkIndex, l1, "ZtraceBackwardFeedDelta")] = tf.Variable(tf.zeros(n_h[l1], dtype=tf.dtypes.float32))


def neuralNetworkPropagation(x, networkIndex=1):
	if(feedbackConnections):
		pred = neuralNetworkPropagationFBANN(x, networkIndex)
	else:
		pred = neuralNetworkPropagationANN(x, networkIndex)
	return pred
			
def neuralNetworkPropagationANN(x, networkIndex=1):
	
	#AprevLayer = x
	#for l in range(1, numberOfLayers+1):
	#	Z = tf.add(tf.matmul(AprevLayer, Wf[generateParameterNameNetwork(networkIndex, l, "Wf")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
	#	A = activationFunction(Z)	
	#	AprevLayer = A
	#return tf.nn.softmax(Z)
		
	ZhighestLayer, AhighestLayer = neuralNetworkPropagationANNfeedForward(x, networkIndex=networkIndex)
	return tf.nn.softmax(ZhighestLayer)

def neuralNetworkPropagationANNfeedForward(x, networkIndex=1):
			
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
		
	for l1 in range(1, highestLayer+1):
		#print("l1 = " + str(l1))
		if(supportSkipLayers):
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(0, l1):
				#print("l2 = " + str(l2))
				#print("Atrace[generateParameterNameNetwork(networkIndex, l2, \"Atrace\")] = ", Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")].shape)
				#print("Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, \"Wf\")] = ", Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")].shape)
				#print("B[generateParameterNameNetwork(networkIndex, l1, \"B\")] = ", B[generateParameterNameNetwork(networkIndex, l1, "B")].shape)
				Z = tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]))
			Z = tf.add(Z, B[generateParameterNameNetwork(networkIndex, l1, "B")])
		else:
			Z = tf.add(tf.matmul(AprevLayer, Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]), B[generateParameterNameNetwork(networkIndex, l1, "B")])		
		A = activationFunction(Z)
		AprevLayer = A
		#print(Z)
		
		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A

	ZhighestLayer = Z
	AhighestLayer = A
		
	return ZhighestLayer, AhighestLayer	
	
def neuralNetworkPropagationFBANN(x, networkIndex=1):
			
	AlowestLayer = x
	
	resetTraceNeuralNetworkParametersFBANN()

	if(additiveZmethodRecordBackwardFeedDelta):
		additiveZfeedForward=True
	else:
		additiveZfeedForward=False
	additiveZfeedBackward = True
			
	for feedIteration in range(1, numberOfFeedBackwardsIterations+1):
		ZhighestLayer, AhighestLayer = neuralNetworkPropagationFBANNfeedForward(AlowestLayer, additiveZ=additiveZfeedForward, networkIndex=networkIndex)
		ZlowestLayer, AlowestLayer = neuralNetworkPropagationFBANNfeedBackward(AhighestLayer, additiveZ=additiveZfeedBackward, networkIndex=networkIndex)
			
	ZhighestLayer, AhighestLayer = neuralNetworkPropagationFBANNfeedForward(x, additiveZ=additiveZfeedForward, networkIndex=networkIndex)
	
	return tf.nn.softmax(ZhighestLayer)
	
def neuralNetworkPropagationFBANNfeedForward(x, additiveZ=False, networkIndex=1):
			
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	
	for l1 in range(1, highestLayer+1):
	
		#print("l1 = " + str(l1))

		if(additiveZ):
			if(additiveZmethodRecordBackwardFeedDelta):
				Z = ZtraceBackwardFeedDelta[generateParameterNameNetwork(networkIndex, l1, "ZtraceBackwardFeedDelta")] 
			else:
				Z = Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] 
		else:
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
		
		#print("Z = ", Z)
			
		if(supportSkipLayers):
			Zmod = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(0, l1):
				Zmod = tf.add(Zmod, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")]), B[generateParameterNameNetwork(networkIndex, l1, "B")]))
		else:
			Zmod = tf.add(tf.matmul(AprevLayer, Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")]), B[generateParameterNameNetwork(networkIndex, l1, "B")])
		
		Z = tf.add(Z, Zmod)
		
		A = activationFunction(Z)
			
		Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
		Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A

		AprevLayer = A
	
	ZhighestLayer = Z
	AhighestLayer = A
		
	return ZhighestLayer, AhighestLayer

def neuralNetworkPropagationFBANNfeedBackward(AhighestLayer, additiveZ=True, networkIndex=1):
				
	AprevLayer = AhighestLayer
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, highestLayer+1, "Atrace")] = AprevLayer
	
	#print("lowestLayerWithIncomingBackwardsConnections = " , lowestLayerWithIncomingBackwardsConnections)
	#print("highestLayerWithIncomingBackwardsConnections = " , highestLayerWithIncomingBackwardsConnections)
	
	for l1 in reversed(range(lowestLayerWithIncomingBackwardsConnections, highestLayerWithIncomingBackwardsConnections+1)):
	
		#print("l1 = " + str(l1))

		if(additiveZ):
			Z = Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] 
		else:
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			
		if(supportSkipLayers):
			Zmod = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(l1+1, highestLayer+1):
				#print("l2 = ", l2)
				Zmod = tf.add(Zmod, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")]), B[generateParameterNameNetwork(networkIndex, l1, "B")]))
		else:
			Zmod = tf.add(tf.matmul(AprevLayer, Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")]), B[generateParameterNameNetwork(networkIndex, l1, "B")])
		
		Z = tf.add(Z, Zmod)
		
		A = activationFunction(Z)
		
		Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
		Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
		
		if(additiveZmethodRecordBackwardFeedDelta):
			ZtraceBackwardFeedDelta[generateParameterNameNetwork(networkIndex, l1, "ZtraceBackwardFeedDelta")]  = Zmod
		
		AprevLayer = A
			
	ZlowestLayer = Z
	AlowestLayer = A
		
	return ZlowestLayer, AlowestLayer
	

def activationFunction(Z):
	return reluBasic(Z)
			
def reluBasic(Z):

	A = tf.nn.relu(Z)
	
	return A

  
