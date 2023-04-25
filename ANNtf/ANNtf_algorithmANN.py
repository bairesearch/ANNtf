"""ANNtf_algorithmANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf_main.py

# Usage:
see ANNtf_main.py

# Description:
ANNtf algorithm ANN - define artificial neural network

"""

import tensorflow as tf
import numpy as np
from ANNtf_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf_operations
import ANNtf_globalDefs

debugOnlyTrainFinalLayer = False
debugSingleLayerNetwork = False
debugFastTrain = False
debugSingleLayerNetwork = False

supportMultipleNetworks = True

supportSkipLayers = False

normaliseFirstLayer = False
equaliseNumberExamplesPerClass = False


W = {}
B = {}
if(supportMultipleNetworks):
	WallNetworksFinalLayer = None
	BallNetworksFinalLayer = None
if(supportSkipLayers):
	Ztrace = {}
	Atrace = {}

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0

def defineTrainingParameters(dataset):
	global batchSize
	
	learningRate = 0.001
	batchSize = 100
	numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000

	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	if(debugSingleLayerNetwork):
		n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf_operations.defineNetworkParametersANNsingleLayer(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet)
	else:
		n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=False)
	
	return numberOfLayers
	
	
def defineNetworkParametersANNlegacy(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	datasetNumClasses = num_output_neurons
	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	n_h_0 = n_x
	if(dataset == "POStagSequence"):
		n_h_1 = int(datasetNumFeatures*3) # 1st layer number of neurons.
		n_h_2 = int(datasetNumFeatures/2) # 2nd layer number of neurons.
	elif(dataset == "SmallDataset"):
		n_h_1 = 4
		n_h_2 = 4
	else:
		print("dataset unsupported")
		exit()
	n_h_3 = n_y
	if(debugSingleLayerNetwork):
		n_h = [n_h_0, n_h_3]	
	else:
		n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
	numberOfLayers = len(n_h)-1
	
	print("defineNetworkParametersANNlegacy, n_h = ", n_h)
	
	return n_h, numberOfLayers, numberOfNetworks, datasetNumClasses
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		
		for l1 in range(1, numberOfLayers+1):
		
			if(supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						Wlayer = randomNormal([n_h[l2], n_h[l1]]) 
						W[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "W")] = tf.Variable(Wlayer)
			else:	
				Wlayer = tf.Variable(randomNormal([n_h[l1-1], n_h[l1]]))
				W[generateParameterNameNetwork(networkIndex, l1, "W")] = Wlayer
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(tf.zeros(n_h[l1]))

			if(supportSkipLayers):
				Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))

			#print("Wlayer = ", W[generateParameterNameNetwork(networkIndex, l1, "W")])

	if(supportMultipleNetworks):
		if(numberOfNetworks > 1):
			global WallNetworksFinalLayer
			global BallNetworksFinalLayer
			WlayerF = randomNormal([n_h[numberOfLayers-1]*numberOfNetworks, n_h[numberOfLayers]])
			WallNetworksFinalLayer = tf.Variable(WlayerF)
			BlayerF = tf.zeros(n_h[numberOfLayers])
			BallNetworksFinalLayer= tf.Variable(BlayerF)	#not currently used
					
def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationANN(x, networkIndex)

def neuralNetworkPropagationLayer(x, networkIndex=1, l=None):
	return neuralNetworkPropagationANN(x, networkIndex, l)

#if(supportMultipleNetworks):
def neuralNetworkPropagationAllNetworksFinalLayer(AprevLayer):
	Z = tf.add(tf.matmul(AprevLayer, WallNetworksFinalLayer), BallNetworksFinalLayer)	
	#Z = tf.matmul(AprevLayer, WallNetworksFinalLayer)	
	pred = tf.nn.softmax(Z)	
	return pred
		
def neuralNetworkPropagationANN(x, networkIndex=1, l=None):
			
	#print("numberOfLayers", numberOfLayers)

	if(l == None):
		maxLayer = numberOfLayers
	else:
		maxLayer = l
			
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	
	for l1 in range(1, maxLayer+1):
		if(supportSkipLayers):
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(0, l1):
				Wlayer = W[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "W")]
				at = Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")]
				Z = tf.add(Z, tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], Wlayer))
			Z = tf.add(Z, B[generateParameterNameNetwork(networkIndex, l1, "B")])
		else:
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l1, "W")]), B[generateParameterNameNetwork(networkIndex, l1, "B")])
		
		A = activationFunction(Z)

		#print("l1 = " + str(l1))		
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l1, "W")] )
		
		if(debugOnlyTrainFinalLayer):
			if(l1 < numberOfLayers):
				A = tf.stop_gradient(A)

		if(supportSkipLayers):
			Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
			Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
						
		AprevLayer = A

	if(maxLayer == numberOfLayers):
		return tf.nn.softmax(Z)
	else:
		return A

def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	

