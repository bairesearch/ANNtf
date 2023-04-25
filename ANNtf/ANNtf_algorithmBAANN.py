"""ANNtf_algorithmBAANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf_main.py

# Usage:
see ANNtf_main.py

# Description:
ANNtf algorithm BAANN - define breakaway artificial neural network 

layers are incrementally trained using class targets
each layer is fully connected to all previous layers (skip connections)
network learns to better (more accurately) discriminate the class target based upon more complex inferences performed by higher layers
based on https://github.com/baxterai/ANN/blob/master/ANN/ANNalgorithmBreakawayNetwork.cpp
ANNtf_algorithmBAANN uses keras standard api

"""



import tensorflow as tf
import numpy as np
import ANNtf_operations
import ANNtf_globalDefs

useTFdataset = True	#repeat and shuffle data

normaliseFirstLayer = False
equaliseNumberExamplesPerClass = False

onlyTrainFinalLayer = True
learningAlgorithmNone = True	#create a very large network (eg x10) neurons per layer, and perform final layer backprop only
if(learningAlgorithmNone):
	num_hidden_neurons = 100
	debugLearningAlgorithmNoneTrainAllLayers = False
	if(debugLearningAlgorithmNoneTrainAllLayers):
		onlyTrainFinalLayer = False	
	
numberLayers = 6

def defineTrainingParameters(dataset):
	learningRate = 0.001
	batchSize = 100
	numEpochs = 10	#100 #10
	if(useTFdataset):
		trainingSteps = 1000
	else:
		trainingSteps = 25

	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	
	
def BAANNmain(train_x, train_y, test_x, test_y, num_input_neurons, num_output_neurons, batchSize, trainingSteps, numEpochs):
	BAANNtrainNetwork(train_x, train_y, test_x, test_y, num_input_neurons, num_output_neurons, batchSize, trainingSteps, numEpochs)
	
def BAANNtrainNetwork(train_x, train_y, test_x, test_y, num_input_neurons, num_output_neurons, batchSize, trainingSteps, numEpochs):
	
	#onehot encode y;
	train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_output_neurons, dtype='float32')
	test_y = tf.keras.utils.to_categorical(test_y, num_classes=num_output_neurons, dtype='float32')

	if(useTFdataset):
		datasetNumExamples = train_x.shape[0]	
		shuffleSize = datasetNumExamples
		train_x = ANNtf_operations.generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)	#CHECKTHIS (is repeat or shuffle required?)

	firstLayer = True
	model = None
	for l in range(numberLayers):
		print("\nl = ", l)
		model = upgradeModelAddLayer(firstLayer, model, num_input_neurons, num_output_neurons)
		#model = createTestModel(firstLayer, model, num_input_neurons, num_output_neurons)
		trainModel(firstLayer, model, train_x, train_y, trainingSteps, numEpochs)
		firstLayer = False
		
		print("model.evaluate")
		model.evaluate(test_x, test_y, batch_size=batchSize)   
		#testsetPredictions = model.predict(test_x)

def createTestModel(firstLayer, existingModel, num_input_neurons, num_output_neurons):
	currentInput = tf.keras.Input(shape=(num_input_neurons), name="currentInput")
	currentOutput = tf.keras.layers.Dense(num_output_neurons)(currentInput)
	currentOutput = tf.keras.layers.Softmax()(currentOutput)
	model = tf.keras.Model(inputs=currentInput, outputs=currentOutput)
	return model
	
def upgradeModelAddLayer(firstLayer, existingModel, num_input_neurons, num_output_neurons):
	#create a new model on top of previous model
	print("num_input_neurons = ", num_input_neurons)
	currentInput = tf.keras.Input(shape=(num_input_neurons), name="currentInput")
	if(firstLayer):
		layerInput = currentInput
		if(learningAlgorithmNone):
			layerInput = tf.keras.layers.Dense(num_hidden_neurons)(layerInput)
		if(onlyTrainFinalLayer):
			layerInput = tf.stop_gradient(layerInput)		
		currentOutput = tf.keras.layers.Dense(num_output_neurons)(layerInput)
		currentOutput = tf.keras.layers.Softmax()(currentOutput)
		skipConnectionOutput = layerInput
		upgradedModel = tf.keras.Model(inputs=currentInput, outputs=[currentOutput, skipConnectionOutput])
	else:
		previousLayerOutput, previousLayerSkipConnectionOutput = existingModel(currentInput, training=False)
		layerInput = tf.keras.layers.Concatenate()([previousLayerOutput, previousLayerSkipConnectionOutput])	#add skip connections
		if(learningAlgorithmNone):
			layerInput = tf.keras.layers.Dense(num_hidden_neurons)(layerInput)
		if(onlyTrainFinalLayer):
			layerInput = tf.stop_gradient(layerInput)
		currentOutput = tf.keras.layers.Dense(num_output_neurons)(layerInput)
		currentOutput = tf.keras.layers.Softmax()(currentOutput)
		skipConnectionOutput = layerInput
		#upgradedModel = tf.keras.Model(input=[currentInput, skipConnectionInput], output=[currentOutput, skipConnectionOutput])
		upgradedModel = tf.keras.Model(inputs=currentInput, outputs=[currentOutput, skipConnectionOutput])
	return upgradedModel

def trainModel(firstLayer, model, train_x, train_y, trainingSteps, numEpochs):
	model.compile(optimizer=tf.keras.optimizers.Adam(), loss=[tf.keras.losses.CategoricalCrossentropy(from_logits=True), None], metrics=['accuracy'])	#BinaryCrossentropy
	#print("numEpochs = ", numEpochs)
	#print("trainingSteps = ", trainingSteps)
	#model.fit(x=train_x, y=train_y, epochs=numEpochs, steps_per_epoch=trainingSteps)
	model.fit(x=train_x, epochs=numEpochs, steps_per_epoch=trainingSteps)
		#https://www.tensorflow.org/api_docs/python/tf/keras/Model
		#model.fit(x, (y, y))
