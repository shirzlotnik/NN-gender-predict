#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:45:14 2020

@author: shirzlotnik
"""

#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Created on Sat Oct 10 17:42:56 2020

@authors: shirzlotnik and karenIthak
"""

import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(0)

class NeuralNetwork:
  def __init__(self, inputs, outputs):
        # Random weights and bias initialization
    hiddenLayerNeurons, outputLayerNeurons = 2,1  # how many neurons in hidden layer 
                                                  # how many neurons in output layer
    inputLayerNeurons = inputs.shape[1]  # number of hidden layer is equal to number of columns in input neuron
    self.inputs  = inputs
    self.outputs = outputs
    self.hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons)) # create np array of random numbers in the size of the inputLayerNeurons and hiddenLayerNeurons for the weights for the hidden layers
    self.hidden_bias =np.random.uniform(size=(1,hiddenLayerNeurons)) # create np array of random numbers in the size of the hiddenLayerNeurons for the bias for the hidden layers
    self.output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons)) # create np array of random numbers in the size of the inputLayerNeurons and hiddenLayerNeurons for the weights for the output layers
    self.output_bias = np.random.uniform(size=(1,outputLayerNeurons)) # create np array of random numbers in the size of the hiddenLayerNeurons for the bias for the output layers
        
    self.error_history = [] # for the graph
    self.epoch_list = [] # for the graph
        
    
    # print the weights and bias so we can check them 
    print("Initial hidden weights: ",end='')
    print(self.hidden_weights)
    print("Initial hidden biases: ",end='')
    print(self.hidden_bias)
    print("Initial output weights: ",end='')
    print(self.output_weights)
    print("Initial output biases: ",end='')
    print(self.output_bias)
        
# activation functions
  def sigmoid (self, x):
        return 1/(1 + np.exp(-x))

  def sigmoid_derivative(self,x):
        return x * (1 - x)
    
  """
  def forward_pass(self):
        hidden_layer_activation = np.dot(self.inputs,self.hidden_weights)
        hidden_layer_activation += self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_activation)
        output_layer_activation = np.dot(hidden_layer_output,self.output_weights)
        output_layer_activation += self.output_bias
        return self.sigmoid(output_layer_activation)
   """
      
    
  def train(self, epochs,lr ):
    # Training algorithm
    for epoch in range(epochs):
        	# Forward Propagation
        # data will flow through the neural network.
        # flow forward and produce an output
        hidden_layer_activation = np.dot(self.inputs,self.hidden_weights)
        hidden_layer_activation += self.hidden_bias
        hidden_layer_output = self.sigmoid(hidden_layer_activation)
        
        output_layer_activation = np.dot(hidden_layer_output,self.output_weights)
        output_layer_activation += self.output_bias
        predicted_output = self.sigmoid(output_layer_activation)

        #predicted_output = forward_pass()
        
        	# Backpropagation
        # go back though the network to make corrections based on the output
        error = self.outputs - predicted_output
        d_predicted_output = error * self.sigmoid_derivative(predicted_output)
        self.error_history.append(np.average(np.abs(error)))
        self.epoch_list.append(epoch) # keep track of the error history 
        	
        error_hidden_layer = d_predicted_output.dot(self.output_weights.T)
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_output)
        
        	# Updating Weights and Biases
        # going backwards through the network to update weights
        self.output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
        self.output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
        self.hidden_weights += self.inputs.T.dot(d_hidden_layer) * lr
        self.hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr

    print("Final hidden weights: ",end='')
    print(self.hidden_weights)
    print("Final hidden bias: ",end='')
    print(self.hidden_bias)
    print("Final output weights: ",end='')
    print(self.output_weights)
    print("Final output bias: ",end='')
    print(self.output_bias)
    print("\nOutput from neural network after 10000 epochs: ",end='')
    print(predicted_output)
        
        
        
        
def main():
    #Input datasets
    inputs = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6],]) #diana
    expected_output = np.array([
  [1], # Alice
  [0], # Bob
  [0], # Charlie
  [1], ]) # diana
    NN = NeuralNetwork(inputs, expected_output) # מופע של המחלקה 
    epochs = 10000 # number for the train
    lr = 0.2 #learning rate
    NN.train(epochs, lr) # call train function


    # plot the error over the entire training duration
    My_list = [*range(0, 10000, 1)] # for the epochs (ציר x)
    
    plt.figure(figsize=(10,5))
    plt.plot(My_list, NN.error_history, color = 'm', linewidth=5) # create graph (thicc boi)
    plt.xlabel('Epoch') # שם של ציר x
    plt.ylabel('Error') # שם של ציר y
    plt.show()
    
    
if __name__ == "__main__": 
    main()