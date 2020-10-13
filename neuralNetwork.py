#%%
import pandas as pd
from pandas import Series
from pandas import DataFrame
import math
import random

#%%

def sigmoidActivationFunction(x: float) -> float:
    return 1 / ( 1 + math.exp(-x))

def sigmoidActivationFunctionDerivative(x: float) -> float:
    gX = sigmoidActivationFunction(x) 
    return gX * (1 - gX)
#%%
class Neuron:
    """# Neuron
    This is the class representation of a Neuron.
    """
    def __init__(self, weights: Series, bias: float) -> None:
        self.weights: Series = weights
        self.bias = bias
        self.functionInput: float = 0
        self.dWeight1 = 0
        self.dWeight2 = 0
        self.dBias = 0
        self.dOutput = 0
    
    def calculateFunctionInput(self, inputXVector: Series):
        self.functionInput = self.weights.dot(inputXVector) + self.bias

    def feedForward(self, inputXVector: Series) -> float:
        
        #X = [3,4]
        #W = [1,1]
        #b = 7
        # X*W + b
        # [3,4] * [1,1] + 7
        # 3 * 1 + 4 * 1 + 7 == 14
        # feedforward ==> returns activationFunction(14)
        self.calculateFunctionInput(inputXVector=inputXVector)
        return sigmoidActivationFunction(self.functionInput)
    
    def dNeuronFeedforward(self,neuron, inputXVector: Series) -> float:
        self.calculateFunctionInput(inputXVector=inputXVector)
        self.dWeight1 = inputXVector[0] * sigmoidActivationFunctionDerivative(self.functionInput)
        self.dWeight2 = inputXVector[1] * sigmoidActivationFunctionDerivative(self.functionInput)
        self.dBias = sigmoidActivationFunctionDerivative(self.functionInput)
        self.dOutput = sigmoidActivationFunctionDerivative(self.functionInput)
# %%

weights = pd.Series([0.3,0.7])
bias = 3

someNeuron: Neuron = Neuron(weights=weights,bias=bias)
# %%
someInputVector: Series = pd.Series([5,6])

print(someNeuron.feedForward(someInputVector))
# %%
# %%
weights = pd.Series([0.3,0.7])
bias = 0
someNeuron: Neuron = Neuron(weights=weights,bias=bias)

someInputVector: Series = pd.Series([0,0])

print(someNeuron.feedForward(someInputVector))
# %%
class NeuralNetwork:
    """# NeuralNetwork
    This is the network implementation that creates a network of Neuron class objects.
    """
    def __init__(self, weights: Series, bias: float) -> None:
        self.weights = weights
        self.bias = bias
        self.inputLayer: Neuron = Neuron(weights=weights,bias=bias)
        self.input2Layer: Neuron = Neuron(weights=weights,bias=bias)
        self.outputLayer: Neuron = Neuron(weights=weights,bias=bias)
    
    def feedForward(self, inputXVector: float) -> float:
        """ This function feeds the two inputs into the output layer. """
        inputOut: float = self.inputLayer.feedForward(inputXVector=inputXVector)
        input2Out: float = self.input2Layer.feedForward(inputXVector=inputXVector)
        output = self.outputLayer.feedForward(pd.Series([inputOut,input2Out]))
        return output



#%%
neuralNet: NeuralNetwork = NeuralNetwork(weights=weights,bias=bias)
# %%
print(neuralNet.feedForward(inputXVector=someInputVector))
# %%
class trainableNeuralNetwork:

    def __init__(self) -> None:
        self.weight1: float = random.random()
        self.weight2: float = random.random()
        self.weight3: float = random.random()
        self.weight4: float = random.random()
        self.weight5: float = random.random()
        self.weight6: float = random.random()

        self.bias1: float = random.random()
        self.bias2: float = random.random()
        self.bias3: float = random.random()

        self.input1: Neuron = Neuron(weights=pd.Series([self.weight1,self.weight2]), bias=self.bias1)
        self.input2: Neuron = Neuron(weights=pd.Series([self.weight3,self.weight4]), bias=self.bias2)
        self.output: Neuron = Neuron(weights=pd.Series([self.weight5,self.weight6]), bias=self.bias3)
    
    def feedForward(self, inputXVector: Series) -> float:
        input1Out: float = self.input1.feedForward(inputXVector=inputXVector)
        input2Out: float = self.input2.feedForward(inputXVector=inputXVector)
        out: float = self.output.feedForward(inputXVector=pd.Series([input1Out,input2Out]))
    

    def trainNeuralNet(self, inputData: DataFrame, lables: Series) -> None:

        trainingIterations: int = 100
        learningRate: float = 0.25
        potentialNeuralNets = []

        for i in range(0, trainingIterations):
            tnn = None
            for j in range(0,len(inputData)):
                tnn = trainableNeuralNetwork()
                predictedOutput: float = tnn.feedForward(inputData[j])

                dPredictedOutput: float = -2 * (lables[j] - predictedOutput)
                tnn.input1.dNeuronFeedforward(inputXVector=inputData[j])
                tnn.input2.dNeuronFeedforward(inputXVector=inputData[j])
                tnn.output.dNeuronFeedforward(inputXVector=inputData[j])

                tnn.input1.weights[0] = tnn.input1.weights[0] - learningRate*dPredictedOutput*tnn.input1.dOutput*tnn.input1.dWeight1
                tnn.input1.weights[1] = tnn.input1.weights[1] - learningRate*dPredictedOutput*tnn.input1.dOutput*tnn.input1.dWeight2
                tnn.input1.bias = learningRate * dPredictedOutput * dPredictedOutput*tnn.input1.dOutput * tnn.input1.dBias

                tnn.input2.weights[0] = tnn.input2.weights[0] - learningRate*dPredictedOutput*tnn.input2.dOutput*tnn.input2.dWeight1
                tnn.input2.weights[1] = tnn.input2.weights[1] - learningRate*dPredictedOutput*tnn.input2.dOutput*tnn.input2.dWeight2
                tnn.input2.bias = learningRate * dPredictedOutput * dPredictedOutput*tnn.input2.dOutput * tnn.input2.dBias

                tnn.output.weights[0] = tnn.output.weights[0] - learningRate*dPredictedOutput*tnn.output.dOutput*tnn.output.dWeight1
                tnn.output.weights[1] = tnn.output.weights[1] - learningRate*dPredictedOutput*tnn.output.dOutput*tnn.output.dWeight2
                tnn.output.bias = learningRate * dPredictedOutput * dPredictedOutput*tnn.output.dOutput * tnn.output.dBias
            potentialNeuralNets.append(tnn)
        k = 0
        for tnn in potentialNeuralNets:
            inputData[f'{k}_lables'] = inputData.apply(tnn.feedForward)
            k += 1



