#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
#
# Author 2 : Uthama Kadengodlu          NETID: UXK210012
# Implements Artificial Neural Network
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class ArtificialNeuralNetwork:
    def get_data(void):
        Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ANN/main/s41598-020-73558-3_sepsis_survival_primary_cohort.csv",skiprows=[0], header = None)
        Testing_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ANN/main/s41598-020-73558-3_sepsis_survival_validation_cohort.csv",skiprows=[0], header = None)  
        return Training_file,Testing_file

    def sigmoid_ac(x):
        return 1/(1 + np.exp(-x))

    def tanh_ac(x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    def relu_ac(x):
        if(x < 0):
            return 0
        else: 
            return x

    def forward_propagation(weights,activation,data):
        x1 = data[0]
        x2 = data[1]
        x3 = data[2]

        net123 = [1,x1,x2,x3]
        weight4 = np.array(weights[0])
        weight5 = np.array(weights[1])
        weight6 = np.array(weights[2])

        net4 = weight4.T @ net123
        net5 = weight5.T @ net123
        net6 = weight6.T @ net123
        
        if(activation == "sigmoid"):
            x4 = ArtificialNeuralNetwork.sigmoid_ac(net4)
            x5 = ArtificialNeuralNetwork.sigmoid_ac(net5)
            x6 = ArtificialNeuralNetwork.sigmoid_ac(net6)
        if(activation == "tanh"):
            x4 = ArtificialNeuralNetwork.tanh_ac(net4)
            x5 = ArtificialNeuralNetwork.tanh_ac(net5)
            x6 = ArtificialNeuralNetwork.tanh_ac(net6)
        if(activation == "relu"):
            x4 = ArtificialNeuralNetwork.relu_ac(net4)
            x5 = ArtificialNeuralNetwork.relu_ac(net5)
            x6 = ArtificialNeuralNetwork.relu_ac(net6)

        net7 = weights[3][0]*1 + weights[3][1]*x4 + weights[3][2]*x5 +weights[3][3]*x6

        if(activation == "sigmoid"):
            return  ArtificialNeuralNetwork.sigmoid_ac(net7)
        if(activation == "tanh"):
            return  ArtificialNeuralNetwork.tanh_ac(net7)
        if(activation == "relu"):
            return  ArtificialNeuralNetwork.relu_ac(net7)



if __name__ == "__main__":
    nn = ArtificialNeuralNetwork
    print(nn.forward_propagation([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],"sigmoid",[2,2,2]))
    print(nn.forward_propagation([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],"tanh",[2,2,2]))
    print(nn.forward_propagation([[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]],"relu",[2,2,2]))
