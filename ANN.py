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