#
#
# Author 1 : Shishir Sunil Yalburgi     NETID: SSY220000
#
# Author 2 : Uthama Kadengodlu          NETID: UXK210012
# Implements Artificial Neural Network

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Training_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ANN/main/Iris.csv",skiprows=[0], header = None)
Training_file = (Training_file - Training_file.mean())/Training_file.std()
Testing_file = pd.read_csv("https://raw.githubusercontent.com/Raftaar-98/ANN/main/Train.csv",skiprows=[0], header = None)
Testing_file2 = (Testing_file - Testing_file.mean())/Testing_file.std()


X = Training_file.iloc[:,0:3] # features,
y = Training_file.iloc[:,3] # label
T_X  = Testing_file2.iloc[:,0:3]
T_y  = Testing_file2.iloc[:,3] 
TT_y = Testing_file.iloc[:,3] 
T_X = T_X.to_numpy()
T_y = T_y.to_numpy()
TT_y = TT_y.to_numpy()
X = X.to_numpy()
y = y.to_numpy()


    
class ArtificialNeuralNetwork(object):
    def __init__(self):
        self.input = 3 #number of features present
        self.output = 1 #Number of outputs
        self.hidden_units = 4 #Number of nodes in hidden layer
        
        # Random weight seeds
        np.random.seed(1) 
        # weights for input layer to hidden layer
        self.w1 = np.random.randn(self.input, self.hidden_units) # 4*6 matrix
        # weights for hidden layer to output layer
        self.w2 = np.random.randn(self.hidden_units, self.output) # 6*1 matrix

    def _forward_pass(self, X, activation):
        self.z2 = np.dot(self.w1.T, X.T)
        if(activation == "sigmoid"):
            self.a2 = self._sigmoid(self.z2) 
        if(activation == "relu"):
             self.a2 = self._relu(self.z2) 
        if(activation == "tanh"):
            self.a2 = self._tanh(self.z2) 
        self.z3 = np.dot(self.w2.T, self.a2)

        if(activation == "sigmoid"):
            self.a3 = self._sigmoid(self.z3)
        if(activation == "relu"):
             self.a3 = self._relu(self.z3)
        if(activation == "tanh"):
            self.a3 = self._tanh(self.z3)
        return self.a3

    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def _tanh(self, z):
        return (np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))

    def _relu(self, z):
        return z*(z>0)

    def _loss(self, predict, y):
        m = y.shape[0]
        logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
        loss = - np.sum(logprobs) / m
        return loss

    def _backward_propagation(self, X, y,activation):
        predict = self._forward_pass(X,activation)
        m = X.shape[0]
        delta3 = predict - y
        if(activation == "sigmoid"):
            dz3 = np.multiply(delta3, self._sigmoid_prime(self.z3))
        if(activation == "relu"):
            dz3 = np.multiply(delta3, self._relu_prime(self.z3))
        if(activation == "tanh"):
            dz3 = np.multiply(delta3, self._tanh_prime(self.z3))

        self.dw2 = (1/m)*np.sum(np.multiply(self.a2, dz3), axis=1).reshape(self.w2.shape)
        
        if(activation == "sigmoid"):
             delta2 = delta3*self.w2*self._sigmoid_prime(self.z2)
        if(activation == "relu"):
             delta2 = delta3*self.w2*self._relu_prime(self.z2)
        if(activation == "tanh"):
             delta2 = delta3*self.w2*self._tanh_prime(self.z2)
       
        self.dw1 = (1/m)*np.dot(X.T, delta2.T)
        
    def _sigmoid_prime(self, z):
        return self._sigmoid(z)*(1-self._sigmoid(z))

    def _tanh_prime(self,z):
        return 1 - (self._tanh(z) * self._tanh(z))

    def _relu_prime(self,z):
        return np.where(z > 0, 1.0, 0.0)


    def _update(self, learning_rate=0.2):
        self.w1 = self.w1 - learning_rate*self.dw1
        self.w2 = self.w2 - learning_rate*self.dw2

    def train(self, X, y, iteration,activation):
        for i in range(iteration):
            y_hat = self._forward_pass( X,activation)
            loss = self._loss(y_hat, y)
            self._backward_propagation(X,y,activation)
            self._update()
            if (i%10==0):
                print("Epoch",i,": ","loss: ",loss)
                
    def predict(self, X,activation):
        y_hat = self._forward_pass( X,activation)
        print(y_hat)
        y_hat = [1 if i[0] >= 0.15 else 0 for i in y_hat.T]
        return np.array(y_hat)
    
    def score(self, predict, y):
        cnt = np.sum(predict==y)
        return (cnt/len(y))*100

if __name__=='__main__':
    train_X = X # split training data and testing data
    train_y = y
    test_X = T_X
    test_y = T_y
    model = ArtificialNeuralNetwork() #initialize the model
    iteration = 200
    activation = "sigmoid"
    model.train(train_X, train_y, iteration, activation) # train model
    pre_y = model.predict(test_X,activation) # predict
    
    ax = plt.axes(projection='3d')

    xline = T_X[:50,0]
    yline = np.array(T_X[:50,2])
    zline = pre_y[:50]
  
    zline2 = np.array(TT_y[:50])
    ax.scatter3D(xline, yline, zline, 'gray',marker = '^')
    
    plt.xlabel('Normalized age')
    plt.ylabel('Normalized Episode number')
    ax.set_zlabel('dead-0 alive-1')
    
    ax.scatter3D(xline, yline, zline2,'red')
    
    plt.show()
    score = model.score(pre_y, TT_y) # get the accuracy score
    print('predict: ', pre_y)
    print('actual:', TT_y)
    print('score: ', score)