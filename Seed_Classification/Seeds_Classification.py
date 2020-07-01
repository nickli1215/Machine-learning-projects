import csv
import torch
import torch.nn as nn
import numpy as np
import numpy.random as rand
from pathlib import Path
import matplotlib.pyplot as plt

data_folder = Path("C:/Users/Nicholas/AppData/Local/Programs/Python/Python36-32/PyTorch practice")
datafile = data_folder/ "Seed_Data.csv"
features = 7
data = np.empty([0,0])

#reading data and formatting into 2 arrays
with open(datafile) as file:
    seeds = csv.reader(file, delimiter = ',')
    next(seeds)
    data = np.array([[float(i) for i in next(seeds)]], dtype = float)
    for row in seeds:
        data = np.append(data,[[float(i) for i in row]], axis = 0)

        
#shuffle and assign data. Adding bias term features to X.
#Convert y into M x K sized matrix so that each row vector represents a class.
#For example, a data sample of class 2 would be converted into a vector [0,0,1].
rand.shuffle(data)
X = np.array(data[0:,0:features])
classes = data[0:,features]
y = np.zeros ((data.shape[0],3))
for i in range(len(classes)):
    y[i,int(classes[i])] = 1

    
#feature scaling data with Z-score standardization
m = X.mean(0)
std = X.std(0)
X = (X-m)/std
X = np.hstack((np.ones((data.shape[0],1),dtype = np.float64),X))

#separating data into training and testing samples.
totalSize = X.shape[0]
trainingSize = totalSize*7//10
X_training = X[0:trainingSize]
y_training = y[0:trainingSize]
X_test = X[trainingSize:totalSize]
y_test = y[trainingSize:totalSize]

 
class Logistic_Regression:
    def __init__(self):
        self.inputSize = 8
        self.outputSize = 3
        epsilon = -1.0
        # I use numpy.random.randint over torch.randn so that I can limit the initial size of the parameters
        self.W = rand.uniform(low = -epsilon,high = epsilon,size = (self.inputSize,self.outputSize))

    def sigmoid(self,x): #sigmoid function
        return 1/(1+np.exp(x))
    
    def costfunction (self,X,y): #cost function 
        z = np.dot(X,self.W)
        h = self.sigmoid(z)
        m = X.shape[0]
        cost = sum(sum((-y*np.log(h)) - ((1-y)*np.log(1-h))))/m
        return cost
    def train (self, X,y,iterations): #Training the algorithm using a simple gradient descent algorithm.
        for i in range(iterations):
            p = self.sigmoid(np.dot(X,self.W))
            d = p-y
            m = y.shape[0]
            grad = (1/m)*(np.dot(np.transpose(X),d))
            alpha = 0.1
            self.W = self.W + alpha*grad
            print(self.costfunction(X,y))
    def predict (self, X):
        z = np.dot(X,self.W)
        h = self.sigmoid(z)
        return np.argmax(h, axis = 1)
    
K  = Logistic_Regression()
print(K.costfunction(X,y))
K.train(X_training,y_training,2000)
ypred = K.predict(X_test)
y_ans = np.argmax(y_test,axis = 1)
print(y_ans == ypred)
print( "Accuracy: " + str(sum(y_ans ==  ypred)/len(y_ans)))
print(K.W)

