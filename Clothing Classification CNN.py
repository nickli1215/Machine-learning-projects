#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries
import pandas as pd
import numpy as np
import os
from os import path as op
# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# for creating validation set
from sklearn.model_selection import train_test_split

#for evaluating the model, this uses sklearn metrics
#tqdm is to give us an estimate of how long our model might take
#has a nice little progress bar :D
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm

#Pytorch libraries/modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn as nn


# In[2]:


directory = os.getcwd()
trainingfile = "train.csv"
testingfile = "test.csv"
sample_file = "sample_submission_I5njJSF.csv"
print(directory)
print(op.join(directory, ("test_ScVgIM0"+ trainingfile)))


# In[3]:


directory = os.getcwd()
trainingfile = "train.csv"
testingfile = "test.csv"
sample_file = "sample_submission_I5njJSF.csv"
train = pd.read_csv(op.join(directory, trainingfile))
test = pd.read_csv(op.join(directory, testingfile))
sample_submission = pd.read_csv(op.join(directory, sample_file))

train.head(10)


# In[5]:


#loading images
train_img = []
for img_name in tqdm(range(1,60001)):
    image_path = op.join(directory, "train",  str(img_name) + '.png')
    img = imread(image_path, as_gray = True)
    img = img/255.0
    img = img.astype('float32')
    train_img.append(img)


# In[6]:


train_x = np.array(train_img)
train_y = train['label'].values
train_y = train_y[0:60000].astype('long')
# visualizing images
i = 0
plt.figure(figsize=(10,10))
plt.subplot(221), plt.imshow(train_x[0], cmap='gray')
plt.subplot(222), plt.imshow(train_x[1], cmap='gray')
plt.subplot(223), plt.imshow(train_x[2], cmap='gray')
plt.subplot(224), plt.imshow(train_x[19997], cmap='gray')

print(train_y)


# In[7]:


train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.2)
val_x, test_x, val_y, test_y = train_test_split(val_x,val_y, test_size = 0.5)
(train_x.shape,train_y.shape),(val_x.shape,val_y.shape), (test_x.shape, test_y.shape)


# In[8]:


train_x = train_x.reshape(48000,1,28,28)
train_x = torch.from_numpy(train_x)

train_y = torch.from_numpy(train_y)

train_x.shape, train_y.shape


# In[9]:


val_x = val_x.reshape(6000, 1, 28, 28)
val_x = torch.from_numpy(val_x)


# In[10]:


val_y = val_y.astype('long')
val_y = torch.from_numpy(val_y)

val_x.shape, val_y.shape


# In[11]:


test_x = test_x.reshape(6000,1,28, 28)
test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y)
test_x.shape, test_y.shape


# In[13]:


class CNN(Module):
    def __init__(self):
        super(CNN, self).__init__()
        #Defining the 2d convolutional layers
        self.cnn_layers = Sequential(
            #layer 1 #28x28
            Conv2d(1,8,kernel_size=3, stride=1, padding=1), #convolution 28 to 28
            BatchNorm2d(8), 
            ReLU(inplace=True), #RelU activation
            MaxPool2d(kernel_size=2, stride=2), #28 to 14
        )
        self.linear_layers = Sequential(
            Linear(8*14*14, 100),
            Linear(100, 10)
        )
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0),-1)
        x = self.linear_layers(x)
        return x


# In[14]:


model = CNN()

optimizer = Adam(model.parameters(), lr = 0.01)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()


# In[15]:


def NNtrain(epoch):
    model.train()
    tr_loss = 0

    x_train, y_train = Variable(train_x), Variable(train_y.long())
    x_val, y_val = Variable(val_x), Variable(val_y.long())
    
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    optimizer.zero_grad()
    output_train = model(x_train)
    output_val = model(x_val)
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    
    train_losses.append(loss_train)
    val_losses.append(loss_val)
    
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    
    if epoch%2 == 0:
        print ('Epoch: ', epoch+1, '\t', 'loss :', loss_val)


# In[16]:


n_epochs = 50

train_losses = []
val_losses = []
for epoch in range(n_epochs): 
    NNtrain(epoch)


# In[21]:


# plotting the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()


# In[22]:


# prediction for training set
with torch.no_grad():
    test_output = model(test_x)
    training_output = model(train_x)

test_softmax = torch.exp(test_output)
test_prob = list(test_softmax.numpy())
test_predictions = np.argmax(test_prob, axis=1)
test_breakdown = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
train_softmax = torch.exp(training_output)
train_prob = list(train_softmax.numpy())
train_breakdown = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]
#The code below was for me to breakdown the testing and training accuracies so that I could determine which classes are doing badly
#Its not really necessary because the confusion matrix already does this, so I will probably remove it later.
for i in range(0,len(test_predictions)):
    if test_predictions[i] == test_y[i]:
        test_breakdown[test_y[i]][0]+=1
    test_breakdown[test_y[i]][1]+=1
for i in range(0,len(test_predictions)):
    if train_predictions[i] == train_y[i]:
        train_breakdown[train_y[i]][0]+=1
    train_breakdown[train_y[i]][1]+=1
print ("   Train   Testing")
for i in range(0, len(train_breakdown)):
    print(str(i)+": "+"{:.2f}".format(train_breakdown[i][0]/train_breakdown[i][1])+ "    "+"{:.2f}".format(test_breakdown[i][0]/test_breakdown[i][1]) )
print(" Test accuracy: " + str(accuracy_score(test_y, test_predictions)))
print("Training accuracy: " + str(accuracy_score(train_y,train_predictions)))

train_confusion = confusion_matrix(train_y, train_predictions)
test_confusion = confusion_matrix(test_y, test_predictions)
print ("Training confusion matrix: ")
print(train_confusion)
print("Testing confusion matrix: ")
print(test_confusion)


# In[18]:





# In[ ]:





# In[ ]:





# In[ ]:




