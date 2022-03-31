import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
#Neural Network Project mnist wrapper implementation
#By: Edgar A. Rodriguez
#Date: 11/22/2020
#################################
#Python function implementations#
#################################
def percentdiff(new,old):
    difference = np.absolute(new - old)
    if (difference == 0.0):
        return difference
    denominator = (new + old) / 2
    percentdifference = (difference / denominator) * 100
    return percentdifference
######
#Main#
######
#initializing the mnist dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
#reshaping and normalizing the dataset
X_train = X_train/255.0
X_test = X_test/255.0
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

#One-hot Encoding
numbers = np.arange(10)
temp = [(k == numbers).astype(int) for k in y_train] 
y_train = np.vstack(temp)
temp = [(k == numbers).astype(int) for k in y_test]
y_test = np.vstack(temp)

net = nn.net(784, alpha = 0.01, mu = 0.01, loss = 'CCE')
#creating the artificial neural network layers
#(number of nodes, activation function)
#default activation function = ReLU
net.add_layer(512,activation = 'ReLU')
net.add_layer(256, activation = 'ReLU')
net.add_layer(10, activation = 'sigmoid')
# creating an empty 2d array for predicting
pred_array = np.empty((0,10))
#train the keras mnist dataset
epochs = 1
loss = []
count = 1
#early stopping function parameters
losscounter = 0 #if counter reaches 100, then early stopping function comes into effect
prevloss = 0
epochcount = 0
for epoch in range(epochs):
    #obtains loss value list for each row being trained
    temp = []
    print('for epoch: ', epoch, "\n")
    for x,y in zip(X_train, y_train):
        #print current row being trained
        print(count)
        net.feed_forward(x)
        net.back_propogation(x,y)
        net.update_weights()
        temp.append(net.loss_value)
        
        '''#save the weights individually
        with open('mnistweight0.npy', 'wb') as f:
            np.save(f, net.layers[0].weights)
        with open('mnistweight1.npy', 'wb') as f:
            np.save(f, net.layers[1].weights)
        with open('mnistweight2.npy', 'wb') as f:
            np.save(f, net.layers[2].weights)'''
        count += 1
    count = 0    
    epochcount = epochcount + 1
    #early stopping function check
    losspercentage = percentdiff(np.mean(temp), prevloss)
    prevloss = np.mean(temp)
    print(losspercentage)
    if (losspercentage < 10.0):
        losscounter = losscounter + 1
    else:
        losscounter = 0
    loss.append(np.mean(temp))
    #if the loss doesn't change after 100 times, break the loop
    if (losscounter == 100):
        print('x is', x, 'and y is', y, '\n')
        print('output is: ', net.output, '\n')
        break

'''#save the weights individually
with open('mnistweight0.npy', 'wb') as f:
    np.save(f, net.layers[0].weights)
with open('mnistweight1.npy', 'wb') as f:
    np.save(f, net.layers[1].weights)
with open('mnistweight2.npy', 'wb') as f:
    np.save(f, net.layers[2].weights) '''
#load the weights individually
'''    
with open('mnistweight0.npy', 'rb') as f:
    net.layers[0].weights = np.load(f)
with open('mnistweight1.npy', 'rb') as f:
    net.layers[1].weights = np.load(f)
with open('mnistweight2.npy', 'rb') as f:
    net.layers[2].weights = np.load(f) '''
####################
#Output Evaluations#
####################
#Comment out which plot to not evaluate (loss vs density vs confusion matrix)
#plotting the epoch vs mean loss graph
'''if (losscounter != 100):   
    plt.plot(range(epochs), loss)
else:
    plt.plot(range(epochcount), loss)
plt.xlabel('epochs')
if(net.lossfunction == 'MSE'):
    plt.ylabel('MSE')
if(net.lossfunction == 'CCE'):
    plt.ylabel('CCE')'''
    

#predicting the output
for x in X_test:
    net.feed_forward(x)
    pred_array = np.append(pred_array, np.array([net.output]), axis=0)
    
#one hot encoding to an integer between 0 and 9
y_pred = [np.argmax(r) for r in pred_array]
y_true = [np.argmax(r) for r in y_test]

#Density plot using pandas
#Converting array to pandas DataFrame
'''df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred,} ) 
df.plot.kde()'''

#compute and print the accuracy score
print(accuracy_score(y_true, y_pred))
#generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')
#print the classification report 
print(classification_report(y_true, y_pred)) 