import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
#Neural Network Project sklearn wrapper implementation
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
#initializing the sklearn dataset
digits = datasets.load_digits()
X = digits.images.reshape((len(digits.images), -1))
labels = digits.target
#Normalizing the dataset
normalizer = MinMaxScaler()
X = normalizer.fit_transform(X)

#one hot encode the output
lr = np.arange(10)
labels = [(k == lr).astype(int) for k in labels]
labels = np.vstack(labels)

#split the data
X_train, X_test, y_train, y_test = train_test_split(X,labels)

#creating the artificial neural network
net = nn.net(64, alpha = 0.01, mu = 0.01, loss = 'CCE')
net.add_layer(512, activation = 'ReLU')
net.add_layer(56, activation = 'ReLU')
net.add_layer(10, activation = 'sigmoid')
# creating an empty 2d array for predicting
pred_array = np.empty((0,10))
#train the sklearn mnist dataset
epochs = 5
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

#save the weights individually after training is complete
with open('sklearnweight0.npy', 'wb') as f:
    np.save(f, net.layers[0].weights)
with open('sklearnweight1.npy', 'wb') as f:
    np.save(f, net.layers[1].weights)
with open('sklearnweight2.npy', 'wb') as f:
    np.save(f, net.layers[2].weights)
    
#load the weights individually
'''    
with open('sklearnweight0.npy', 'rb') as f:
    net.layers[0].weights = np.load(f)
with open('sklearnweight1.npy', 'rb') as f:
    net.layers[1].weights = np.load(f)
with open('sklearnweight2.npy', 'rb') as f:
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