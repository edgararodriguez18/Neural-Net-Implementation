import numpy as np
import neuralnet as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
#Neural Network Project XOR wrapper implementation
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
#input vectors
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
#output labels 
labels = np.array([[0],
                   [1],
                   [1],
                   [0]])
#creating the artificial neural network
test = nn.net(input_size = 2, alpha = 0.1, mu = 0.1, loss = 'CCE')
#creating the artificial neural network layers
#(number of nodes, activation function)
test.add_layer(5, activation = 'ReLU')
test.add_layer(1, activation = 'sigmoid')
#train the xor dataset
epochs = 1000
loss = []
#early stopping function parameters
losscounter = 0 
#if counter reaches 100, then early stopping function comes into effect
prevloss = 0
epochcount = 0
for epoch in range(epochs):
    #obtains loss value list for each row being trained
    temp = []
    for x,y in zip(X,labels):
        print('----------------------')
        print('for epoch: ', epoch, "\n")
        
        test.feed_forward(x)
        test.back_propogation(x,y)
        test.update_weights()
        temp.append(test.loss_value)
        #if its the last epoch, print the output    
        if (epoch == (epochs - 1)):
            print('x is', x, 'and y is', y, '\n')
            print('output is: ', test.output, '\n')                    
    epochcount = epochcount + 1
    #early stopping function check
    losspercentage = percentdiff(np.mean(temp), prevloss)
    prevloss = np.mean(temp)
    print(losspercentage)
    if (losspercentage < 0.01):
        losscounter = losscounter + 1
    else:
        losscounter = 0
    loss.append(np.mean(temp))
     #if the loss doesn't change after 100 times, break the loop
    if (losscounter == 100):
        print('x is', x, 'and y is', y, '\n')
        print('output is: ', test.output, '\n')
        break
####################
#Output Evaluations#
####################
#Comment out which plot to not evaluate (loss vs confusion matrix)
#plotting the epoch vs mean loss graph
if (losscounter != 100):   
    plt.plot(range(epochs), loss)
else:
    plt.plot(range(epochcount), loss)
plt.xlabel('epochs')
if(test.lossfunction == 'MSE'):
    plt.ylabel('MSE')
if(test.lossfunction == 'CCE'):
    plt.ylabel('CCE')  
#predicting the output
pred_array = np.empty((0,1))
for x in X:
    test.feed_forward(x)
    pred_array = np.append(pred_array, np.array([test.output]), axis=0)
    
#encoding output to either 0 or 1
for index in range(4):
    if((index == 0) or (index == 3)):
         if(pred_array[index,0] <= 0.5):
             pred_array[index,0] = 0
         else:
             pred_array[index,0] = 1
    if((index == 1) or (index == 2)):
        if(pred_array[index,0] > 0.5):
             pred_array[index,0] = 1
        else:
             pred_array[index,0] = 0
             
#compute and print the accuracy score
print('The accuracy score is:', accuracy_score(labels, pred_array))
#generate and plot the confusion matrix
conf_matrix = confusion_matrix(labels, pred_array)
sns.heatmap(conf_matrix, annot = True)
plt.xlabel('Predicted')
plt.ylabel('True')
#print the classification report 
print(classification_report(labels, pred_array)) 
    


