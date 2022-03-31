import numpy as np
#Neural Network Project implementation
#By: Edgar A. Rodriguez
#Date: 11/22/2020

#################################
#Python function implementations#
#################################
#returns the sigmoid activation function value
def sigmoid(x):
    return 1/(1 + np.exp(-x))
#returns the derivative value of the sigmoid activation function  
def dsigmoid(x): 
    return sigmoid(x) * (1 - sigmoid(x))
#returns the Relu activation function value
def Relu(x):
    if x <= 0:
        return 0
    else:
        return x
#returns the derivative value of the Relu activation function
def dRelu(x):
    if x <= 0:
        return 0
    else:
        return 1
#####################################
#Neural Network class implementation#
#####################################
class net:
    def __init__(self, input_size, alpha = 0.1, mu = 0, loss = 'MSE'):
        '''
        Neural Network Initialization
        Parameters
        ----------
        input_size : integer
            The input size of the dataset being trained.
        alpha : float, optional
            learning rate parameter. The default is 0.1.
        mu : float, optional
            Momentum parameter used with SGD W/ Momentum . The default is 0.
        loss : string, optional
            The loss function the user wants to use between 'MSE' and 'CCE'. The default is 'MSE'.
        '''
        self.input_size = input_size
        self.alpha = alpha
        self.lossfunction = loss
        self.mu = mu
        #initialize the list containing the layers that the artificial neural network will have
        self.layers = []
        #output of the loss function
        self.loss_value = 0
        #activation output of the last layer given in the form of an array once overwritten
        self.output = 0
    
    def add_layer(self, n, activation = 'ReLU'):
        '''
        Add_layer 
        creates a new layer given an n number of nodes and a activation function
        Parameters
        ----------
        n : integer
            n represents the number of nodes the new layer will have.
        activation : string, optional
            The activation function utilized in that layer
            has the option to choose between 'sigmoid' or 'ReLU'. The default is 'ReLU'.
        '''
        #if this is the first layer being created,
        #make the number of columns for the weights in the new layer be equal
        #to the input size + 1
        if len(self.layers) == 0:
            #initialize the parameters for the new layer being created
            self.layers.append(layer((n,self.input_size+1), activation)) 
            
        #if there is a previous layer already created
        #make the number of columns for the weights in the new layer
        #be equal to the number of nodes in the previous layer + 1
        else:
            #take the index of the previous layer that was created
            index = len(self.layers) - 1
            #initialize the parameters for the new layer being created
            self.layers.append(layer((n,self.layers[index].n + 1), activation))

    def feed_forward(self, xvector):
        '''
        feed_forward
        Performing the feed forward process given an input vector x  
        '''
        #inputx takes the input vector and concatenates it with 1 at the beginning
        inputx = np.concatenate(([[1]],[xvector]),axis = 1)
        
        layersize = len(self.layers)
        
        #feed forwarding the first layer
        #obtaining weighted sum 
        self.layers[0].z = self.layers[0].weights @ inputx.T
        
        #obtaining output a
        #the initial value of matrix a in the first layer is [1]
        self.layers[0].a = np.array([[1]])
        #iterate through each scalar value in the matrix z
        for x in self.layers[0].z:
            #appending activation function outputs to the matrix a in the first layer
            if self.layers[0].activation == 'sigmoid':
                self.layers[0].a = np.append(self.layers[0].a, sigmoid(x))
            if self.layers[0].activation == 'ReLU':
                self.layers[0].a = np.append(self.layers[0].a, Relu(x))
        
        #feed forwarding every other layer besides the first layer        
        for ii in range(1, layersize):
                #obtaining weighted sum  
                self.layers[ii].z = self.layers[ii].weights @ self.layers[ii-1].a  
                #obtaining output a
                #if index ii is the last layer, dont append 1 to the activation function matrix
                if ii == (layersize - 1):
                    self.layers[ii].a = []
                #append 1 at the beginning of the matrix if it is not the last layer
                else:
                    self.layers[ii].a = [1]
                #iterate through each scalar value in the matrix z    
                for x in self.layers[ii].z:
                    #appending activation function outputs to the a matrix at layer index ii 
                    if self.layers[ii].activation == 'sigmoid':
                        self.layers[ii].a = np.append(self.layers[ii].a, sigmoid(x))
                    if self.layers[ii].activation == 'ReLU':
                        self.layers[ii].a = np.append(self.layers[ii].a, Relu(x))
        #getting the activation output value of the last layer
        self.output = self.layers[len(self.layers)-1].a
        #Feed forward process is complete 
    
    def back_propogation(self, xvector,yvector):
        '''
        back_propogation
        performing the back propogation process given an input vector x and an output vector y
        '''
        layerindex = len(self.layers) - 1
        #creating the corresponding input and output vectors 
        inputx = np.concatenate(([[1]],[xvector]),axis = 1)
        outputy = np.concatenate(([[]],[yvector]),axis = 1)
     
        loss_value = 0
        #------------------------------
        #Back propogation for the last layer
        #computing the partial derivative of the cost function (dcost) for the last layer
        #as well as compute the total loss value
        for j in range(outputy.shape[1]):
            #for each element in the output vector, insert the derivative value computed at that index
            if (self.lossfunction == 'MSE'):
                self.layers[layerindex].dcost[j] = 2 *(self.layers[layerindex].a[j] - outputy[0,j])
                loss_value += (outputy[0,j] - self.layers[layerindex].a[j]) * (outputy[0,j] - self.layers[layerindex].a[j])                 
            if (self.lossfunction == 'CCE'):
                self.layers[layerindex].dcost[j] = (self.layers[layerindex].a[j] - outputy[0,j])
                loss_value += (outputy[0,j]) * np.log10(self.layers[layerindex].a[j])
        #Inserting the final loss value result to another variable for graphing purposes
        if (self.lossfunction == 'MSE'):
            self.loss_value = loss_value / outputy.shape[1]
        if(self.lossfunction == 'CCE'):
            self.loss_value = (-1) * (loss_value / outputy.shape[1])
        
        #computing the derivative of the activation function (da) for the last layer
        #iterate through each scalar value in the weight sum matrix z 
        for x in self.layers[layerindex].z:
            #appending derivative activation function outputs to the da matrix in the last layer
            if self.layers[layerindex].activation == 'sigmoid':
                self.layers[layerindex].da = np.append(self.layers[layerindex].da, dsigmoid(x))
            if self.layers[layerindex].activation == 'ReLU':
                self.layers[layerindex].da = np.append(self.layers[layerindex].da, dRelu(x)) 
        
        #turning the da matrix in the current layer to a 2D matrix
        delta1 = np.reshape(self.layers[layerindex].da,(1,self.layers[layerindex].da.shape[0]))
        #element-wise multiplication between da and dcost
        delta = np.multiply(delta1.T,self.layers[layerindex].dcost)
        #turning the activation matrix to a 2D matrix
        a = np.reshape(self.layers[layerindex-1].a,(1,self.layers[layerindex-1].a.shape[0]))
        #computing the gradient for the last layer
        self.layers[layerindex].gradient = delta @ a
        #----------------------------
        #proceed to the previous layer  
        #back propogation for every other hidden layer
        for layerindex in range(len(self.layers) - 2, -1, -1):
            #computing the derivative of the activation function for the corresponding hidden layer
            #iterate through each scalar value in the weight sum matrix z 
            for x in self.layers[layerindex].z:
                #appending derivative activation function outputs to the da matrix in the last layer
                if self.layers[layerindex].activation == 'sigmoid':
                    self.layers[layerindex].da = np.append(self.layers[layerindex].da, dsigmoid(x))
                if self.layers[layerindex].activation == 'ReLU':
                    self.layers[layerindex].da = np.append(self.layers[layerindex].da, dRelu(x)) 
            #computing the partial derivative of the cost function for the corresponding hidden layer
            #j will iterate from 0 up to the number of nodes in the next layer
            #jprime will iterate from 0 up to the number of nodes in the current layer
            for jprime in range(self.layers[layerindex].n):
                for j in range(self.layers[layerindex+1].n):
                    self.layers[layerindex].dcost[jprime] += self.layers[layerindex+1].weights[j,jprime+1] * self.layers[layerindex+1].da[j] * self.layers[layerindex+1].dcost[j]

  
            #turning the da matrix in the current layer to a 2D matrix
            delta1 = np.reshape(self.layers[layerindex].da,(1,self.layers[layerindex].da.shape[0]))
            #element-wise multiplication between da and dcost
            delta = np.multiply(delta1.T,self.layers[layerindex].dcost)
            
            #Computing the gradient for the current hidden layer
            #if the current layer is not the first layer, use the activation function rather than the input vector
            if (layerindex != 0):
               #turning the activation matrix to a 2D matrix
               a = np.reshape(self.layers[layerindex-1].a,(1,self.layers[layerindex-1].a.shape[0]))
               self.layers[layerindex].gradient = delta @ a
            #if the current layer is the first layer, use the input vector and not the activation function matrix
            if (layerindex == 0):
                self.layers[layerindex].gradient = delta @ inputx 
            
            #proceed to the previous layer
            #and repeat back propogation until the for loop is done
            
     
        #cleanup for the next training iteration 
        for layerindex in range(len(self.layers) - 1, -1, -1):
            self.layers[layerindex].da = []
            self.layers[layerindex].dcost = np.zeros((self.layers[layerindex].n,1))
        #back propogation is now complete
       
    def update_weights(self):
        '''
        update_weights
        updating the weights for each layer using SGD  with momentum  
        '''
        for layerindex in range(len(self.layers) - 1, -1, -1):
            newv = (self.mu * self.layers[layerindex].old_v) + (self.alpha * self.layers[layerindex].gradient)
            self.layers[layerindex].weights = self.layers[layerindex].weights - newv
            self.layers[layerindex].old_v = newv
    
############################
#Layer class implementation#
############################  
class layer:
    def __init__(self, shape, activation):
        '''
        Layer Initialization
        class that will contain the attributes for each layer
        Parameters
        ----------
        shape : tuple
            Will be used as the shape for the weight matrix in the current layer.
        activation : string
            activation function utilized in the current layer
            'sigmoid' or 'ReLU' expected
        '''
        #take in the number of nodes in the layer
        self.n = shape[0]
        #take in the shape dimension size corresponding to the layer
        self.shape = shape
        #take in the activation function given
        self.activation = activation
        #randomize weight values given a certain dimension
        self.weights = np.random.randn(*shape)*np.sqrt(2/self.n)
        #feed forward parameters
        self.z = None
        self.a = None
        #back propogation parameters
        self.da = []
        self.dcost = np.zeros((shape[0],1))
        self.gradient = np.zeros(shape)
        self.old_v = np.zeros(shape)

        
        