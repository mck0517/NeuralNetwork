#simple neural network test using mnist 

import numpy 

#for sigmoid function expit() 
import scipy.special 

#for matrix visualization 
import matplotlib.pyplot

get_ipython().magic('matplotlib inline')

#neural network class definition 
class neuralNetwork:
    
    #initialization for neural network  
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
         
        #setting for input, hidden, output node numbers 
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        #weight for matrix: wih, who
        #ex: w_i_j means connecting from node i to node j 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
        #setting for learning rate  
        self.lr = learningrate
        
        #setting for activation function using sigmoid function 
        self.activation = lambda x: scipy.special.expit(x)
    
    #training for neural network 
    def train(self, inputs_list, targets_list):
        
        #convert input list to 2-dimension matrix 
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #calculate signal for input`s hidden layer, H_input : W_input_hidden x I   
        hidden_inputs = numpy.dot(self.wih, inputs)
        
        #calculate signal for output`s hidden layer, H_output : sigmoid(hidden_inputs)
        hidden_outputs = self.activation(hidden_inputs)
        
        #calculate signal for input`s final layer, F_input : W_hidden_output x H_output
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        #calculate signal for output`s final layer, F_output : sigmoid(final_inputs)
        final_outputs = self.activation(final_inputs)
        
        #calculate error in output layer
        output_errors = targets-final_outputs
        
        #calculate error in hidden layer
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #weight update between hidden layer and output layer  
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))
    
        #weight update between input layer and hidden layer
        self.wih += self.lr * numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))
        
    #query for neural network  
    def query(self, inputs_list):
        
        #convert input list to 2-dimension matrix  
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        #calculate signal for input`s hidden layer, H_input : W_input_hidden x I  
        hidden_inputs = numpy.dot(self.wih, inputs)
        
        #calculate signal for output`s hidden layer, H_output : sigmoid(hidden_inputs)
        hidden_outputs = self.activation(hidden_inputs)
        
        #calculate signal for input`s final layer, F_input : W_hidden_output x H_output
        final_inputs = numpy.dot(self.who, hidden_outputs)
        
        #calculate signal for output`s final layer, F_output : sigmoid(final_inputs)
        final_outputs = self.activation(final_inputs)
        
        return final_outputs

#training using mnist data set
#input node need 784 nodes(image size: 28 x 28 = 784), fixed value     
input_nodes = 784 

#hidden node less more than input node, tunning value 
hidden_nodes = 200 

#output node need 10 nodes(number: 0~9), fixed value 
output_nodes = 10 

#learning rate: tunning value(range: 0.1~0.9)  
learning_rate = 0.1 

#instance for neural network 
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)   
 
#mnist_train.csv file loading using list 
trainig_data_file = open("mnist_dataset\mnist_train.csv", 'r')
trainig_data_list = trainig_data_file.readlines()
trainig_data_file.close()

#train neural network 
epochs = 5 #setting aepochs  

for e in range(epochs):
    
    #search all records in training data collection  
    for record in trainig_data_list: 
        
        #seperate records using comma 
        all_values = record.split(',')
        
        #scaling range & value for input values(from 0~255 to 0.01~0.99) 
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        
        #setting for result values(real value: 0.99, rest values: 0.01) 
        targets = numpy.zeros(output_nodes)+0.01
        
        #all_values[0] means result values in this record 
        targets[int(all_values[0])] = 0.99
        
        #traing 
        n.train(inputs, targets) 

#mnist_test.csv file loading using list 
test_data_file = open("mnist_dataset\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

#test neural network 
#initialization list(for result score)
scorecard = []

#search all records in test data collection  
for record in test_data_list:
    
    #seperate records using comma 
    all_values = record.split(',')
    
    #first index mean correct label 
    correct_label = int(all_values[0])
    
    #scaling range & value for input values(from 0~255 to 0.01~0.99) 
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    
    #query for neural network
    outputs = n.query(inputs)
    
    #index for high value same index of label  
    label = numpy.argmax(outputs)
    
    #add correct or incorrect label to list   
    if(label == correct_label):
        
        #in case of correct label, add 1 to list  
        scorecard.append(1)
        
    else:
        #in case of incorrect label, add 0 to list  
        scorecard.append(0)
         
#calculate for performance 
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum()/scorecard_array.size)
    
    

