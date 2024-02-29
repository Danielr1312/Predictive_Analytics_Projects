import torch
from torch import nn


class Network(nn.Module):
    """
    A flexible neural network class that can be used to create a variety of network architectures 
    ranging from simple linear models to more complex multi-layer perceptrons.
    
    Parameters:
        n_inputs (int): The number of input features for the network.
        n_outputs (int): The number of output classes for the network.
        num_hidden_layers (int, optional): The number of hidden layers in the network. 
                                            Defaults to 0, which results in a linear model.
        hidden_sizes (list of int, optional): A list containing the sizes of each hidden layer. 
                                              The length of this list should match num_hidden_layers. 
                                              Defaults to an empty list, and if num_hidden_layers is 
                                              greater than 0 but hidden_sizes is empty, each hidden 
                                              layer defaults to a size of 10 neurons.
    
    Methods:
        forward(x): Performs a forward pass through the network with input x.
                    Parameters:
                        x (Tensor): A tensor containing input data.
                    Returns:
                        Tensor: The network's output tensor.
                        
        prediction(output): Converts the network's output tensor into predicted class labels.
                            Parameters:
                                output (Tensor): A tensor containing the network's raw output.
                            Returns:
                                Tensor: A tensor containing predicted class labels starting from 1.
    """
    def __init__(self, n_inputs, n_outputs, num_hidden_layers=0, hidden_sizes=[]):
        super(Network, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        
        layers = []
        last_size = n_inputs  # Set the size of the last layer to the input size initially
        
        # Create hidden layers if num_hidden_layers > 0
        if num_hidden_layers > 0:
            for i in range(num_hidden_layers):
                # Use the provided hidden size for each layer, default to 10 if not enough sizes provided
                hidden_size = hidden_sizes[i] if i < len(hidden_sizes) else 10
                layers.append(nn.Linear(last_size, hidden_size))
                layers.append(nn.Sigmoid())
                last_size = hidden_size  # Update last_size to the current hidden size
        
        # Output layer
        layers.append(nn.Linear(last_size, n_outputs))
        
        # Combine all layers
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
    def prediction(self, output):
        preds = torch.argmax(output, dim=1) + 1  
        return preds

# class Network(nn.Module):
#     def __init__(self, num_hidden): # here we can play with the number of hidden layers 
#         super(Network, self).__init__()
#         self.num_hidden = num_hidden
        
#         # Inputs to hidden linear combination
#         self.hidden = nn.Linear(13, self.num_hidden) # We have 13 predictors
#         # hidden to output layer, 3 classes - one for each cultivar
#         self.output = nn.Linear(self.num_hidden, 3) # We have 3 target classes to predict
        
#         # Defining activation functions
#         self.sigmoid = nn.Sigmoid() # for PyTorch it knows that calling nn.Sigmoid requires its derivative as well
#         # it creates something called dynamic computational graph which is a graph of the gradient (it computes every epoch)
        
#     def forward(self, x):
#         z1 = self.hidden(x)
#         out1 = self.sigmoid(z1)
#         z2 = self.output(out1)
#         #out2 = self.sigmoid(z2)
        
#         return z2
    
#     def prediction(self, output):
#         preds = torch.zeros(1,output.shape[0]).flatten().long()
#         for i in range(len(preds)):
#             index = torch.argmax(output[i,:])
            
#             if index == 0:
#                 preds[i] = 1
#             elif index == 1:
#                 preds[i] = 2
#             else:
#                 preds[i] = 3
                
#         return preds
    
# class Network2(nn.Module):
#     def __init__(self, num_hidden1, num_hidden2):
#         super(Network2, self).__init__()
#         self.num_hidden1 = num_hidden1
#         self.num_hidden2 = num_hidden2
        
#         # Inputs to hidden linear combination
#         self.hidden1 = nn.Linear(13, self.num_hidden1)
#         # hidden to output layer, 3 classes - one for each cultivar
#         self.hidden2 = nn.Linear(self.num_hidden1, self.num_hidden2)
#         self.out = nn.Linear(self.num_hidden2, 3)
        
#         # Defining activation functions
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         z1 = self.hidden1(x)
#         out1 = self.sigmoid(z1)
#         z2 = self.hidden2(out1)
#         out2 = self.sigmoid(z2)
#         z3 = self.out(out2)
        
#         return z3
    
#     def prediction(self, output):
#         preds = torch.zeros(1,output.shape[0]).flatten().long()
#         for i in range(len(preds)):
#             index = torch.argmax(output[i,:])
            
#             if index == 0:
#                 preds[i] = 1
#             elif index == 1:
#                 preds[i] = 2
#             else:
#                 preds[i] = 3
                
#         return preds
    
# class Network3(nn.Module):
#     def __init__(self):
#         super(Network3, self).__init__()
        
#         # Inputs to outputs linear combination
#         self.out = nn.Linear(13, 3)
        
#         # Defining activation functions
#         self.sigmoid = nn.Sigmoid()
        
#     def forward(self, x):
#         z1 = self.out(x)
        
#         return z1
    
#     def prediction(self, output):
#         preds = torch.zeros(1,output.shape[0]).flatten().long()
#         for i in range(len(preds)):
#             index = torch.argmax(output[i,:])
            
#             if index == 0:
#                 preds[i] = 1
#             elif index == 1:
#                 preds[i] = 2
#             else:
#                 preds[i] = 3
                
#         return preds