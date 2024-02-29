import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, num_hidden): # here we can play with the number of hidden layers 
        super(Network, self).__init__()
        self.num_hidden = num_hidden
        
        # Inputs to hidden linear combination
        self.hidden = nn.Linear(13, self.num_hidden) # We have 13 predictors
        # hidden to output layer, 3 classes - one for each cultivar
        self.output = nn.Linear(self.num_hidden, 3) # We have 3 target classes to predict
        
        # Defining activation functions
        self.sigmoid = nn.Sigmoid() # for PyTorch it knows that calling nn.Sigmoid requires its derivative as well
        # it creates something called dynamic computational graph which is a graph of the gradient (it computes every epoch)
        
    def forward(self, x):
        z1 = self.hidden(x)
        out1 = self.sigmoid(z1)
        z2 = self.output(out1)
        #out2 = self.sigmoid(z2)
        
        return z2
    
    def prediction(self, output):
        preds = torch.zeros(1,output.shape[0]).flatten().long()
        for i in range(len(preds)):
            index = torch.argmax(output[i,:])
            
            if index == 0:
                preds[i] = 1
            elif index == 1:
                preds[i] = 2
            else:
                preds[i] = 3
                
        return preds