import torch

class StopCriteria(object):
    def __init__(self, patience = 5):
        self.patience = patience
        self.best = None
        self.num_higher_epochs = 0
        self.is_better = lambda a, best: a < best
        
        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False
            
    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        if torch.isnan(metric):
            return True

        if self.is_better(metric, self.best):
            self.num_bad_epochs = 0
            self.best = metric
        else:
            self.num_higher_epochs += 1

        if self.num_higher_epochs >= self.patience:
            return True

        return False
    
def ConfusionMatrix(model, output, target):
    pred = model.prediction(output)
    targets = model.prediction(target)
    
    pred = torch.unsqueeze(pred, 1)
    targets = torch.unsqueeze(targets, 1)
    
    combined = torch.cat((pred, targets), 1)
    #comb_transpose = torch.transpose(combined) # we get (pred, target) pairs as we go down the rows
    
    CM = torch.zeros(3,3, dtype = torch.long)
    for i in range(len(pred)):
        if combined[i, 0] == combined[i,1]:
            CM[combined[i, 0] - 1, combined[i, 0] - 1] += 1
        else:
            CM[combined[i, 0] - 1, combined[i, 1] - 1] += 1
                    
    return CM