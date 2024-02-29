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

def convert_results_to_dict(params, results):
    """
    Converts a list of lists containing results into a dictionary with labels corresponding to the values in params.
    
    Parameters:
        params (dict): A dictionary containing parameter values, with keys 'hidden_size' and 'learning_rate'.
        results (list of lists): A list where each sublist contains results corresponding to a combination of 
                                 'hidden_size' and 'learning_rate' from the params.
    
    Returns:
        dict: A nested dictionary where each 'hidden_size' maps to a sub-dictionary where each 'learning_rate'
              maps to the corresponding result.
    """
    results_dict = {}
    for i, hidden_size in enumerate(params['hidden_size']):
        # Prepare a sub-dictionary for each hidden_size
        hidden_size_results = {}
        for j, learning_rate in enumerate(params['learning_rate']):
            # Check if the current index is within the bounds of the result list to avoid IndexError
            if i < len(results) and j < len(results[i]):
                # Assign each result to its corresponding learning rate
                hidden_size_results[learning_rate] = results[i][j]
        # Convert hidden_size to string to use as a dictionary key
        results_dict[str(hidden_size)] = hidden_size_results
    return results_dict

def convert_all_results(params, results_dict):
    """
    Converts multiple lists of results into dictionaries with labels corresponding to the values in params.
    
    Parameters:
        params (dict): A dictionary containing parameter values, with keys like 'hidden_size' and 'learning_rate'.
        results_dict (dict): A dictionary where each key is a string representing the type of result (e.g., 
                             'training_losses') and each value is a list of lists containing the results.
    
    Returns:
        dict: A nested dictionary where each type of result maps to another dictionary structured by 'hidden_size'
              and 'learning_rate' according to the params.
    """
    converted_results = {}
    for key, result_lists in results_dict.items():
        converted_results[key] = convert_results_to_dict(params, result_lists)
    return converted_results