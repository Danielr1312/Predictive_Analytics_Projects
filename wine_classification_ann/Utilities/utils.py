import torch
from itertools import product
import re

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
    
def ConfusionMatrix(pred, targets):
    
    # convert targets from one-hot encoding to labels if necessary
    if targets.shape[1] > 1:
        targets = torch.argmax(targets, 1)

    # Get the number of unique classes from the targets tensor
    num_classes = max(targets.max(), pred.max()) + 1

    # Initialize the confusion matrix
    CM = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    # Fill the confusion matrix
    for t, p in zip(targets, pred):
        CM[t,p] += 1

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

def find_best_model(experiments_dict, accuracy_metric='test_accuracy', loss_metric='test_loss'):

    valid_loss_keys = ['train_loss', 'val_loss', 'test_loss']
    valid_accuracy_keys = ['training_accuracy', 'validation_accuracy', 'test_accuracy']

    if accuracy_metric not in valid_accuracy_keys or loss_metric not in valid_loss_keys:
        raise ValueError("Invalid accuracy or loss metric")
    
    perfect_models = []  # To store param_ids of models with perfect accuracy
    best_loss = float('inf')  # Start with the worst case
    best_param_id = None  # To store the best parameter id
    best_experiment = None  # To store the best experiment name

    # First, collect all models with perfect accuracy
    for experiment_name, results_list in experiments_dict.items():
        for result_dict in results_list:
            for param_id, results in result_dict.items():
                if results[accuracy_metric] == 1.0:  # Assuming perfect accuracy is represented as 1.0
                    perfect_models.append((experiment_name, param_id, results[loss_metric]))

    # If there are models with perfect accuracy, find the one with the lowest test loss
    if perfect_models:
        for experiment_name, param_id, loss in perfect_models:
            if loss < best_loss:
                best_loss = loss
                best_param_id = param_id
                best_experiment = experiment_name
                best_accuracy = 1.0  # Perfect accuracy
    else:
        # If no models have perfect accuracy, find the best accuracy and corresponding loss as before
        best_accuracy = -1  # Start with the worst case
        for experiment_name, results_list in experiments_dict.items():
            for result_dict in results_list:
                for param_id, results in result_dict.items():
                    if results[accuracy_metric] > best_accuracy:
                        best_accuracy = results[accuracy_metric]
                        best_loss = results[loss_metric]
                        best_param_id = param_id
                        best_experiment = experiment_name

    if best_experiment is None or best_param_id is None:
        print(f"Current best experiment: {best_experiment}, best param_id: {best_param_id}, best accuracy: {best_accuracy}, best loss: {best_loss}")
        raise ValueError("Best experiment and/or parameter ID is None")

    return best_experiment, best_param_id, best_accuracy, best_loss, len(perfect_models) > 0