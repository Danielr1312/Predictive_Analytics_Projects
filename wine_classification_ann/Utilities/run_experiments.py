import torch
import numpy as np
from Networks.Network import Network
from Utilities.utils import *
import time


def train_model(model, num_epochs, learning_rate, X_train, y_train, X_val, y_val, patience = 10, verbose = False):
    """
    Trains a model on the provided training data and evaluates it on the provided validation data.
    """

    # Initialize the loss function, optimizer, and stop criteria
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    stop_criteria = StopCriteria(patience)

    if verbose:
        # print(f"Training {model} for {num_epochs} epochs with learning rate {learning_rate} and patience {patience}...")
        model.eval()
        output = model.forward(X_train).squeeze()
        loss = criterion(output, torch.max(y_train, 1)[1])
        print(f"Initial training loss: {loss}")


    # Train the model
    training_losses = []
    validation_losses = []
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model.forward(X_train)
        loss = criterion(output.squeeze(), torch.max(y_train, 1)[1])
        training_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # Evaluate the model on the validation data
        model.eval()
        with torch.no_grad():
            val_output = model.forward(X_val)
            val_loss = criterion(val_output.squeeze(), torch.max(y_val, 1)[1])
            validation_losses.append(val_loss.item())

        # Check if we should stop training
        if stop_criteria.step(val_loss):
            # if verbose:
            #     print(f"Stopping early on epoch {epoch} with training loss {loss} and validation loss {val_loss}"); print()
            end_time = time.time()
            break

    total_time = end_time - start_time

    # Evaluate the model on the training data
    train_cm = ConfusionMatrix(model, output, y_train)
    val_cm = ConfusionMatrix(model, val_output, y_val)

    return model, training_losses, validation_losses, train_cm, val_cm, epoch + 1, total_time

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a model on the provided test data.
    """

    model.eval()
    with torch.no_grad():
        test_output = model.forward(X_test)
        test_loss = torch.nn.CrossEntropyLoss()(test_output.squeeze(), torch.max(y_test, 1)[1])
        test_cm = ConfusionMatrix(model, test_output, y_test)

    return test_loss, test_cm
