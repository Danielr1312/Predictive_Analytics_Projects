# Wine Classification Artificial Neural Network

# Table of Contents
- [Wine Classification Artificial Neural Network](#wine-classification-artificial-neural-network)
- [Table of Contents](#table-of-contents)
  - [Dataset and Dataset Info](#dataset-and-dataset-info)
  - [Project Goal and Structure](#project-goal-and-structure)
  - [Installation and Usage](#installation-and-usage)
    - [Environment Setup](#environment-setup)
    - [Usage](#usage)
  - [Project History and Future Work](#project-history-and-future-work)
    - [History](#history)
    - [Future Work](#future-work)


## Dataset and Dataset Info

This former homework assignment uses an wine dataset for predicting origin of wines (3 classes) using chemical analysis information. The dataset is obtained from the UC Irvine Machine Learning Repository which can be found [here](https://archive.ics.uci.edu/dataset/109/wine). The dataset contains 13 numerical features without missingness. 


## Project Goal and Structure

  The goal of this code was to perform hyperparameter experimentation and find the best performing model on some metrics. The structure of main.ipynb is as follows: 

1. **Data Preprocessing**: Scaling and splitting the data.
2. **Global Parameters**: Used for determining whether to run or evaluate the experiments and modifying the experiments
3. **Experiments**: Trains models for each of 0, 1, or 2 hidden layers for various other hyperparameter in the parameter grid 'params'.
4. **Best Model**: Loads experimental results, identifies best model in terms of accuracy and loss, and reports findings.


## Installation and Usage

### Environment Setup

For information on environment setup please see the README.md file in the parent folder: [Predictive_Analytics_Projects](Predictive_Analytics_Projects/)

### Usage

 * To view the code, open [main.ipynb](Predictive_Analytics_Projects/wine_classification_ann/main.ipynb) in your IDE
 * Ensure that you have the ```ucimlrepo``` package installed. If not, you can run ```!pip3 install -U ucimlrepo ``` inside the Jupyter Notebook
 * All but one parameter can be set from the section **Global Parameters**
   * ```RUN_EXPERIMENTS``` determines whether to run the experiments or just identify the best model from stored results
   * Experimental parameters can be modified in ```experiment_params```
     * Specifically, the parameter grid tested for each experiment can be changed by modifying the respective nested dictionary 'params'
 * To run the code, just use Run All


## Project History and Future Work

### History

This code was originally used as the first homework assignment for a graduate course on artificial neural networks (ANNs) in Fall 2020. The goal of the assignment was understand how hyperparameters impacted model training and performance which is why a dataset as simple as the UCI Wine dataset was used. Anyways, this assignment was my first implementation of ANNs, and my first experience with the PyTorch library. Upon digging through old files and finding original Jupyter Notebook, I figured I would upload it to GitHub. However, since I had only just started learning the library at the time (and I had only just started using Python in general the prior semester), the code was a mess and something I would've been embarrased to make public. Long story short, I decided to update it a bit, but maintain the general goal of the code.  

### Future Work

There is a lot I have learned since this code was originally written. Thus, there is a lot that can be modified or optimized. I probably won't change much in the future (especially in terms of the scope of this code), but below are a few things I may change if I find the time.

* Create a script for just training and testing a single model if we don't want to run all the experiments
* Convert main.ipynb to be used solely for viewing the results of the experiments
  * Create a dedicated script for running and saving the experimental results
* Modify experimental result storage to keep track of the best performing model within the experiment loops and only save the best results for each experiment
  * The current method results in a massive JSON file, not-to-mention that running ```.cpu().to_numpy().list()``` on results in order to save the results dictionary of each hyperparameter combination is slow
  * additionally save a file containing experiments that have already been run so that we don't rerun them unless explicity told to
* Increase the flexibility of the experiment running algorithm and the experiment parameter setting and result saving structures
  * e.g. perhaps we want to experiment with different activation functions
* Add plotting methods