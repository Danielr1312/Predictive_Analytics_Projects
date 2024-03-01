# Table of Contents
- [Table of Contents](#table-of-contents)
- [Predictive Analytics Projects](#predictive-analytics-projects)
  - [Projects Overview](#projects-overview)
  - [Environment Setup](#environment-setup)
  - [Work in Progress](#work-in-progress)
  - [Contributing](#contributing)
  - [Collaboration](#collaboration)
  - [Contact Information](#contact-information)

# Predictive Analytics Projects

Welcome to my collection of predictive analytics projects. This repository is designed to showcase different machine learning and data science techniques applied across various datasets and scenarios. Each subfolder represents an individual project focused on a specific topic or dataset.

## Projects Overview

Below is the list of projects currently included in this repository:

- **[Adult Income Classification](./adult_income_classification)**: This project focuses on predicting whether an individual's income exceeds $50K/year based on census [data](https://archive.ics.uci.edu/dataset/2/adult). It covers data cleaning, exploratory data analysis, feature engineering, and model building.
- **[Wine Classification ANN](./wine_classification_ann)**: This was a former homework assignment utilizing [Wine data](https://archive.ics.uci.edu/dataset/109/wine) which looks into how hyperparameters impact training and performance of artificial neural networks

## Environment Setup

This project code was written with Python 3.11.8. To run this code yourself, you can set up your virtual environment through Anaconda or venv, depending on your preference. Before you attempt this, ensure you have navigate to the project directory where `requirements.txt` and `environment.yml` is located before running the installation commands below. Below are instructions for both methods:

1. Using Conda and `environment.yml`:
If you are using Anaconda, you can create a new environment directly from the `environment.yml` file which should be located in the project directory. This file contains all the necessary packages along with the specific Python version:
```bash
# Create a new conda environment from environment.yml file
conda env create -f environment.yml

# Activate the conda environment
conda activate predictive_analytics
```

2. Using Conda and `requirements.txt`:
Alternatively, if you prefer to create a conda environment and then install Python packages using `pip` and `requirements.txt`, follow these steps:
```bash
# Create a new conda environment named 'myenv' with Python version 3.11.8
conda create --name myenv python=3.11.8

# Activate the conda environment
conda activate myenv

# Install required packages
pip install -r requirements.txt
```

3. Using venv and `requirements.txt`
If you prefer not using Anaconda, you can use `venv` to create a virtual environment:
```bash
# Create a new virtual environment named 'myenv' in the directory 'venv'
python3 -m venv /path/to/new/virtual/environment/myenv

# Activate the virtual environment
# For Windows
myenv\Scripts\activate
# For Unix or MacOS
source myenv/bin/activate

# Install required packages
pip install -r requirements.txt
```
After setting up the environment using any of the above methods, you should be able to run code in any of sub-folders in this repository.

## Work in Progress

This repository is a work in progress, and new projects will be added over time. The aim is to continually expand the collection with diverse datasets and predictive modeling challenges. Keep an eye on this space for updates!

## Contributing

I am always looking to improve and expand the projects in this repository. If you have suggestions, notice any errors, or identify any misuses of data or techniques, please feel free to notify me. You can open an issue in the repository or submit a pull request with your proposed changes.

## Collaboration

Collaboration is welcome! If you're interested in contributing to a project or have ideas for new projects, please reach out. Together, we can make this repository a valuable resource for learning and applying predictive analytics techniques.

## Contact Information

If you need to get in touch, you can contact me through [GitHub issues](https://github.com/Danielr1312/Predictive_Analytics_Projects/issues).

Thank you for visiting and exploring the Predictive Analytics Projects!
