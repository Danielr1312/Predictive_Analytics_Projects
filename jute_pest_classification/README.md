# Table of Contents
- [Table of Contents](#table-of-contents)
- [Jute Pest Classification Project](#jute-pest-classification-project)
  - [Dataset and Dataset Info](#dataset-and-dataset-info)
  - [Project Structure](#project-structure)
  - [Installation and Usage](#installation-and-usage)
    - [Environment Setup](#environment-setup)
    - [Usage](#usage)
  - [Future Work](#future-work)

# Jute Pest Classification Project

## Dataset and Dataset Info

This project focuses on classifying various jute pests using a dataset of pest images to aid in agricultural pest management. The [Jute Pest dataset](https://archive.ics.uci.edu/dataset/920/jute+pest+dataset) consists of images categorized by pest type, stored in `train`, `val`, and `test` directories. Each category corresponds to a specific pest species, making it a multi-class classification problem. The dataset aims to enable automated pest detection using advanced image classification techniques.

## Project Structure

The project uses transfer learning with a pre-trained ResNet-18 model to classify jute pests. Below is the workflow:

1. **Data Preparation**: Load the dataset from the `train`, `val`, and `test` directories and apply data augmentation techniques such as random rotation and flipping to improve model generalization.
2. **Model Implementation**: Implemented a CNN with three convolutional layers, achieving initial classification results before moving to transfer learning using a pre-trained ResNet-18 model to adapt to the pest classification task.
3. **Evaluation**: Assess the model’s performance using metrics like accuracy and the confusion matrix.
4. **Visualization**: Generate visualizations such as the confusion matrix and ROC curves to interpret model performance.

## Installation and Usage

### Environment Setup

For environment setup, refer to the README.md file in the parent folder: [Predictive_Analytics_Projects](../README.md).

### Usage

* To view the code, open [main.ipynb](Predictive_Analytics_Projects/jute_pest_classification/main.ipynb) in your Jupyter Notebook or IDE.
* Ensure you have all necessary dependencies installed. Install the required packages using:
  ```bash
  pip install -r requirements.txt
  ```
* Run all cells in the notebook to execute the complete workflow, including data loading, model training, evaluation, and visualization.

## Future Work

While the current model demonstrates strong performance, achieving a test accuracy of 96.83%, there are several areas for potential improvement:

1. **Advanced Data Augmentation**: Incorporate advanced techniques like CutMix or MixUp to enhance model robustness to varied pest images.
2. **Hyperparameter Optimization**: Explore hyperparameter tuning techniques such as grid search or Bayesian optimization to identify the optimal learning rate, batch size, and regularization parameters.
3. **Deeper Architectures**: Experiment with more advanced models such as ResNet-50, EfficientNet, or Vision Transformers to capture more complex image features.
4. **Class Imbalance Handling**: Address any class imbalances by oversampling underrepresented classes or applying class weights during training.
5. **Explainability**: Utilize tools like Grad-CAM to visualize which parts of the images the model focuses on for classification.
6. **Ensemble Methods**: Combine multiple models through ensemble techniques to potentially improve performance further.
7. **Real-Time Deployment**: Integrate the model into a real-time pest detection system for use in agricultural fields.

By addressing these areas, this project can contribute further to the automation and efficiency of pest management in agriculture. Contributions and suggestions are welcome to enhance the project further.

