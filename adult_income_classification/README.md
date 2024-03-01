# Table of Contents
- [Table of Contents](#table-of-contents)
- [Adult Income Classification Project](#adult-income-classification-project)
  - [Dataset and Dataset Info](#dataset-and-dataset-info)
  - [Project Structure](#project-structure)
  - [Installation and Usage](#installation-and-usage)
    - [Environment Setup](#environment-setup)
    - [Usage](#usage)
  - [Future Work](#future-work)


# Adult Income Classification Project

## Dataset and Dataset Info

This 'just-for-fun' project uses an adult income dataset for predicting whether an individual's income exceeds $50K/year based on census data. The dataset is obtained from the UC Irvine Machine Learning Repository which can be found [here](https://archive.ics.uci.edu/dataset/2/adult). The dataset contains 14 features such as age, education, occupation, and hours per week. This analysis aims to uncover significant predictors of income level and to develop a predictive model.

## Project Structure

This notebook follows a structured approach to data analysis, covering data cleaning, exploratory data analysis, feature engineering, and model building to predict adult income levels. Below is the structure of the analysis:

1. **Data Cleaning**: Handling missing values and removing irrelevant information.
2. **Exploratory Data Analysis (EDA)**: Analyzing distributions, relationships, and trends in the data.
3. **Feature Engineering**: Creating new features that might improve the model's predictive power.
4. **Model Building**: Developing a machine learning model to predict whether an individual's income exceeds $50K per year.
5. **Evaluation**: Assessing the model's performance and identifying areas for improvement.

## Installation and Usage

### Environment Setup

For information on environment setup please see the README.md file in the parent folder: [Predictive_Analytics_Projects](../README.md)

### Usage

* To view the code, open [main.ipynb](Predictive_Analytics_Projects/adult_income_classification/main.ipynb) in your IDE
* Ensure that you have the ```ucimlrepo``` package installed. If not, you can run ```!pip3 install -U ucimlrepo ``` inside the Jupyter Notebook
* Global parameters can be found under the heading **Logistic Regression Models**
* Each model type has experiment parameters which can be changed by modifying their respective `param_grid`
* To run the code, just run all

## Future Work

While the current project provides valuable insights into income classification, several improvements and extensions can be made to enhance its capabilities and performance:

1. **Class Imbalance Handling**: Investigate more sophisticated techniques for addressing class imbalance, such as oversampling the minority class or undersampling the majority class during training. This could lead to improved performance metrics, particularly for the minority class.

2. **Encoding Strategy**: Re-evaluate the use of `handle_unknown='ignore'` in our one-hot encoding process. While it helps to manage unknown categories during the prediction phase, it may result in information loss for infrequent categories. Exploring alternative strategies or encoding techniques could help retain valuable information.

3. **Stratified Splits**: Implement stratified splits and/or stratified cross-validation to ensure that each fold of the cross-validation process represents the overall dataset proportionally. This approach could help mitigate the information loss and improve model robustness.

4. **Feature Interactions**: Consider adding polynomial feature interactions within the preprocessor by incorporating `PolynomialFeatures(degree=2, interaction_only=True)`. This could uncover significant interaction effects among input features, potentially boosting model performance. Caution is advised due to the potential increase in model complexity and computational cost, especially after one-hot encoding. Initially, applying this step to a limited set of numerical predictors may be more practical.

5. **Alternative Models**: Explore a variety of machine learning models beyond the linear models currently in use. Non-linear models or ensemble methods may provide improved predictive performance and offer insights into different aspects of the data.

6. **Scoring and Class Weighting**: Further investigate the trade-offs between different scoring metrics and class weighting strategies. Depending on the primary goal of the classifier—whether it's maximizing overall accuracy or focusing on reducing specific types of classification errors—different combinations of scoring metrics and class weighting may be more appropriate.

7. **False Positives vs. True Positives Trade-off**: Delve deeper into the balance between false positives and true positives, especially in scenarios where minimizing one type of error is more critical than minimizing the other. This will involve a careful examination of the costs associated with different types of classification errors in the context of the application.

By addressing these areas, we can potentially improve the model's performance, make it more applicable to different scenarios, and gain deeper insights into the factors influencing income levels. Contributions and suggestions are welcome to help drive these improvements.