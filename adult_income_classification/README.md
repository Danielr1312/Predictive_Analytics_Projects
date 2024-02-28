# Adult Income Classification Project

## Dataset and Dataset Info

This 'just-for-fun' project uses an adult income dataset for predicting whether an individual's income exceeds $50K/year based on census data. The dataset is obtained from the UC Irvine Machine Learning Repository which can be found [here](https://archive.ics.uci.edu/dataset/2/adult). The dataset contains various features such as age, education, occupation, and hours per week. This analysis aims to uncover significant predictors of income level and to develop a predictive model.

## Project Structure

This notebook follows a structured approach to data analysis, covering data cleaning, exploratory data analysis, feature engineering, and model building to predict adult income levels. Below is the structure of the analysis:

1. **Data Cleaning**: Handling missing values and removing irrelevant information.
2. **Exploratory Data Analysis (EDA)**: Analyzing distributions, relationships, and trends in the data.
3. **Feature Engineering**: Creating new features that might improve the model's predictive power.
4. **Model Building**: Developing a machine learning model to predict whether an individual's income exceeds $50K per year.
5. **Evaluation**: Assessing the model's performance and identifying areas for improvement.

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

## Installation and Usage

### Environment Setup

This project code was written with Python 3.11.8. To run this code yourself, setup your virtual environment through Anaconda (or venv if you prefer) using the following commands:
```bash
# Create a new conda environment named 'myenv' with Python version 3.11.8
conda create --name myenv python=3.11.8

# Activate the conda environment
conda activate myenv
```
Once your environment is activated, navigate to the project directory where `requirements.txt` is located and run:
```bash
pip install -r requirements.txt
```

### Jupyter Notebook

Ensure you have Jupyter Notebook installed to run this analysis. If not, you can install it using pip:
```bash
pip install notebook
```

To start the Jupyter Notebook, run:

```bash
jupyter notebook
```

Navigate to the project directory and open the `main.ipynb` file to view the analysis.
