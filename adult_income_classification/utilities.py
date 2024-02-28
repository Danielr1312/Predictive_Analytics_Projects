import pandas as pd
import numpy as np
from scipy.sparse import issparse


def check_preprocessed_data(preprocessor, X, numerical_features, categorical_features):
    # try:
    # Attempt to transform the data using the preprocessor
    # This will raise an error if the preprocessor has not been fitted
    X_transformed = preprocessor.transform(X)
    
    # If your transformer outputs a dense array, convert it to a DataFrame for easier handling
    # Adjust the column extraction if your preprocessing does not align with this structure
    if issparse(X_transformed):  # Checks if the output is a sparse matrix
        X_transformed = X_transformed.toarray()

    # get column names for the transformed data
    transformed_feature_names = preprocessor.named_transformers_['categorical'].named_steps['encoder'].get_feature_names_out(categorical_features)
    transformed_df = pd.DataFrame(X_transformed, columns=numerical_features + transformed_feature_names.tolist())

    # Check for NaN values
    if transformed_df.isnull().any().any():
        print("Warning: Preprocessed data contains NaN values.")
    else:
        print("Passed: No NaN values found in preprocessed data.")
    
    # Check for infinite values
    if np.isinf(transformed_df.values).any():
        print("Warning: Preprocessed data contains infinite values.")
    else:
        print("Passed: No infinite values found in preprocessed data.")
    
    # Check for constant columns (zero variance)
    if (transformed_df.var() == 0).any():
        print("Warning: Preprocessed data contains constant columns.")
    else:
        print("Passed: No constant columns found in preprocessed data.")

    # except:
    #     # If the preprocessor has not been fitted, return an empty DataFrame
    #     transformed_df = pd.DataFrame()
    #     print("Warning: Preprocessor has not been fitted. Returning an empty DataFrame.")

    return transformed_df