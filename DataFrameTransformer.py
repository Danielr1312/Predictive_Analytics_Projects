from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from scipy.sparse import issparse

# Custom Transformer to convert to a dataframe and relabel the columns
class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor, numerical_features, categorical_features):
        self.preprocessor = preprocessor
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features # combining the binary and categorical features

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        # Check if the input is a sparse matrix and convert to dense if necessary
        if issparse(X):
            X = X.toarray()

        # Extract the feature names from the preprocessor
        new_feature_names = self.numerical_features

        # Categorical pipeline
        if 'categorical' in self.preprocessor.named_transformers_:
            transformer = self.preprocessor.named_transformers_['categorical']
            
            # Check if the inner transformer is a pipeline and extract the encoder
            if hasattr(transformer, 'named_steps') and 'encoder' in transformer.named_steps:
                encoder = transformer.named_steps['encoder']
            else:
                encoder = transformer

            for idx, feature in enumerate(self.categorical_features):
                # Retrieve the catigories for this feature from the encoder
                categories = encoder.categories_[idx]

                if encoder.drop_idx_ is not None and encoder.drop_idx_[idx] == 0:
                    feature_categories = categories[1:]
                else:
                    feature_categories = categories

                new_feature_names.extend([f"{feature}_{cat}" for cat in feature_categories])
                # if len(category) == 2: # and self.preprocessor.named_transformers_['categorical']['encoder'].drop == 'if_binary':
                #     new_feature_names.append(f"{original_feature}_{category[1]}") # if binary, only keep the second category since the first category is the reference category
                # else:
                #     new_feature_names.extend([f"{original_feature}_{cat}" for cat in category])

        return pd.DataFrame(X, columns=new_feature_names)