import pandas as pd

### Functions for label encoding, one-hot encoding and k-1

# Ordinal Encoding: for ordinal categorical variables => has ordinal sense
def ordinal_encoding(df, column, categories_order=None):
    if categories_order is None:
        categories_order = sorted(df[column].unique())
    ordinal_map = {cat: idx for idx, cat in enumerate(categories_order)}
    df[column] = df[column].map(ordinal_map)
    return df

# K-1 Encoding: for categorical variables with more than 2 categories => non-ordinal, but drop one category
def k1_encoding(df, column):
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
    df = df.drop(columns=[column])
    df = pd.concat([df, dummies], axis=1)
    return df