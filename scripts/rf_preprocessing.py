"""Fold-fitted preprocessing for the COMMON_COVARIATES_V2 random forests."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


def category_codes(values):
    """Convert validated integer category codes to strings; preserve missingness."""
    return (pd.DataFrame(values).astype("Int64").astype("string")
            .fillna("Missing").to_numpy(dtype=object))


def make_preprocessor(numeric_columns, categorical_columns):
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median", add_indicator=False,
                                  keep_empty_features=True), numeric_columns),
        ("categorical", Pipeline([
            ("codes", FunctionTransformer(category_codes, validate=False,
                                           feature_names_out="one-to-one")),
            ("one_hot", OneHotEncoder(drop=None, handle_unknown="ignore",
                                       sparse_output=False)),
        ]), categorical_columns),
    ], remainder="drop", verbose_feature_names_out=False)
