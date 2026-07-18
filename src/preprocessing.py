import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from config import (
    TEST_SIZE,
    RANDOM_STATE,
    REMOVE_DUPLICATES,
    FILL_NUMERIC_METHOD,
    FILL_CATEGORICAL_METHOD,
    USE_SMOTE,
)


def load_data(file_path):
    return pd.read_csv(file_path)


def remove_duplicates(df):
    if REMOVE_DUPLICATES:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def fill_missing_values(df):
    df = df.copy()

    for col in df.columns:

        if pd.api.types.is_numeric_dtype(df[col]):

            if FILL_NUMERIC_METHOD == "median":
                df[col] = df[col].fillna(df[col].median())

            elif FILL_NUMERIC_METHOD == "mean":
                df[col] = df[col].fillna(df[col].mean())

        else:

            if FILL_CATEGORICAL_METHOD == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])

    return df


def apply_smote(X_train, y_train):
    if not USE_SMOTE:
        return X_train, y_train

    smote = SMOTE(random_state=RANDOM_STATE)

    return smote.fit_resample(
        X_train,
        y_train,
    )


def preprocess(file_path, target_column):

    df = load_data(file_path)

    df = remove_duplicates(df)

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column,
    )

    X_train = fill_missing_values(X_train)
    X_test = fill_missing_values(X_test)

    X_train, y_train = apply_smote(
        X_train,
        y_train,
    )

    return X_train, X_test, y_train, y_test