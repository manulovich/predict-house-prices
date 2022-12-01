import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List
from sklearn.preprocessing import normalize

def split_numerical_categorical_features(df: pd.DataFrame):
    """
    Получение численных и категориальных признаков из DataFrame
    """
    
    df = df.copy()
    
    numerical_features = list()
    categorical_features = list()

    for column in df.head(1):
        if df.head(1)[column].dtype == 'object':
            categorical_features.append(column)
        else:
            numerical_features.append(column)
    
    return numerical_features, categorical_features

def flating(list_with_nested_lists: np.ndarray) -> list:
    """
    Функия для убирания вложенности np.ndarray
    """
    
    flating_list = list()
    
    for item in list_with_nested_lists:
        if type(item) == np.ndarray:
            flating_list += flating(item)
        else:
            flating_list.append(item)
    
    return flating_list

def ohencoder(df: pd.DataFrame, columns: List[str], one: OneHotEncoder = None) -> Tuple[pd.DataFrame, OneHotEncoder, List[str]]:
    """
    OneHotEncoder
    
    Переделенная функция для возможности кодирования валидационных и тестовых данных после обучения на обучающей выборке
    """
    
    df = df.copy()
    
    index = df.index
    
    if one == None:
        one = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
        ohe_tmp = one.fit_transform(df[columns])
    else:
        ohe_tmp = one.transform(df[columns])
        
    col_names = one.get_feature_names_out(input_features = columns)
    df = df.drop(columns, axis=1)
    df = df.reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame(ohe_tmp, columns=col_names)], axis = 1)
    df = df.set_index(index)
    return (df, one, col_names)

def normalize_df(df: pd.DataFrame):
    """
    Функция для нормализации DataFrame
    """
    
    df.copy()
    return pd.DataFrame(normalize(df), columns=df.columns)