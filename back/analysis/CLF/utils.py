# utils.py
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import pickle
from django.core.cache import cache


def infer_and_convert_data_types(df):
    """
    增强版的数据类型推断函数
    """
    result_types = {}
    
    for column in df.columns:
        # 获取非空值
        non_null_values = df[column].dropna()
        
        if len(non_null_values) == 0:
            df[column] = df[column].astype('object')
            continue

        # 首先尝试转换为整数
        try:
            if pd.to_numeric(non_null_values, downcast='integer').dtype.name.startswith('int'):
                df[column] = pd.to_numeric(df[column], downcast='integer')
                continue
        except (ValueError, TypeError):
            pass

        # 尝试转换为浮点数
        try:
            if any('.' in str(x) for x in non_null_values):
                df[column] = pd.to_numeric(df[column], errors='coerce')
                if not df[column].isna().all():
                    continue
        except (ValueError, TypeError):
            pass

        # 尝试转换为日期
        try:
            df[column] = pd.to_datetime(df[column], errors='coerce')
            if not df[column].isna().all():
                continue
        except Exception:
            pass

        # 检查是否为布尔值
        if non_null_values.isin(['True', 'False', True, False, 1, 0, '1', '0']).all():
            df[column] = df[column].map({'True': True, 'False': False, 
                                       '1': True, '0': False,
                                       1: True, 0: False})
            continue

        # 检查是否应该为分类型
        unique_ratio = len(non_null_values.unique()) / len(non_null_values)
        if unique_ratio < 0.5 and len(non_null_values) > 10:
            df[column] = pd.Categorical(df[column])
            continue

        # 默认保持为字符串类型
        df[column] = df[column].astype('object')

    return df

def generate_file_id():
    return str(uuid.uuid4())

def save_dataframe(file_id, df):
    # 使用Django的缓存系统存储DataFrame
    cache.set(file_id, pickle.dumps(df), timeout=3600)  # 1小时过期

def get_dataframe(file_id):
    # 从缓存中获取DataFrame
    df_bytes = cache.get(file_id)
    if df_bytes is None:
        raise ValueError('File not found or expired')
    return pickle.loads(df_bytes)