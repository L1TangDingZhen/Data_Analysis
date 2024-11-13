# utils.py
import pandas as pd
import uuid
from django.core.cache import cache
import pickle
import numpy as np
import re
import logging
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def optimize_dataframe(df):
    """优化DataFrame内存使用"""
    
    # 对象列优化
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        
        # 如果唯一值较少，转换为category
        if num_unique / len(df) < 0.5:
            df[col] = df[col].astype('category')
            
    # 数值列优化
    for col in df.select_dtypes(include=['int64']).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # 根据数据范围选择最小的数据类型
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype(np.int16)
                
    # 浮点数列优化
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
        
    return df

def generate_file_id():
    """生成唯一的文件ID"""
    return str(uuid.uuid4())

def save_dataframe(file_id, df):
    """将DataFrame保存到缓存中"""
    cache.set(file_id, pickle.dumps(df), timeout=3600)  # 1小时过期

def get_dataframe(file_id):
    """从缓存中获取DataFrame"""
    df_bytes = cache.get(file_id)
    if df_bytes is None:
        raise ValueError('File not found or expired')
    return pickle.loads(df_bytes)

def generate_preview_data(df):
    """
    生成数据预览
    
    Args:
        df (pd.DataFrame): 要预览的DataFrame
        
    Returns:
        list: 包含前5行数据的字典列表
    """
    preview_data = []
    for _, row in df.head(5).iterrows():
        row_dict = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                row_dict[col] = "No data available"
            elif isinstance(value, (np.datetime64, pd.Timestamp)):
                row_dict[col] = value.strftime('%d/%m/%Y')  # 澳洲日期格式
            elif isinstance(value, (np.floating, float)):
                if np.isnan(value):
                    row_dict[col] = "No data available"
                else:
                    row_dict[col] = f"{float(value):.2f}" if value % 1 != 0 else str(int(value))
            elif isinstance(value, bool):
                row_dict[col] = str(int(value))  # 或者使用 'Yes'/'No'
            else:
                row_dict[col] = str(value).strip()
        preview_data.append(row_dict)
    return preview_data

# utils.py
def generate_sample_value(value):
    """生成一致的样本值显示格式"""
    if pd.isna(value):
        return "No data available"
    elif isinstance(value, (np.datetime64, pd.Timestamp)):
        return value.strftime('%d/%m/%Y')
    elif isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return "No data available"
        return f"{float(value):.2f}" if value % 1 != 0 else str(int(value))
    elif isinstance(value, (np.integer, int)):
        return str(int(value))
    elif isinstance(value, bool):
        # 统一使用 '1' 和 '0' 来表示布尔值
        return '1' if value else '0'
    else:
        str_value = str(value).strip()
        return str_value if str_value else "No data available"

def get_column_sample(df, column):
    """
    获取DataFrame某列的样本值
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): 列名
        
    Returns:
        str: 格式化后的样本值
    """
    try:
        non_null_values = df[column].dropna()
        if not non_null_values.empty:
            return generate_sample_value(non_null_values.iloc[0])
        return "No data available"
    except Exception as e:
        print(f"Error processing sample for column {column}: {str(e)}")
        return "No data available"


def clean_special_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理DataFrame中的特殊值和格式清理
    
    Args:
        df (pd.DataFrame): 输入的DataFrame
        
    Returns:
        pd.DataFrame: 处理后的DataFrame
    """
    try:
        # 创建DataFrame的副本以避免修改原始数据
        df = df.copy()
        
        # 统一空值处理
        null_values: Dict[str, Any] = {
            'NA': pd.NA,
            'N/A': pd.NA,
            'null': pd.NA,
            'NULL': pd.NA,
            'None': pd.NA,
            '-': pd.NA,
            'missing': pd.NA,
            'MISSING': pd.NA,
            'undefined': pd.NA,
            'UNDEFINED': pd.NA,
            'not available': pd.NA,
            'NOT AVAILABLE': pd.NA,
            'nan': pd.NA,
            'NaN': pd.NA,
            '': pd.NA,
        }
        
        # 处理无效数字
        invalid_numbers: List[str] = ['#REF!', '#VALUE!', '#DIV/0!', '#NUM!']
        for num in invalid_numbers:
            null_values[num] = pd.NA
            
        # 替换空值
        df = df.replace(null_values)
        
        # 处理日期时间混合值
        def clean_datetime(value: Any) -> Any:
            """清理日期时间值"""
            if pd.isna(value):
                return value
                
            try:
                value = str(value).strip()
                
                # 日期时间组合的模式
                patterns = [
                    # 处理 "2024-03-12 14:30:00" 格式
                    (r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', r'\1'),
                    # 处理 "12/03/2024 2:30 PM" 格式
                    (r'(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}\s*(?:AM|PM)?', r'\1'),
                    # 处理 "20240312" 格式
                    (r'(\d{4})(\d{2})(\d{2})', r'\2/\3/\1'),  # 转换为 MM/DD/YYYY
                    # 处理 "Mar 12, 2024" 格式
                    (r'([A-Za-z]{3})\s+(\d{1,2}),\s*(\d{4})', lambda m: f"{m.group(2)}/{months[m.group(1).lower()]}/{m.group(3)}"),
                    # 处理 "12-Mar-2024" 格式
                    (r'(\d{1,2})-([A-Za-z]{3})-(\d{4})', lambda m: f"{m.group(1)}/{months[m.group(2).lower()]}/{m.group(3)}"),
                ]
                
                # 月份名称映射
                months = {
                    'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
                    'may': '05', 'jun': '06', 'jul': '07', 'aug': '08',
                    'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'
                }
                
                # 应用模式
                for pattern, replacement in patterns:
                    if re.match(pattern, value):
                        value = re.sub(pattern, replacement, value)
                        break
                
                # 尝试转换为日期格式
                try:
                    # 如果是数字格式的日期（如 20240312）
                    if value.isdigit() and len(value) == 8:
                        value = f"{value[4:6]}/{value[6:]}/{value[:4]}"
                    return value
                except:
                    return value
                    
            except Exception as e:
                logger.warning(f"Error cleaning datetime value '{value}': {str(e)}")
                return value
        
        # 处理日期列
        date_columns = [col for col in df.columns 
                       if any(date_term in col.lower() 
                             for date_term in ['date', 'time', 'day', 'year', 'month'])]
        
        for col in date_columns:
            logger.info(f"Cleaning datetime values in column: {col}")
            df[col] = df[col].apply(clean_datetime)
        
        # 处理数值列中的特殊字符
        numeric_cleaner = lambda x: str(x).replace('$', '').replace(',', '') if pd.notna(x) else x
        
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            df[col] = df[col].apply(numeric_cleaner)
            
        # 处理百分比
        def clean_percentage(value: Any) -> Any:
            if pd.isna(value):
                return value
            try:
                value = str(value)
                if '%' in value:
                    return float(value.replace('%', '')) / 100
                return value
            except:
                return value
                
        percentage_columns = [col for col in df.columns 
                            if any(term in col.lower() 
                                  for term in ['percent', 'percentage', 'rate', '%'])]
        
        for col in percentage_columns:
            df[col] = df[col].apply(clean_percentage)
            
        logger.info("Successfully cleaned special values in DataFrame")
        return df
        
    except Exception as e:
        logger.error(f"Error in clean_special_values: {str(e)}")
        raise

def infer_and_convert_data_types(df):
    """增强的数据类型推断函数"""

    df = clean_special_values(df)


    for column in df.columns:
        # 获取非空值
        non_null_values = df[column].dropna()
        if len(non_null_values) == 0:
            df[column] = df[column].astype(str)
            continue

        # 检查列名是否暗示了数据类型
        col_lower = column.lower()
        
        # 名称相关的列通常是字符串
        if any(name in col_lower for name in ['name', 'title', 'label', 'id']):
            df[column] = df[column].astype(str)
            continue
            
        # 日期相关的列
        if any(date in col_lower for date in ['date', 'time', 'year', 'month', 'day']):
            try:
                df[column] = pd.to_datetime(df[column], format='%d/%m/%Y', errors='coerce')
                continue
            except:
                pass

        # 检查是否为成绩（grade）
        if 'grade' in col_lower:
            # 检查是否所有值都是单个字母或有限的分类值
            unique_values = non_null_values.unique()
            if len(unique_values) <= 5 and all(len(str(x).strip()) <= 2 for x in unique_values):
                df[column] = pd.Categorical(df[column])
                continue
            else:
                df[column] = df[column].astype(str)
                continue

        # 检查是否为布尔值
        if ('is_' in col_lower or 
            all(str(x).lower() in ['true', 'false', '1', '0', 'yes', 'no'] 
                for x in df[column].dropna())):
            try:
                bool_map = {
                    'true': True, 'false': False,
                    '1': True, '0': False,
                    'yes': True, 'no': False,
                    1: True, 0: False
                }
                df[column] = df[column].map(bool_map)
                continue
            except:
                pass

        # 尝试转换为数值类型
        if all(str(x).replace('.', '').isdigit() or str(x).lower() in ['nan', 'not available', 'n/a', ''] 
               for x in non_null_values):
            try:
                if non_null_values.astype(str).str.contains('\.').any():
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                else:
                    df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                continue
            except:
                pass

        # 检查是否应该为分类型（对于其他列）
        unique_ratio = len(non_null_values.unique()) / len(non_null_values)
        if unique_ratio < 0.5:  # 如果唯一值比例小于50%
            # 额外检查：确保唯一值的数量不太大
            if len(non_null_values.unique()) <= 10:  # 假设分类值不应超过10个
                df[column] = pd.Categorical(df[column])
                continue

        # 默认保持为字符串类型
        df[column] = df[column].astype(str)

    return df



