# utils.py
import pandas as pd
import uuid
from django.core.cache import cache
import pickle
import numpy as np
import re
import logging
from typing import Dict, List, Any
import spacy
from typing import Optional, Dict, Any, List, Tuple


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpacyModelCache:
    _instance = None
    _model = None

    @classmethod
    def get_model(cls) -> spacy.language.Language:
        """获取或加载spaCy模型的单例实例"""
        if cls._model is None:
            try:
                cls._model = spacy.load('en_core_web_md')
                logger.info("Successfully loaded spaCy model")
            except Exception as e:
                logger.error(f"Error loading spaCy model: {str(e)}")
                raise
        return cls._model

    @classmethod
    def analyze_complex_data(cls, 
                           values: List[Any], 
                           column_name: str) -> Tuple[str, float]:
        """
        分析复杂数据并推断类型
        
        Args:
            values: 要分析的数据列表
            column_name: 列名
            
        Returns:
            Tuple[str, float]: (推断的类型, 置信度)
        """
        try:
            model = cls.get_model()
            predictions = []
            confidence_scores = []
            
            # 取前10个非空值进行分析
            sample_values = [str(v) for v in values[:10] if pd.notna(v)]
            
            for value in sample_values:
                doc = model(value)
                
                # 获取实体类型
                if doc.ents:
                    ent = doc.ents[0]
                    predictions.append(cls._map_entity_to_type(ent.label_))
                    confidence_scores.append(ent._.trf_score if hasattr(ent._, 'trf_score') else 0.5)
                else:
                    # 使用词性标注作为后备
                    pos_type = cls._get_type_from_pos(doc)
                    if pos_type:
                        predictions.append(pos_type)
                        confidence_scores.append(0.3)  # 较低的置信度
            
            if predictions:
                # 获取最常见的预测类型
                most_common = max(set(predictions), key=predictions.count)
                confidence = sum(c for p, c in zip(predictions, confidence_scores) 
                               if p == most_common) / len(predictions)
                return most_common, confidence
            
            return 'text', 0.0  # 默认返回文本类型
            
        except Exception as e:
            logger.warning(f"Error in complex data analysis: {str(e)}")
            return 'text', 0.0
            
    @staticmethod
    def _map_entity_to_type(ent_label: str) -> str:
        """将spaCy实体类型映射到数据类型"""
        type_mapping = {
            'DATE': 'datetime',
            'TIME': 'datetime',
            'CARDINAL': 'number',
            'MONEY': 'number',
            'PERCENT': 'number',
            'QUANTITY': 'number',
            'ORDINAL': 'number'
        }
        return type_mapping.get(ent_label, 'text')
        
    @staticmethod
    def _get_type_from_pos(doc: spacy.tokens.Doc) -> Optional[str]:
        """从词性标注推断类型"""
        # 获取主要词性
        main_pos = doc[0].pos_
        
        if main_pos in ['NUM']:
            return 'number'
        elif main_pos in ['PROPN', 'NOUN']:
            return 'category'
            
        return None

# utils.py
def is_complex_data(values: List[Any], column_name: str) -> bool:
    """
    判断是否为复杂数据
    
    Args:
        values: 要检查的数据列表
        column_name: 列名
        
    Returns:
        bool: 是否为复杂数据
    """
    try:
        # 获取非空值
        clean_values = [v for v in values if pd.notna(v)]
        if not clean_values:
            return False
            
        # 检查是否存在混合类型
        value_types = set()
        for value in clean_values[:10]:  # 只检查前10个值
            # 如果是pandas的Timestamp类型
            if isinstance(value, pd.Timestamp):
                value_types.add('date')
            # 如果是数字类型
            elif isinstance(value, (int, float)):
                value_types.add('number')
            elif isinstance(value, str):
                # 如果是字符串，检查它是否可以转换为其他类型
                # 尝试转换为数字
                try:
                    float(value)
                    value_types.add('number')
                    continue
                except ValueError:
                    pass
                
                # 尝试转换为日期
                try:
                    pd.to_datetime(value)
                    value_types.add('date')
                    continue
                except:
                    pass
                
                # 检查是否是货币格式 (例如: $1,234.56)
                if re.match(r'^\$?\d{1,3}(,\d{3})*(\.\d+)?$', value):
                    value_types.add('currency')
                    continue
                    
                # 检查是否是百分比格式
                if re.match(r'^\d+(\.\d+)?%$', value):
                    value_types.add('percentage')
                    continue
                
                # 其他情况认为是文本
                value_types.add('text')
        
        # 如果存在多种类型，或者包含特殊类型（货币、百分比），认为是复杂数据
        special_types = {'currency', 'percentage'}
        return len(value_types) > 1 or bool(value_types.intersection(special_types))
            
    except Exception as e:
        logger.warning(f"Error checking complex data: {str(e)}")
        return False





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
        # print("!!!!")
        # print(preview_data)
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

# utils.py 中修改 infer_and_convert_data_types 函数

def infer_and_convert_data_types(df):
    """增强的数据类型推断函数"""
    # 首先清理特殊值
    df = clean_special_values(df)
    
    for column in df.columns:
        try:
            # 获取非空值
            non_null_values = df[column].dropna().tolist()
            if len(non_null_values) == 0:
                df[column] = df[column].astype(str)
                continue

            # 检查是否为复杂数据
            if is_complex_data(non_null_values, column):
                logger.info(f"Complex data detected in column {column}, using spaCy model")
                inferred_type, confidence = SpacyModelCache.analyze_complex_data(
                    non_null_values, column
                )
                
                if confidence > 0.5:  # 只在置信度足够高时使用模型推断结果
                    logger.info(f"Using model inference for {column}: {inferred_type} (confidence: {confidence:.2f})")
                    try:
                        if inferred_type == 'datetime':
                            df[column] = pd.to_datetime(df[column], errors='coerce')
                        elif inferred_type == 'number':
                            df[column] = pd.to_numeric(df[column], errors='coerce')
                        elif inferred_type == 'category':
                            df[column] = pd.Categorical(df[column])
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to convert {column} using model inference: {str(e)}")
                else:
                    logger.info(f"Low confidence ({confidence:.2f}) for {column}, falling back to rule-based inference")

            # 以下是原有的规则基础推断逻辑
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

            # Grade列处理
            if 'grade' in col_lower:
                unique_values = pd.Series(non_null_values).unique()
                if len(unique_values) <= 5 and all(len(str(x).strip()) <= 2 for x in unique_values):
                    df[column] = pd.Categorical(df[column])
                    continue
                else:
                    df[column] = df[column].astype(str)
                    continue

            # 布尔值检查
            if ('is_' in col_lower or 
                all(str(x).lower() in ['true', 'false', '1', '0', 'yes', 'no'] 
                    for x in non_null_values)):
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

            # 数值类型检查
            try:
                if all(str(x).replace('.', '').isdigit() or str(x).lower() in ['nan', 'not available', 'n/a', ''] 
                       for x in non_null_values):
                    if any('.' in str(x) for x in non_null_values):
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    else:
                        df[column] = pd.to_numeric(df[column], errors='coerce', downcast='integer')
                    continue
            except:
                pass

            # 分类类型检查
            unique_ratio = len(pd.Series(non_null_values).unique()) / len(non_null_values)
            if unique_ratio < 0.5 and len(pd.Series(non_null_values).unique()) <= 10:
                df[column] = pd.Categorical(df[column])
                continue

            # 默认为字符串类型
            df[column] = df[column].astype(str)

        except Exception as e:
            logger.error(f"Error processing column {column}: {str(e)}")
            df[column] = df[column].astype(str)  # 发生错误时默认转为字符串类型

    return df


